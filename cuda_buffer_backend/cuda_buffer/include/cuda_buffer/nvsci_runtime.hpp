// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CUDA_BUFFER__NVSCI_RUNTIME_HPP_
#define CUDA_BUFFER__NVSCI_RUNTIME_HPP_

#include <cuda_runtime.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <sys/types.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"
#include "cuda_buffer/visibility_control.h"

#define NVSCI_CHECK(expr) \
  do { \
    NvSciError _nvsci_err = (expr); \
    if (_nvsci_err != NvSciError_Success) { \
      throw ::cuda_buffer_backend::CudaError( \
              std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #expr \
              " failed: NvSciError " + std::to_string(static_cast<int>(_nvsci_err))); \
    } \
  } while (0)

namespace cuda_buffer_backend
{

class NvSciBuffer;
class NvSciPeer;

namespace detail
{
class NvSciLedger;
}  // namespace detail

/// \brief An NvSciSyncObj together with the CUDA semaphore imported from it.
///
/// The NvSci counterpart of VmmBuffer's write event, with one difference: a
/// CUDA event belongs to one buffer, while one sync object orders the writes
/// of every buffer that shares it. Each write yields its own NvSciSyncFence.
struct CUDA_BUFFER_PUBLIC NvSciSyncChannel
{
  /// \brief Take ownership of \p sync_obj and import it into CUDA.
  NvSciSyncChannel(uint32_t channel_id, NvSciSyncObj sync_obj);
  ~NvSciSyncChannel();

  NvSciSyncChannel(const NvSciSyncChannel &) = delete;
  NvSciSyncChannel & operator=(const NvSciSyncChannel &) = delete;

  /// Identifies the channel to a peer; unique within the exporting process.
  const uint32_t id;
  NvSciSyncObj obj{nullptr};
  cudaExternalSemaphore_t semaphore{nullptr};
  /// CUDA device current at construction, which the semaphore belongs to.
  int device{-1};
  /// Producer side: the peer connections (NvSciPeer sessions) the object was
  /// reconciled with, and so the only ones it can be exported to.
  std::set<uint64_t> sessions;
};

/// \brief An NvSciBufObj mapped into this process's CUDA address space.
struct CUDA_BUFFER_PUBLIC NvSciMapping
{
  /// \brief Take ownership of \p buf_obj and map \p byte_size bytes of it.
  NvSciMapping(NvSciBufObj buf_obj, size_t byte_size);
  ~NvSciMapping();

  NvSciMapping(const NvSciMapping &) = delete;
  NvSciMapping & operator=(const NvSciMapping &) = delete;

  NvSciBufObj obj{nullptr};
  cudaExternalMemory_t memory{nullptr};
  uint8_t * ptr{nullptr};
  size_t size{0};
  /// CUDA device current at construction, which the mapping belongs to.
  int device{-1};
};

/// \brief A pooled NvSci allocation of this process: the counterpart of a VmmBlock.
///
/// Every importer, local or remote, holds a reference in the shared
/// IPCMetadata while it reads the block. The pool reuses the block once that
/// count is zero and every subscriber a descriptor of it was sent to has
/// imported it or can no longer (see NvSciLedger). The per-publish uid lets a
/// late importer tell that the block has been reused.
struct CUDA_BUFFER_PUBLIC NvSciBlock
{
  NvSciBlock(uint64_t block_id, std::unique_ptr<NvSciMapping> block_mapping);
  ~NvSciBlock();

  NvSciBlock(const NvSciBlock &) = delete;
  NvSciBlock & operator=(const NvSciBlock &) = delete;

  /// \brief Name of the shared-memory lease of block \p block_id of process \p pid.
  static std::string lease_name(pid_t pid, uint64_t block_id);

  /// \brief Record the fence of the latest write, for importers in this process.
  void record_write(
    const std::shared_ptr<NvSciSyncChannel> & channel, const NvSciSyncFence & fence);
  /// \brief Duplicate the latest write's fence into \p fence.
  /// \return false if the block was not written since it was last allocated.
  bool dup_last_write(std::shared_ptr<NvSciSyncChannel> * channel, NvSciSyncFence * fence);
  void clear_last_write();

  const uint64_t id;
  const std::unique_ptr<NvSciMapping> mapping;
  /// Peer connections (NvSciPeer sessions) the buffer object was reconciled with.
  std::set<uint64_t> sessions;

  /// \brief Whether the lease is in shared memory, so other processes can take it.
  bool lease_shared() const {return lease_in_shm_;}

  IPCMetadata * lease{nullptr};
  /// Per-allocation uid published with every descriptor; 0 until first published.
  uint64_t current_uid{0};

private:
  std::string shm_name_;
  bool lease_in_shm_{false};
  std::mutex write_mutex_;
  std::shared_ptr<NvSciSyncChannel> last_channel_;
  NvSciSyncFence last_fence_ = NvSciSyncFenceInitializer;
  bool has_last_fence_{false};
};

/// \brief What a subscriber needs in order to map an NvSci buffer: the
/// NvSci part of a CudaBufferDescriptor, kept free of the message type.
struct NvSciExport
{
  /// Publisher process.
  pid_t pid{0};
  uint64_t buffer_id{0};
  /// Per-publish uid, checked against the lease to detect a reused block.
  uint64_t uid{0};
  uint32_t sync_channel_id{0};
  bool has_fence{false};
  /// NvSciSyncFenceIpcExportDescriptor, for a subscriber in another process.
  std::array<uint8_t, sizeof(NvSciSyncFenceIpcExportDescriptor)> fence{};
  /// The publisher's id of the NvSciIpc connection the objects were sent on;
  /// 0 within a process.
  uint64_t connection{0};
  /// The publisher's stream for this subscriber, and this descriptor's place
  /// in it. The subscriber reports them back once it has handled it.
  uint32_t stream{0};
  uint64_t seq{0};
};

/// \brief The subscriber endpoint a descriptor is for.
struct NvSciSubscriber
{
  /// Its process.
  pid_t pid{0};
  /// Identifies the endpoint, such as its GID bytes.
  std::string key;
  /// Depth of its KEEP_LAST history, or 0 if unknown (or KEEP_ALL).
  uint32_t history_depth{0};
};

/// \brief Process-wide NvSci state: modules, the CUDA device's identity, the
/// block pool, the sync object local writes are signaled on, and the NvSciIpc
/// channels to peer processes.
///
/// The counterpart of CudaMemoryPool plus CudaVmmIPCManager. Like those it is
/// never destroyed, so it cannot be torn down after CUDA.
///
/// Sharing with another process needs an NvSciIpc INTER_PROCESS endpoint pair
/// for each pair of processes. Pairs are declared in /etc/nvsciipc.cfg, and
/// the host's endpoint pool inventory (CUDA_BUFFER_NVSCI_ENDPOINT_POOL, by
/// default /etc/cuda_buffer/nvsci_endpoint_pool.json) lists those every
/// process on the host may claim, through a shared-memory region. Without an
/// inventory, or where NvSciIpc cannot be initialized, NvSci buffers are still
/// shared within the process and other processes get the CPU path.
///
/// The runtime serves one CUDA device: the one current on the thread that
/// first calls get(). It makes that device current around its own CUDA calls,
/// so callers on other devices still get buffers of that device.
///
/// Reuse is tuned by CUDA_BUFFER_NVSCI_HOLD_MS, how long a block stays
/// reserved for a subscriber that neither imports it nor is known to have
/// dropped it (default 2000), and CUDA_BUFFER_NVSCI_POOL_MAX_BYTES, past which
/// the pool takes back such reservations, oldest first, instead of growing
/// (default 0, no limit; blocks being read are never taken back).
class CUDA_BUFFER_PUBLIC NvSciRuntime
{
public:
  /// \throws CudaError if NvSci or its CUDA interop is unusable in this process.
  static NvSciRuntime & get();

  /// \brief Whether get() succeeds. Never throws; reports the cause once.
  static bool is_available() noexcept;

  /// \brief Whether buffers can be shared with other processes over NvSciIpc.
  /// \details The first call initializes NvSciIpc and starts the dispatcher
  /// thread. Never throws; reports the cause once.
  bool ipc_available() noexcept;

  int device_id() const {return device_id_;}
  NvSciBufModule buf_module() const {return buf_module_;}
  NvSciSyncModule sync_module() const {return sync_module_;}

  /// \brief Allocate \p byte_size bytes from the block pool.
  std::unique_ptr<NvSciBuffer> allocate(size_t byte_size);

  /// \brief Open the NvSciIpc channel to process \p pid, if it is not open.
  /// \details Asynchronous: the processes exchange attributes once both ends
  /// are open, and buffers allocated after that can be shared with \p pid.
  /// Both processes must call this. Never throws.
  void connect_peer(pid_t pid) noexcept;

  /// \brief Describe \p buffer for \p subscriber.
  /// \details Pushes the buffer and sync objects to the subscriber's process
  /// the first time, and reserves the block until the subscriber has handled
  /// the descriptor. Call it once per descriptor sent, in the order they are
  /// sent to that subscriber.
  /// \return false if \p buffer cannot be shared with that process (the
  /// caller then uses the CPU path).
  bool export_buffer(
    const NvSciBuffer & buffer, const NvSciSubscriber & subscriber, NvSciExport * out);

  /// \brief Map the buffer described by \p in.
  /// \details Whether or not it succeeds, tells the publisher that the
  /// descriptor has been handled.
  /// \throws CudaError if it cannot be mapped or the block has been reused.
  std::unique_ptr<NvSciBuffer> import_buffer(const NvSciExport & in, size_t byte_size);

  // -- Used by NvSciPeer --

  /// \brief Unreconciled attributes for this process's use of a raw buffer.
  /// \param byte_size 0 for a consumer, which does not dictate the size.
  NvSciBufAttrList create_buf_attrs(size_t byte_size, NvSciBufAttrValAccessPerm perm) const;

  /// \brief Unreconciled attributes for this process's CUDA device.
  /// \param cuda_flags cudaNvSciSyncAttrSignal or cudaNvSciSyncAttrWait.
  NvSciSyncAttrList create_sync_attrs(int cuda_flags) const;

  void on_peer_ready(pid_t pid);

  /// \brief Process \p pid handled descriptors: (stream, highest number) pairs.
  void on_acknowledged(pid_t pid, const std::vector<std::pair<uint32_t, uint64_t>> & acks);

private:
  NvSciRuntime();

  /// A process this one wants an NvSciIpc channel to, and when to try next.
  struct WantedPeer
  {
    std::chrono::steady_clock::time_point next_attempt;
    bool warned{false};
    /// Set once the endpoint of the last channel to this process is closed.
    /// An import can hold a dropped channel for a while, and the endpoint
    /// cannot be opened again, nor its pair released, until it closes.
    std::shared_ptr<std::atomic<bool>> endpoint_closed;
  };

  void recycle(const std::shared_ptr<NvSciBlock> & block) noexcept;
  std::shared_ptr<NvSciBlock> create_block(size_t aligned_size);
  /// Over the pool limit: take back the free block of at least \p aligned_size
  /// bytes whose oldest reservation is oldest. Called with pool_mutex_ held.
  std::shared_ptr<NvSciBlock> reclaim_locked(size_t aligned_size);
  void notify_retired(const std::vector<uint64_t> & block_ids);
  std::shared_ptr<NvSciSyncChannel> producer_channel();
  /// Sessions of ready peers whose attributes are not known to conflict.
  std::set<uint64_t> eligible_sessions_locked(const std::set<uint64_t> & conflicting) const;
  std::vector<std::shared_ptr<NvSciPeer>> peers_for_sessions_locked(
    const std::set<uint64_t> & sessions) const;
  std::shared_ptr<NvSciPeer> find_peer(pid_t pid);
  void open_peer(pid_t pid, WantedPeer * want);
  void open_wanted_peers();
  void drop_peer(pid_t pid);
  void dispatch();

  NvSciBufModule buf_module_{nullptr};
  NvSciSyncModule sync_module_{nullptr};
  int device_id_{0};
  NvSciRmGpuId gpu_id_{};

  std::atomic<uint64_t> next_block_id_{1};
  std::atomic<uint32_t> next_channel_id_{1};
  std::atomic<uint64_t> next_session_{1};

  /// Which subscribers still owe an import of each block. Its lock is taken
  /// after pool_mutex_, never before.
  const std::unique_ptr<detail::NvSciLedger> ledger_;
  /// Limit of pool_bytes_ past which reservations are taken back; 0 for none.
  const size_t pool_max_bytes_;

  // Pool: guarded by pool_mutex_.
  std::mutex pool_mutex_;
  std::map<size_t, std::vector<std::shared_ptr<NvSciBlock>>> free_blocks_;
  std::unordered_map<uint64_t, std::shared_ptr<NvSciBlock>> all_blocks_;
  /// Bytes of every block in all_blocks_.
  size_t pool_bytes_{0};
  std::chrono::steady_clock::time_point last_pool_warning_{};

  // Peers and the producer channel: guarded by peers_mutex_. Lock order:
  // pool_mutex_, then peers_mutex_, then a peer's own mutexes.
  std::mutex peers_mutex_;
  std::unordered_map<pid_t, std::shared_ptr<NvSciPeer>> peers_;
  std::set<uint64_t> ready_sessions_;
  /// Sessions whose buffer or sync attributes do not reconcile with this
  /// process's; they are left out instead of failing every allocation.
  std::set<uint64_t> conflicting_buf_sessions_;
  std::set<uint64_t> conflicting_sync_sessions_;
  std::shared_ptr<NvSciSyncChannel> producer_channel_;
  /// Processes to keep a channel open to. The dispatcher opens (and reopens)
  /// channels, so publish and discovery never wait on the claim table.
  std::unordered_map<pid_t, WantedPeer> wanted_peers_;

  std::once_flag ipc_once_;
  bool ipc_available_{false};
  int epoll_fd_{-1};
  int wake_fd_{-1};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__NVSCI_RUNTIME_HPP_
