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

// Private to the cuda_buffer library.

#ifndef NVSCI_PEER_HPP_
#define NVSCI_PEER_HPP_

#include <nvscibuf.h>
#include <nvsciipc.h>
#include <nvscisync.h>
#include <sys/types.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda_buffer/nvsci_runtime.hpp"

namespace cuda_buffer_backend
{

/// \brief A block of a publisher in another process, mapped into this one.
struct NvSciImportedBlock
{
  NvSciImportedBlock(std::unique_ptr<NvSciMapping> block_mapping, IPCMetadata * block_lease);
  ~NvSciImportedBlock();

  NvSciImportedBlock(const NvSciImportedBlock &) = delete;
  NvSciImportedBlock & operator=(const NvSciImportedBlock &) = delete;

  const std::unique_ptr<NvSciMapping> mapping;
  /// The publisher's lease, mapped from its shared memory.
  IPCMetadata * const lease;
};

/// \brief The NvSciIpc channel between this process and one peer process.
///
/// A VMM descriptor names a block that any subscriber can fetch whenever it
/// likes. NvSci objects can only be exported *to an NvSciIpc endpoint*, and
/// imported by the process at the other end of it, so sharing needs a
/// channel per pair of processes. Both processes are producers and consumers
/// of each other. The messages are:
///
///   Hello        this process's unreconciled waiter attributes. NvSci
///                refuses to export an object to a peer whose attributes did
///                not take part in its reconciliation.
///   SyncChannel  an exported sync object, pushed once per channel.
///   Block        an exported buffer object, pushed once per block.
///   Retire       the publisher destroyed a block; drop the mapping.
///   Nack         the subscriber could not import a block or sync object;
///                the publisher stops sending descriptors that need it.
///   Ack          the subscriber has handled every descriptor of one of the
///                publisher's streams up to a number (see NvSciLedger).
///                Coalesced: one is queued per stream at a time.
///
/// Each side picks a random connection id and sends it in its hello. The ROS
/// descriptor carries the publisher's, with ids, the per-publish uid, the
/// fence and the descriptor's place in its stream; a subscriber refuses a
/// descriptor of another connection at once, since the objects it names were
/// sent on that one. Messages may be larger than an NvSciIpc frame and are
/// fragmented.
///
/// Endpoint reads and writes happen under one mutex; the dispatcher thread
/// calls service(), and publish and subscribe paths call the export and
/// wait functions.
class NvSciPeer
{
public:
  /// \brief Take ownership of the open endpoint \p endpoint, which leads to \p pid.
  /// \param session identifies this connection; a reconnect gets a new one.
  /// \param on_closed runs once the endpoint has been closed.
  NvSciPeer(
    NvSciRuntime & runtime, pid_t pid, uint64_t session, NvSciIpcEndpoint endpoint,
    std::function<void()> on_closed);
  ~NvSciPeer();

  NvSciPeer(const NvSciPeer &) = delete;
  NvSciPeer & operator=(const NvSciPeer &) = delete;

  pid_t pid() const {return pid_;}
  uint64_t session() const {return session_;}
  /// This side's random id of the connection, sent in its hello.
  uint64_t connection_id() const {return connection_id_;}
  int event_fd() const {return event_fd_;}

  /// \brief Handle every pending endpoint event: connection, reads, writes.
  void service();

  /// \brief The connection was reset: the peer closed its end or went away.
  bool lost() const;

  /// \brief Connected, and the peer's waiter attributes have arrived.
  bool ready() const;

  /// \brief End every current and future wait on this peer at once.
  /// \details Called when the connection is dropped: nothing more will
  /// arrive on it, and waiters should let go of the peer so that its endpoint
  /// can close.
  void shutdown();

  // -- Producer side --

  /// \brief Append the peer's waiter attribute lists. Requires ready().
  /// \details The lists stay valid for as long as the caller holds this peer:
  /// they are set once, by the peer's only hello, and freed with the peer.
  void append_waiter_attrs(
    std::vector<NvSciBufAttrList> * buf_lists,
    std::vector<NvSciSyncAttrList> * sync_lists) const;

  /// \brief Push \p channel to the peer, unless it has it already.
  void export_channel(const NvSciSyncChannel & channel);

  /// \brief Push \p block to the peer, unless it has it already.
  void export_block(const NvSciBlock & block);

  void export_fence(const NvSciSyncFence & fence, NvSciSyncFenceIpcExportDescriptor * out);

  /// \brief False once the peer reported that it cannot import \p block_id.
  bool accepts_block(uint64_t block_id) const;
  /// \brief False once the peer reported that it cannot import \p channel_id.
  bool accepts_channel(uint32_t channel_id) const;

  /// \brief Tell the peer that \p block_id is gone, if it was pushed to it.
  void retire_block(uint64_t block_id);

  // -- Consumer side --

  enum class WaitResult
  {
    kReady,
    /// The descriptor came from another connection to the same process.
    kWrongConnection,
    /// The objects will never arrive: the import failed, the block was
    /// retired, or the connection was dropped.
    kGone,
    kTimedOut,
  };

  /// \brief Wait until the block and, if \p need_channel, the sync object
  /// named by a descriptor of connection \p connection have arrived.
  /// \details One wait for both; it ends early whenever they cannot arrive.
  WaitResult wait_objects(
    uint64_t connection, uint64_t block_id, bool need_channel, uint32_t channel_id,
    std::chrono::milliseconds timeout,
    std::shared_ptr<NvSciImportedBlock> * block,
    std::shared_ptr<NvSciSyncChannel> * channel);

  /// \brief Tell the publisher that every descriptor of \p stream up to
  /// \p seq has been handled.
  void acknowledge(uint32_t stream, uint64_t seq) noexcept;

private:
  struct Message
  {
    uint32_t type;
    uint64_t id;
    uint64_t arg;
    std::vector<uint8_t> payload;
  };

  void enqueue_locked(uint32_t type, uint64_t id, uint64_t arg, std::vector<uint8_t> payload);
  void flush_locked();
  void read_locked(std::vector<Message> * inbound);
  void queue_hello_locked();
  /// \brief Handle one message from the peer. Called with io_mutex_ held.
  /// \return true if it was the peer's hello.
  bool handle(const Message & message);
  /// \brief Remember that an object could not be imported, and tell the peer.
  void reject_locked(const Message & message);
  /// \brief Queue the acknowledgements waiting in pending_acks_.
  void send_acks_locked();

  NvSciRuntime & runtime_;
  const pid_t pid_;
  const uint64_t session_;
  const uint64_t connection_id_;
  const NvSciIpcEndpoint endpoint_;
  std::function<void()> on_closed_;
  int event_fd_{-1};
  size_t frame_size_{0};

  // Endpoint I/O. Guarded by io_mutex_.
  mutable std::mutex io_mutex_;
  bool connected_{false};
  bool lost_{false};
  std::deque<std::vector<uint8_t>> outbox_;
  std::vector<uint8_t> inbox_;
  std::unordered_set<uint32_t> exported_channels_;
  std::unordered_set<uint64_t> exported_blocks_;
  std::unordered_set<uint64_t> rejected_blocks_;
  std::unordered_set<uint32_t> rejected_channels_;
  /// Highest number to acknowledge per stream, not queued yet.
  std::unordered_map<uint32_t, uint64_t> pending_acks_;

  // What the processes told each other. Guarded by state_mutex_.
  mutable std::mutex state_mutex_;
  std::condition_variable state_cv_;
  NvSciBufAttrList local_buf_waiter_attrs_{nullptr};
  NvSciSyncAttrList local_sync_waiter_attrs_{nullptr};
  NvSciBufAttrList peer_buf_waiter_attrs_{nullptr};
  NvSciSyncAttrList peer_sync_waiter_attrs_{nullptr};
  /// The peer's connection id, from its hello; 0 until then.
  uint64_t peer_connection_id_{0};
  std::unordered_map<uint32_t, std::shared_ptr<NvSciSyncChannel>> imported_channels_;
  std::unordered_map<uint64_t, std::shared_ptr<NvSciImportedBlock>> imported_blocks_;
  // Objects this process failed to import, so that waits for them end at once.
  std::unordered_set<uint32_t> failed_channels_;
  std::unordered_set<uint64_t> failed_blocks_;
  // Blocks the publisher destroyed. Block ids are never reused within a
  // process, so a wait for one of these would never end otherwise.
  std::unordered_set<uint64_t> retired_blocks_;
  bool shut_down_{false};
};

}  // namespace cuda_buffer_backend

#endif  // NVSCI_PEER_HPP_
