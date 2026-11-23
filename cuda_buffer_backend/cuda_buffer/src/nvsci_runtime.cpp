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

#include "cuda_buffer/nvsci_runtime.hpp"

#include <cuda.h>
#include <fcntl.h>
#include <nvsciipc.h>
#include <rcutils/logging_macros.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cuda_buffer/nvsci_buffer.hpp"
#include "nvsci_endpoint_pool.hpp"
#include "nvsci_ledger.hpp"
#include "nvsci_peer.hpp"

namespace cuda_buffer_backend
{

namespace
{

/// Alignment and size granularity of pooled raw buffers.
constexpr uint64_t kBlockAlign = 4096;

/// How long an importer waits for the publisher's objects to arrive over
/// NvSciIpc after the descriptor arrived over the RMW. They were queued before
/// the descriptor was published, and the wait ends at once when they cannot
/// arrive, so this only covers delivery and the dispatcher's wakeup. A late
/// import costs latency, not the message: the block stays reserved for it.
constexpr std::chrono::milliseconds kImportWait{250};

/// How often the dispatcher looks at every peer even without an event.
constexpr int kDispatchTickMs = 100;

/// Default of CUDA_BUFFER_NVSCI_HOLD_MS.
constexpr uint64_t kDefaultHoldMs = 2000;

/// How often the pool repeats a warning about its limit.
constexpr std::chrono::seconds kPoolWarningInterval{5};

/// A non-negative integer from the environment, or \p fallback.
uint64_t env_u64(const char * name, uint64_t fallback)
{
  const char * value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char * end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);  // NOLINT(runtime/int)
  if (errno != 0 || end == value || *end != '\0' || value[0] == '-') {
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend", "Ignoring %s='%s': expected a non-negative integer", name, value);
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

std::string pid_list(const std::vector<pid_t> & pids)
{
  std::string list;
  for (pid_t pid : pids) {
    list += (list.empty() ? "" : ", ") + std::to_string(pid);
  }
  return list;
}

/// Makes a CUDA device current for the scope, then restores the caller's.
///
/// NvSci objects belong to the device of the runtime, while callers may have
/// any device current; CUDA imports and frees them in the current device's
/// context.
class ScopedDevice
{
public:
  /// \param required throw CudaError if \p device cannot be made current,
  /// instead of carrying on with the caller's device.
  ScopedDevice(int device, bool required)
  {
    if (device < 0) {
      return;
    }
    cudaError_t err = cudaGetDevice(&previous_);
    if (err == cudaSuccess && previous_ != device) {
      err = cudaSetDevice(device);
      switched_ = err == cudaSuccess;
    }
    if (err != cudaSuccess) {
      (void)cudaGetLastError();
      if (required) {
        throw CudaError(__FILE__, __LINE__, "cudaSetDevice", err);
      }
    }
  }

  ~ScopedDevice()
  {
    if (switched_ && cudaSetDevice(previous_) != cudaSuccess) {
      (void)cudaGetLastError();
    }
  }

  ScopedDevice(const ScopedDevice &) = delete;
  ScopedDevice & operator=(const ScopedDevice &) = delete;

private:
  int previous_{-1};
  bool switched_{false};
};

/// Device current on this thread, or -1.
int current_device() noexcept
{
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    return -1;
  }
  return device;
}


bool process_alive(pid_t pid)
{
  return !(kill(pid, 0) != 0 && errno == ESRCH);
}

/// Frees an NvSci attribute list at scope exit.
template<typename List, void (*Free)(List)>
struct ScopedList
{
  List list{nullptr};
  ScopedList() = default;
  explicit ScopedList(List l)
  : list(l) {}
  ~ScopedList()
  {
    if (list) {Free(list);}
  }
  ScopedList(const ScopedList &) = delete;
  ScopedList & operator=(const ScopedList &) = delete;
};
using ScopedBufAttrs = ScopedList<NvSciBufAttrList, NvSciBufAttrListFree>;
using ScopedSyncAttrs = ScopedList<NvSciSyncAttrList, NvSciSyncAttrListFree>;

detail::NvSciEndpointPool * endpoint_pool()
{
  // Never destroyed, like the runtime that uses it.
  static detail::NvSciEndpointPool * const instance = new detail::NvSciEndpointPool();
  return instance;
}

}  // namespace

// -- NvSciSyncChannel --

NvSciSyncChannel::NvSciSyncChannel(uint32_t channel_id, NvSciSyncObj sync_obj)
: id(channel_id), obj(sync_obj), device(current_device())
{
  // One CUDA semaphore serves both to signal and to wait.
  cudaExternalSemaphoreHandleDesc desc{};
  desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
  desc.handle.nvSciSyncObj = obj;
  const cudaError_t err = cudaImportExternalSemaphore(&semaphore, &desc);
  if (err != cudaSuccess) {
    NvSciSyncObjFree(obj);
    (void)cudaGetLastError();
    throw CudaError(__FILE__, __LINE__, "cudaImportExternalSemaphore", err);
  }
}

NvSciSyncChannel::~NvSciSyncChannel()
{
  // The last reference may go on any thread, such as the deferred-release one.
  ScopedDevice on_device(device, false);
  if (semaphore) {
    CUDA_CHECK_NOTHROW(cudaDestroyExternalSemaphore(semaphore), (void)0);
  }
  if (obj) {
    NvSciSyncObjFree(obj);
  }
}

// -- NvSciMapping --

NvSciMapping::NvSciMapping(NvSciBufObj buf_obj, size_t byte_size)
: obj(buf_obj), size(byte_size), device(current_device())
{
  cudaExternalMemoryHandleDesc memory_desc{};
  memory_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
  memory_desc.handle.nvSciBufObject = obj;
  memory_desc.size = byte_size;
  cudaError_t err = cudaImportExternalMemory(&memory, &memory_desc);
  if (err != cudaSuccess) {
    NvSciBufObjFree(obj);
    (void)cudaGetLastError();
    throw CudaError(__FILE__, __LINE__, "cudaImportExternalMemory", err);
  }

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = byte_size;
  void * mapped = nullptr;
  err = cudaExternalMemoryGetMappedBuffer(&mapped, memory, &buffer_desc);
  if (err != cudaSuccess) {
    cudaDestroyExternalMemory(memory);
    NvSciBufObjFree(obj);
    (void)cudaGetLastError();
    throw CudaError(__FILE__, __LINE__, "cudaExternalMemoryGetMappedBuffer", err);
  }
  ptr = static_cast<uint8_t *>(mapped);
}

NvSciMapping::~NvSciMapping()
{
  // The last reference may go on any thread, such as the deferred-release one.
  ScopedDevice on_device(device, false);
  if (ptr) {
    CUDA_CHECK_NOTHROW(cudaFree(ptr), (void)0);
  }
  if (memory) {
    CUDA_CHECK_NOTHROW(cudaDestroyExternalMemory(memory), (void)0);
  }
  if (obj) {
    NvSciBufObjFree(obj);
  }
}

// -- NvSciBlock --

std::string NvSciBlock::lease_name(pid_t pid, uint64_t block_id)
{
  return "/cuda_nvsci_" + std::to_string(pid) + "_" + std::to_string(block_id);
}

NvSciBlock::NvSciBlock(uint64_t block_id, std::unique_ptr<NvSciMapping> block_mapping)
: id(block_id), mapping(std::move(block_mapping))
{
  // The lease lives in shared memory so that importers in other processes
  // can take references. Where that fails, an in-process lease still works
  // for subscribers in this process.
  shm_name_ = lease_name(getpid(), id);
  shm_unlink(shm_name_.c_str());
  const int fd = shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd >= 0) {
    void * ptr = MAP_FAILED;
    if (ftruncate(fd, sizeof(IPCMetadata)) == 0) {
      ptr = mmap(nullptr, sizeof(IPCMetadata), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    close(fd);
    if (ptr != MAP_FAILED) {
      lease = new(ptr) IPCMetadata();
      lease_in_shm_ = true;
    } else {
      shm_unlink(shm_name_.c_str());
    }
  }
  if (lease == nullptr) {
    lease = new IPCMetadata();
  }
}

NvSciBlock::~NvSciBlock()
{
  clear_last_write();
  if (lease_in_shm_) {
    munmap(lease, sizeof(IPCMetadata));
    shm_unlink(shm_name_.c_str());
  } else {
    delete lease;
  }
}

void NvSciBlock::record_write(
  const std::shared_ptr<NvSciSyncChannel> & channel, const NvSciSyncFence & fence)
{
  std::lock_guard<std::mutex> lock(write_mutex_);
  NvSciSyncFenceClear(&last_fence_);
  has_last_fence_ = false;
  NVSCI_CHECK(NvSciSyncFenceDup(&fence, &last_fence_));
  last_channel_ = channel;
  has_last_fence_ = true;
}

bool NvSciBlock::dup_last_write(
  std::shared_ptr<NvSciSyncChannel> * channel, NvSciSyncFence * fence)
{
  std::lock_guard<std::mutex> lock(write_mutex_);
  if (!has_last_fence_) {
    return false;
  }
  NVSCI_CHECK(NvSciSyncFenceDup(&last_fence_, fence));
  *channel = last_channel_;
  return true;
}

void NvSciBlock::clear_last_write()
{
  std::lock_guard<std::mutex> lock(write_mutex_);
  NvSciSyncFenceClear(&last_fence_);
  has_last_fence_ = false;
  last_channel_.reset();
}

// -- NvSciRuntime --

NvSciRuntime & NvSciRuntime::get()
{
  // Never destroyed: see the class comment. Construction that throws is
  // retried on the next call.
  static NvSciRuntime * const runtime = new NvSciRuntime();
  return *runtime;
}

bool NvSciRuntime::is_available() noexcept
{
  try {
    (void)get();
    return true;
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_ONCE_NAMED(
      "cuda_buffer_backend", "NvSci storage unavailable: %s", e.what());
    return false;
  }
}

NvSciRuntime::NvSciRuntime()
: ledger_(std::make_unique<detail::NvSciLedger>(
      std::chrono::milliseconds(env_u64("CUDA_BUFFER_NVSCI_HOLD_MS", kDefaultHoldMs)))),
  pool_max_bytes_(static_cast<size_t>(env_u64("CUDA_BUFFER_NVSCI_POOL_MAX_BYTES", 0)))
{
  CUDA_CHECK(cudaGetDevice(&device_id_));
  // Make sure the device has a context before asking the driver about it.
  const cudaError_t init = cudaFree(nullptr);
  if (init != cudaSuccess && init != cudaErrorInvalidValue) {
    throw CudaError(__FILE__, __LINE__, "cudaFree(nullptr)", init);
  }

  // NvSci names a GPU by UUID.
  CUdevice device = 0;
  CUresult r = cuDeviceGet(&device, device_id_);
  if (r != CUDA_SUCCESS) {
    throw CudaError(__FILE__, __LINE__, "cuDeviceGet", r);
  }
  CUuuid uuid;
  r = cuDeviceGetUuid(&uuid, device);
  if (r != CUDA_SUCCESS) {
    throw CudaError(__FILE__, __LINE__, "cuDeviceGetUuid", r);
  }
  static_assert(sizeof(gpu_id_.bytes) == sizeof(uuid.bytes), "GPU id size mismatch");
  std::memcpy(gpu_id_.bytes, uuid.bytes, sizeof(gpu_id_.bytes));

  NVSCI_CHECK(NvSciBufModuleOpen(&buf_module_));
  const NvSciError err = NvSciSyncModuleOpen(&sync_module_);
  if (err != NvSciError_Success) {
    NvSciBufModuleClose(buf_module_);
    NVSCI_CHECK(err);
  }

  // Check now that CUDA can import this device's NvSci objects, so that a
  // device without NvSci interop reports NvSci as unavailable instead of
  // failing on the first allocation.
  try {
    ScopedSyncAttrs probe(create_sync_attrs(cudaNvSciSyncAttrSignal));
  } catch (...) {
    NvSciSyncModuleClose(sync_module_);
    NvSciBufModuleClose(buf_module_);
    throw;
  }
}

bool NvSciRuntime::ipc_available() noexcept
{
  std::call_once(
    ipc_once_, [this] {
      try {
        const NvSciError err = NvSciIpcInit();
        if (err != NvSciError_Success) {
          RCUTILS_LOG_INFO_NAMED(
            "cuda_buffer_backend",
            "NvSciIpc unavailable (NvSciError %d): NvSci buffers are shared within this "
            "process only", static_cast<int>(err));
          return;
        }
        detail::NvSciEndpointPool * pairs = endpoint_pool();
        for (const std::string & warning : pairs->warnings()) {
          RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend", "%s", warning.c_str());
        }
        switch (pairs->state()) {
          case detail::NvSciEndpointPool::State::kReady:
            break;
          case detail::NvSciEndpointPool::State::kNoInventory:
            RCUTILS_LOG_INFO_NAMED(
              "cuda_buffer_backend",
              "No NvSci endpoint pool (%s): NvSci buffers are shared within this process only",
              pairs->error().c_str());
            return;
          case detail::NvSciEndpointPool::State::kError:
            RCUTILS_LOG_ERROR_NAMED(
              "cuda_buffer_backend",
              "%s: NvSci buffers are shared within this process only", pairs->error().c_str());
            return;
        }
        if (pairs->pair_count() == 0) {
          return;  // Warned above.
        }
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        wake_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (epoll_fd_ < 0 || wake_fd_ < 0) {
          throw CudaError(std::string("NvSci dispatcher: ") + std::strerror(errno));
        }
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = wake_fd_;
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wake_fd_, &ev);
        std::thread(&NvSciRuntime::dispatch, this).detach();
        ipc_available_ = true;
        RCUTILS_LOG_INFO_NAMED(
          "cuda_buffer_backend", "NvSci cross-process sharing enabled with %zu endpoint pairs",
          pairs->pair_count());
      } catch (const std::exception & e) {
        RCUTILS_LOG_WARN_NAMED(
          "cuda_buffer_backend", "NvSci cross-process sharing disabled: %s", e.what());
      }
    });
  return ipc_available_;
}

NvSciBufAttrList NvSciRuntime::create_buf_attrs(
  size_t byte_size, NvSciBufAttrValAccessPerm perm) const
{
  ScopedBufAttrs attrs;
  NVSCI_CHECK(NvSciBufAttrListCreate(buf_module_, &attrs.list));

  const NvSciBufType type = NvSciBufType_RawBuffer;
  const bool need_cpu_access = false;
  const uint64_t size = byte_size;
  const uint64_t align = kBlockAlign;
  std::vector<NvSciBufAttrKeyValuePair> pairs = {
    {NvSciBufGeneralAttrKey_Types, &type, sizeof(type)},
    {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
    {NvSciBufGeneralAttrKey_NeedCpuAccess, &need_cpu_access, sizeof(need_cpu_access)},
    {NvSciBufGeneralAttrKey_GpuId, &gpu_id_, sizeof(gpu_id_)},
  };
  if (byte_size != 0) {
    pairs.push_back({NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)});
    pairs.push_back({NvSciBufRawBufferAttrKey_Align, &align, sizeof(align)});
  }
  NVSCI_CHECK(NvSciBufAttrListSetAttrs(attrs.list, pairs.data(), pairs.size()));

  NvSciBufAttrList list = attrs.list;
  attrs.list = nullptr;
  return list;
}

NvSciSyncAttrList NvSciRuntime::create_sync_attrs(int cuda_flags) const
{
  ScopedSyncAttrs attrs;
  NVSCI_CHECK(NvSciSyncAttrListCreate(sync_module_, &attrs.list));
  CUDA_CHECK(cudaDeviceGetNvSciSyncAttributes(attrs.list, device_id_, cuda_flags));
  NvSciSyncAttrList list = attrs.list;
  attrs.list = nullptr;
  return list;
}

std::unique_ptr<NvSciBuffer> NvSciRuntime::allocate(size_t byte_size)
{
  // Blocks, and the sync object, are created and mapped on this runtime's
  // device whatever device the caller has current.
  ScopedDevice on_device(device_id_, true);
  const size_t aligned = ((byte_size + kBlockAlign - 1) / kBlockAlign) * kBlockAlign;

  std::set<uint64_t> sessions;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    sessions = eligible_sessions_locked(conflicting_buf_sessions_);
  }

  std::shared_ptr<NvSciBlock> block;
  std::vector<std::shared_ptr<NvSciBlock>> retired;
  bool over_limit = false;
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    const auto now = std::chrono::steady_clock::now();
    for (auto bucket = free_blocks_.lower_bound(aligned);
      bucket != free_blocks_.end() && !block; )
    {
      auto & blocks = bucket->second;
      for (size_t i = 0; i < blocks.size(); ) {
        IPCMetadata * lease = blocks[i]->lease;
        // Free once nobody reads it and nobody it was sent to still may.
        if (lease->refcount.load(std::memory_order_acquire) > 0 ||
          !ledger_->settled(blocks[i]->id, now))
        {
          ++i;
          continue;
        }
        // Clear the uid, then look at the count again. An importer increments
        // the count, then reads the uid. With both sides sequentially
        // consistent, either the importer sees the cleared uid and backs off,
        // or this sees its reference and leaves the block alone.
        lease->uid.store(0, std::memory_order_seq_cst);
        if (lease->refcount.load(std::memory_order_seq_cst) > 0) {
          ++i;
          continue;
        }
        std::shared_ptr<NvSciBlock> candidate = blocks[i];
        blocks[i] = blocks.back();
        blocks.pop_back();
        // A block can only be shared with the peers it was reconciled with.
        // Once more peers are ready, blocks that leave one out are replaced.
        if (!std::includes(
            candidate->sessions.begin(), candidate->sessions.end(),
            sessions.begin(), sessions.end()))
        {
          all_blocks_.erase(candidate->id);
          pool_bytes_ -= candidate->mapping->size;
          ledger_->forget(candidate->id);
          retired.push_back(std::move(candidate));
          continue;
        }
        block = std::move(candidate);
        break;
      }
      if (blocks.empty()) {
        bucket = free_blocks_.erase(bucket);
      } else {
        ++bucket;
      }
    }
    if (!block && pool_max_bytes_ != 0 && pool_bytes_ + aligned > pool_max_bytes_) {
      block = reclaim_locked(aligned);
      if (block && !std::includes(
          block->sessions.begin(), block->sessions.end(), sessions.begin(), sessions.end()))
      {
        // Same as above: replace it with a block every ready peer can import.
        all_blocks_.erase(block->id);
        pool_bytes_ -= block->mapping->size;
        retired.push_back(std::move(block));
        block.reset();
      }
      over_limit = !block && pool_bytes_ + aligned > pool_max_bytes_;
      if (over_limit) {
        if (now - last_pool_warning_ >= kPoolWarningInterval) {
          last_pool_warning_ = now;
        } else {
          over_limit = false;  // Warned recently.
        }
      }
    }
  }

  if (over_limit) {
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend",
      "NvSci pool exceeds CUDA_BUFFER_NVSCI_POOL_MAX_BYTES (%zu): no block large enough is "
      "free of readers, so it grows", pool_max_bytes_);
  }

  if (!retired.empty()) {
    std::vector<uint64_t> ids;
    for (const auto & r : retired) {
      ids.push_back(r->id);
    }
    retired.clear();  // Frees the blocks.
    notify_retired(ids);
  }

  if (block) {
    block->current_uid = 0;
    block->clear_last_write();
  } else {
    block = create_block(aligned);
  }

  std::shared_ptr<NvSciSyncChannel> channel = producer_channel();
  uint8_t * ptr = block->mapping->ptr;
  return std::make_unique<NvSciBuffer>(
    ptr, byte_size, device_id_, std::move(channel), true, block,
    [this, block]() {recycle(block);});
}

std::shared_ptr<NvSciBlock> NvSciRuntime::reclaim_locked(size_t aligned_size)
{
  // Only blocks nobody reads: a reservation is a guess that a subscriber may
  // still import, a reference is a reader.
  std::vector<std::shared_ptr<NvSciBlock>> * best_list = nullptr;
  size_t best_index = 0;
  std::chrono::steady_clock::time_point best_made{};
  for (auto bucket = free_blocks_.lower_bound(aligned_size); bucket != free_blocks_.end();
    ++bucket)
  {
    auto & blocks = bucket->second;
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (blocks[i]->lease->refcount.load(std::memory_order_acquire) > 0) {
        continue;
      }
      const auto made = ledger_->oldest(blocks[i]->id);
      if (made && (best_list == nullptr || *made < best_made)) {
        best_list = &blocks;
        best_index = i;
        best_made = *made;
      }
    }
  }
  if (best_list == nullptr) {
    return nullptr;
  }
  std::shared_ptr<NvSciBlock> block = (*best_list)[best_index];
  // As in allocate(): clear the uid, then make sure no importer got in.
  block->lease->uid.store(0, std::memory_order_seq_cst);
  if (block->lease->refcount.load(std::memory_order_seq_cst) > 0) {
    return nullptr;
  }
  (*best_list)[best_index] = best_list->back();
  best_list->pop_back();
  const std::vector<pid_t> pids = ledger_->reclaim(block->id);
  const auto now = std::chrono::steady_clock::now();
  if (now - last_pool_warning_ >= kPoolWarningInterval) {
    last_pool_warning_ = now;
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend",
      "NvSci pool is at CUDA_BUFFER_NVSCI_POOL_MAX_BYTES (%zu): reusing a block that "
      "process(es) %s did not import yet; they will drop that message",
      pool_max_bytes_, pid_list(pids).c_str());
  }
  return block;
}

std::set<uint64_t> NvSciRuntime::eligible_sessions_locked(
  const std::set<uint64_t> & conflicting) const
{
  std::set<uint64_t> sessions;
  for (uint64_t session : ready_sessions_) {
    if (conflicting.count(session) == 0) {
      sessions.insert(session);
    }
  }
  return sessions;
}

std::vector<std::shared_ptr<NvSciPeer>> NvSciRuntime::peers_for_sessions_locked(
  const std::set<uint64_t> & sessions) const
{
  std::vector<std::shared_ptr<NvSciPeer>> peers;
  for (const auto & entry : peers_) {
    if (sessions.count(entry.second->session()) != 0 && entry.second->ready()) {
      peers.push_back(entry.second);
    }
  }
  return peers;
}

std::shared_ptr<NvSciBlock> NvSciRuntime::create_block(size_t aligned_size)
{
  // The peers are held until reconciliation is done: their attribute lists
  // are freed with them.
  std::vector<std::shared_ptr<NvSciPeer>> peers;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    peers = peers_for_sessions_locked(eligible_sessions_locked(conflicting_buf_sessions_));
  }

  ScopedBufAttrs local(create_buf_attrs(aligned_size, NvSciBufAccessPerm_ReadWrite));
  auto reconcile = [&local](
    const std::vector<std::shared_ptr<NvSciPeer>> & with, ScopedBufAttrs * out) {
      std::vector<NvSciBufAttrList> lists{local.list};
      for (const auto & peer : with) {
        peer->append_waiter_attrs(&lists, nullptr);
      }
      ScopedBufAttrs conflicts;
      return NvSciBufAttrListReconcile(lists.data(), lists.size(), &out->list, &conflicts.list);
    };

  ScopedBufAttrs reconciled;
  NvSciError err = reconcile(peers, &reconciled);
  if (err != NvSciError_Success && !peers.empty()) {
    // Find the peers this process cannot share buffers with, leave them out
    // from now on, and allocate for the rest.
    std::vector<std::shared_ptr<NvSciPeer>> compatible;
    std::set<uint64_t> conflicting;
    for (const auto & peer : peers) {
      ScopedBufAttrs probe;
      if (reconcile({peer}, &probe) == NvSciError_Success) {
        compatible.push_back(peer);
      } else {
        conflicting.insert(peer->session());
        RCUTILS_LOG_WARN_NAMED(
          "cuda_buffer_backend",
          "NvSci buffer attributes of process %d do not reconcile with this process's; "
          "it gets the CPU path", static_cast<int>(peer->pid()));
      }
    }
    {
      std::lock_guard<std::mutex> lock(peers_mutex_);
      conflicting_buf_sessions_.insert(conflicting.begin(), conflicting.end());
    }
    peers = std::move(compatible);
    err = reconcile(peers, &reconciled);
  }
  NVSCI_CHECK(err);

  NvSciBufObj obj = nullptr;
  NVSCI_CHECK(NvSciBufObjAlloc(reconciled.list, &obj));
  auto block = std::make_shared<NvSciBlock>(
    next_block_id_.fetch_add(1), std::make_unique<NvSciMapping>(obj, aligned_size));
  for (const auto & peer : peers) {
    block->sessions.insert(peer->session());
  }

  std::lock_guard<std::mutex> lock(pool_mutex_);
  all_blocks_[block->id] = block;
  pool_bytes_ += block->mapping->size;
  return block;
}

void NvSciRuntime::recycle(const std::shared_ptr<NvSciBlock> & block) noexcept
{
  std::lock_guard<std::mutex> lock(pool_mutex_);
  free_blocks_[block->mapping->size].push_back(block);
}

void NvSciRuntime::notify_retired(const std::vector<uint64_t> & block_ids)
{
  std::vector<std::shared_ptr<NvSciPeer>> peers;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (auto & entry : peers_) {
      peers.push_back(entry.second);
    }
  }
  for (auto & peer : peers) {
    for (uint64_t id : block_ids) {
      peer->retire_block(id);
    }
  }
}

std::shared_ptr<NvSciSyncChannel> NvSciRuntime::producer_channel()
{
  // Reconciliation happens under peers_mutex_, which keeps the peers, and so
  // their attribute lists, alive.
  std::lock_guard<std::mutex> lock(peers_mutex_);
  std::set<uint64_t> sessions = eligible_sessions_locked(conflicting_sync_sessions_);
  std::vector<std::shared_ptr<NvSciPeer>> peers = peers_for_sessions_locked(sessions);
  sessions.clear();
  for (const auto & peer : peers) {
    sessions.insert(peer->session());
  }
  if (producer_channel_ && producer_channel_->sessions == sessions) {
    return producer_channel_;
  }

  // Signaled by local CUDA writes; waited on by local CUDA reads and by
  // every eligible peer.
  ScopedSyncAttrs signaler(create_sync_attrs(cudaNvSciSyncAttrSignal));
  ScopedSyncAttrs waiter(create_sync_attrs(cudaNvSciSyncAttrWait));
  auto reconcile = [&](
    const std::vector<std::shared_ptr<NvSciPeer>> & with, ScopedSyncAttrs * out) {
      std::vector<NvSciSyncAttrList> lists{signaler.list, waiter.list};
      for (const auto & peer : with) {
        peer->append_waiter_attrs(nullptr, &lists);
      }
      ScopedSyncAttrs conflicts;
      return NvSciSyncAttrListReconcile(lists.data(), lists.size(), &out->list, &conflicts.list);
    };

  ScopedSyncAttrs reconciled;
  NvSciError err = reconcile(peers, &reconciled);
  if (err != NvSciError_Success && !peers.empty()) {
    std::vector<std::shared_ptr<NvSciPeer>> compatible;
    for (const auto & peer : peers) {
      ScopedSyncAttrs probe;
      if (reconcile({peer}, &probe) == NvSciError_Success) {
        compatible.push_back(peer);
      } else {
        conflicting_sync_sessions_.insert(peer->session());
        sessions.erase(peer->session());
        RCUTILS_LOG_WARN_NAMED(
          "cuda_buffer_backend",
          "NvSci sync attributes of process %d do not reconcile with this process's; "
          "it gets the CPU path", static_cast<int>(peer->pid()));
      }
    }
    peers = std::move(compatible);
    err = reconcile(peers, &reconciled);
  }
  NVSCI_CHECK(err);
  NvSciSyncObj obj = nullptr;
  NVSCI_CHECK(NvSciSyncObjAlloc(reconciled.list, &obj));

  // Buffers allocated before this point keep the channel they were given.
  auto channel = std::make_shared<NvSciSyncChannel>(next_channel_id_.fetch_add(1), obj);
  channel->sessions = std::move(sessions);
  producer_channel_ = channel;
  return channel;
}

void NvSciRuntime::on_acknowledged(
  pid_t pid, const std::vector<std::pair<uint32_t, uint64_t>> & acks)
{
  for (const auto & ack : acks) {
    ledger_->acknowledge(pid, ack.first, ack.second);
  }
}

void NvSciRuntime::on_peer_ready(pid_t pid)
{
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(pid);
  if (it != peers_.end()) {
    ready_sessions_.insert(it->second->session());
  }
  RCUTILS_LOG_DEBUG_NAMED(
    "cuda_buffer_backend", "NvSci channel to process %d is ready", static_cast<int>(pid));
}

std::shared_ptr<NvSciPeer> NvSciRuntime::find_peer(pid_t pid)
{
  std::lock_guard<std::mutex> lock(peers_mutex_);
  auto it = peers_.find(pid);
  return it != peers_.end() ? it->second : nullptr;
}

void NvSciRuntime::connect_peer(pid_t pid) noexcept
{
  if (pid <= 0 || pid == getpid() || !ipc_available()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    if (wanted_peers_.count(pid) != 0) {
      return;
    }
    wanted_peers_[pid].next_attempt = std::chrono::steady_clock::now();
  }
  const uint64_t one = 1;
  const ssize_t written = write(wake_fd_, &one, sizeof(one));
  (void)written;
}

void NvSciRuntime::open_peer(pid_t pid, WantedPeer * want)
{
  // Runs on the dispatcher thread without peers_mutex_: opening an endpoint
  // can take a while.
  auto retry_later = [want](std::chrono::milliseconds delay) {
      want->next_attempt = std::chrono::steady_clock::now() + delay;
    };
  detail::NvSciEndpointPool * pairs = endpoint_pool();
  detail::NvSciEndpointPool::Claim claim;
  switch (pairs->claim(pid, &claim)) {
    case detail::NvSciEndpointPool::Result::kClaimed:
      break;
    case detail::NvSciEndpointPool::Result::kWaitForPeer:
      // The process with the lower pid claims the pair; look again on the
      // next tick.
      retry_later(std::chrono::milliseconds(kDispatchTickMs));
      return;
    case detail::NvSciEndpointPool::Result::kExhausted:
      if (!want->warned) {
        RCUTILS_LOG_WARN_NAMED(
          "cuda_buffer_backend",
          "Every pair of the NvSci endpoint pool %s (%zu) is in use, so process %d gets the "
          "CPU path; add pairs to it and to the NvSciIpc configuration. Retrying in the "
          "background", pairs->inventory_path().c_str(), pairs->pair_count(),
          static_cast<int>(pid));
        want->warned = true;
      }
      retry_later(std::chrono::seconds(5));
      return;
    case detail::NvSciEndpointPool::Result::kUnavailable:
      retry_later(std::chrono::seconds(5));
      return;
  }
  // Before this process's end was ever open, the claimer gives the pair back
  // and claims another next time; the other process keeps its half and tries
  // the same end again, since the claimer is waiting on the other end.
  auto give_back_unopened = [pairs, pid, &claim]() {
      if (claim.claimer) {
        pairs->release(pid);
      }
    };
  NvSciIpcEndpoint endpoint = 0;
  const NvSciError err = NvSciIpcOpenEndpoint(claim.endpoint_name.c_str(), &endpoint);
  if (err != NvSciError_Success) {
    give_back_unopened();
    if (!want->warned) {
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend", "Cannot open NvSciIpc endpoint '%s': NvSciError %d. "
        "Retrying in the background", claim.endpoint_name.c_str(), static_cast<int>(err));
      want->warned = true;
    }
    retry_later(std::chrono::seconds(1));
    return;
  }
  std::shared_ptr<NvSciPeer> peer;
  // Set once this channel's endpoint is closed, wherever its last reference
  // goes; until then the dispatcher does not reopen it.
  auto endpoint_closed = std::make_shared<std::atomic<bool>>(false);
  try {
    peer = std::make_shared<NvSciPeer>(
      *this, pid, next_session_.fetch_add(1), endpoint,
      [pid, endpoint_closed]() {
        // Only now that this end is closed may the pair go to anyone else.
        endpoint_pool()->release(pid);
        endpoint_closed->store(true, std::memory_order_release);
      });
  } catch (const std::exception & e) {
    // The peer closed the endpoint on its way out.
    give_back_unopened();
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend", "Cannot set up the NvSciIpc channel to process %d: %s",
      static_cast<int>(pid), e.what());
    retry_later(std::chrono::seconds(5));
    return;
  }
  want->warned = false;
  want->endpoint_closed = std::move(endpoint_closed);
  epoll_event ev{};
  // Edge-triggered: the dispatcher drains every endpoint on each wakeup and
  // also on a timer, so it does not need the fd to stay readable.
  ev.events = EPOLLIN | EPOLLET;
  ev.data.fd = peer->event_fd();
  std::lock_guard<std::mutex> lock(peers_mutex_);
  epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, peer->event_fd(), &ev);
  peers_[pid] = std::move(peer);
  RCUTILS_LOG_DEBUG_NAMED(
    "cuda_buffer_backend", "Opened NvSciIpc endpoint '%s' to process %d",
    claim.endpoint_name.c_str(), static_cast<int>(pid));
}

void NvSciRuntime::open_wanted_peers()
{
  const auto now = std::chrono::steady_clock::now();
  std::vector<std::pair<pid_t, WantedPeer>> due;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (auto it = wanted_peers_.begin(); it != wanted_peers_.end(); ) {
      if (!process_alive(it->first)) {
        // Descriptors sent to it will never be imported.
        ledger_->release_process(it->first);
        it = wanted_peers_.erase(it);
        continue;
      }
      // A dropped channel's endpoint may still be open in an import that
      // holds it; reopening it now would fail and release the pair under it.
      const auto & closed = it->second.endpoint_closed;
      const bool previous_closed = !closed || closed->load(std::memory_order_acquire);
      if (peers_.count(it->first) == 0 && previous_closed && it->second.next_attempt <= now) {
        due.emplace_back(*it);
      }
      ++it;
    }
  }
  for (auto & entry : due) {
    open_peer(entry.first, &entry.second);
    std::lock_guard<std::mutex> lock(peers_mutex_);
    auto it = wanted_peers_.find(entry.first);
    if (it != wanted_peers_.end()) {
      it->second = entry.second;
    }
  }
}

void NvSciRuntime::drop_peer(pid_t pid)
{
  std::shared_ptr<NvSciPeer> peer;
  {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    auto it = peers_.find(pid);
    if (it == peers_.end()) {
      return;
    }
    peer = std::move(it->second);
    peers_.erase(it);
    ready_sessions_.erase(peer->session());
    conflicting_buf_sessions_.erase(peer->session());
    conflicting_sync_sessions_.erase(peer->session());
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, peer->event_fd(), nullptr);
    // If the process is still there, reconnect once both ends had time to close.
    auto wanted = wanted_peers_.find(pid);
    if (wanted != wanted_peers_.end()) {
      wanted->second.next_attempt = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    }
  }
  // Descriptors already sent to it name objects of this connection, and its
  // imports refuse them from now on, whether or not it reconnects.
  ledger_->release_process(pid);
  // The endpoint closes, and its pair is released, when the last reference
  // goes. Imports waiting on this peer give up now instead of at their
  // timeout, so they let go of it promptly; the dispatcher reconnects only
  // once it has closed.
  peer->shutdown();
  peer.reset();
  RCUTILS_LOG_DEBUG_NAMED(
    "cuda_buffer_backend", "Closed the NvSciIpc channel to process %d", static_cast<int>(pid));
}

void NvSciRuntime::dispatch()
{
  // Imports run on this thread and need the device current.
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    (void)cudaGetLastError();
  }
  epoll_event events[16];
  for (;; ) {
    const int n = epoll_wait(epoll_fd_, events, 16, kDispatchTickMs);
    for (int i = 0; i < n; ++i) {
      if (events[i].data.fd == wake_fd_) {
        uint64_t value = 0;
        const ssize_t got = read(wake_fd_, &value, sizeof(value));
        (void)got;
      }
    }

    open_wanted_peers();

    std::vector<std::shared_ptr<NvSciPeer>> peers;
    {
      std::lock_guard<std::mutex> lock(peers_mutex_);
      for (auto & entry : peers_) {
        peers.push_back(entry.second);
      }
    }
    for (auto & peer : peers) {
      peer->service();
      if (peer->lost() || !process_alive(peer->pid())) {
        drop_peer(peer->pid());
      }
    }
  }
}

bool NvSciRuntime::export_buffer(
  const NvSciBuffer & buffer, const NvSciSubscriber & subscriber, NvSciExport * out)
{
  const std::shared_ptr<NvSciBlock> & block = buffer.block();
  if (!block) {
    // Only buffers allocated here are shared; a received one goes by CPU.
    return false;
  }

  NvSciExport result;
  result.pid = getpid();
  result.buffer_id = block->id;

  if (subscriber.pid != result.pid) {
    std::shared_ptr<NvSciPeer> peer = find_peer(subscriber.pid);
    if (!peer) {
      connect_peer(subscriber.pid);
      return false;
    }
    // Another process can only take a lease that lives in shared memory, and
    // only import a block reconciled with this connection that it has not
    // already failed to import.
    if (!block->lease_shared() || !peer->ready() ||
      block->sessions.count(peer->session()) == 0 || !peer->accepts_block(block->id))
    {
      return false;
    }
    buffer.finalize_write_handle();
    NvSciSyncFence fence = NvSciSyncFenceInitializer;
    if (buffer.dup_write_fence(&fence)) {
      const std::shared_ptr<NvSciSyncChannel> & channel = buffer.write_channel();
      if (!channel || channel->sessions.count(peer->session()) == 0 ||
        !peer->accepts_channel(channel->id))
      {
        NvSciSyncFenceClear(&fence);
        return false;
      }
      try {
        peer->export_channel(*channel);
        NvSciSyncFenceIpcExportDescriptor desc;
        peer->export_fence(fence, &desc);
        std::memcpy(result.fence.data(), &desc, sizeof(desc));
      } catch (...) {
        NvSciSyncFenceClear(&fence);
        throw;
      }
      NvSciSyncFenceClear(&fence);
      result.sync_channel_id = channel->id;
      result.has_fence = true;
    }
    peer->export_block(*block);
    result.connection = peer->connection_id();
  } else {
    // A subscriber in this process takes the fence from the block itself.
    buffer.finalize_write_handle();
  }

  if (block->current_uid == 0) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist(1, UINT64_MAX);
    block->current_uid = dist(rng);
    block->lease->uid.store(block->current_uid, std::memory_order_seq_cst);
  }
  result.uid = block->current_uid;
  // Last, so that a descriptor that is not sent reserves nothing.
  const detail::NvSciLedger::Ticket ticket = ledger_->record(
    block->id, subscriber.key, subscriber.pid, subscriber.history_depth,
    std::chrono::steady_clock::now());
  result.stream = ticket.stream;
  result.seq = ticket.seq;
  *out = result;
  return true;
}

std::unique_ptr<NvSciBuffer> NvSciRuntime::import_buffer(
  const NvSciExport & in, size_t byte_size)
{
  // However the import ends, the publisher may stop reserving the block for
  // this descriptor, and for every earlier one of its stream: they were taken
  // before it. On success this runs after the reference was taken.
  struct Acknowledge
  {
    NvSciRuntime & runtime;
    const NvSciExport & in;
    ~Acknowledge()
    {
      if (in.stream == 0) {
        return;
      }
      if (in.pid == getpid()) {
        runtime.ledger_->acknowledge(in.pid, in.stream, in.seq);
      } else if (std::shared_ptr<NvSciPeer> peer = runtime.find_peer(in.pid)) {
        peer->acknowledge(in.stream, in.seq);
      }
    }
  } acknowledge{*this, in};

  // Take a reference first, then check the uid: if the block was reused in
  // between, the uid no longer matches and the reference is given back.
  auto acquire = [&in](IPCMetadata * lease) {
      // Sequentially consistent: pairs with the pool clearing the uid and then
      // reading the count before it reuses a block.
      lease->refcount.fetch_add(1, std::memory_order_seq_cst);
      if (lease->uid.load(std::memory_order_seq_cst) != in.uid) {
        lease->refcount.fetch_sub(1, std::memory_order_acq_rel);
        throw CudaError("stale NvSci descriptor: the block has been reused");
      }
    };

  if (in.pid == getpid()) {
    std::shared_ptr<NvSciBlock> block;
    {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      auto it = all_blocks_.find(in.buffer_id);
      if (it != all_blocks_.end()) {
        block = it->second;
      }
    }
    if (!block) {
      throw CudaError("stale NvSci descriptor: the block no longer exists");
    }
    if (byte_size > block->mapping->size) {
      throw CudaError("NvSci descriptor is larger than the block it names");
    }
    acquire(block->lease);
    std::shared_ptr<NvSciSyncChannel> channel;
    NvSciSyncFence fence = NvSciSyncFenceInitializer;
    const bool has_fence = block->dup_last_write(&channel, &fence);
    auto release = [block]() {
        block->lease->refcount.fetch_sub(1, std::memory_order_acq_rel);
      };
    try {
      auto buffer = std::make_unique<NvSciBuffer>(
        block->mapping->ptr, byte_size, device_id_, std::move(channel), false, nullptr,
        release, has_fence ? &fence : nullptr);
      NvSciSyncFenceClear(&fence);
      return buffer;
    } catch (...) {
      NvSciSyncFenceClear(&fence);
      release();
      throw;
    }
  }

  std::shared_ptr<NvSciPeer> peer = find_peer(in.pid);
  if (!peer) {
    connect_peer(in.pid);
    throw CudaError("no NvSciIpc channel to the publisher yet");
  }
  std::shared_ptr<NvSciImportedBlock> imported;
  std::shared_ptr<NvSciSyncChannel> channel;
  switch (peer->wait_objects(
      in.connection, in.buffer_id, in.has_fence, in.sync_channel_id, kImportWait,
      &imported, &channel))
  {
    case NvSciPeer::WaitResult::kReady:
      break;
    case NvSciPeer::WaitResult::kWrongConnection:
      throw CudaError("NvSci descriptor from an earlier connection to the publisher");
    case NvSciPeer::WaitResult::kGone:
      throw CudaError("the publisher's NvSci objects cannot be imported");
    case NvSciPeer::WaitResult::kTimedOut:
      throw CudaError("the publisher's NvSci objects did not arrive");
  }
  if (byte_size > imported->mapping->size) {
    throw CudaError("NvSci descriptor is larger than the block it names");
  }

  acquire(imported->lease);
  auto release = [imported]() {
      imported->lease->refcount.fetch_sub(1, std::memory_order_acq_rel);
    };
  NvSciSyncFence fence = NvSciSyncFenceInitializer;
  try {
    if (in.has_fence) {
      NvSciSyncFenceIpcExportDescriptor desc;
      std::memcpy(&desc, in.fence.data(), sizeof(desc));
      NVSCI_CHECK(NvSciSyncIpcImportFence(channel->obj, &desc, &fence));
    }
    auto buffer = std::make_unique<NvSciBuffer>(
      imported->mapping->ptr, byte_size, device_id_, std::move(channel), false, nullptr,
      release, in.has_fence ? &fence : nullptr);
    NvSciSyncFenceClear(&fence);
    return buffer;
  } catch (...) {
    NvSciSyncFenceClear(&fence);
    release();
    throw;
  }
}

}  // namespace cuda_buffer_backend
