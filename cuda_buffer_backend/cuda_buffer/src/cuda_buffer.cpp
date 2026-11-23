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

#include "cuda_buffer/cuda_buffer.hpp"

#include <rcutils/logging_macros.h>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_memory_pool.hpp"
#include "cuda_buffer/vmm_buffer.hpp"

#ifdef CUDA_BUFFER_ENABLE_NVSCI
#include "cuda_buffer/nvsci_buffer.hpp"
#include "cuda_buffer/nvsci_runtime.hpp"
#endif

namespace cuda_buffer_backend
{

namespace
{

constexpr const char * kStorageEnv = "CUDA_BUFFER_STORAGE";

/// The default backend, or -1 until it has been resolved.
std::atomic<int> g_default_kind{-1};

StorageKind default_from_environment() noexcept
{
  const char * value = std::getenv(kStorageEnv);
  if (value == nullptr || *value == '\0') {
    return kDefaultStorageKind;
  }
  const std::optional<StorageKind> kind = parse_storage_kind(value);
  if (!kind) {
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend", "Unknown %s='%s' (expected 'vmm' or 'nvsci'); using '%s'",
      kStorageEnv, value, to_string(kDefaultStorageKind));
    return kDefaultStorageKind;
  }
  if (!is_storage_available(*kind)) {
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend", "%s='%s' is not available in this process; using '%s'",
      kStorageEnv, value, to_string(kDefaultStorageKind));
    return kDefaultStorageKind;
  }
  return *kind;
}

}  // namespace

// -- Backend selection --

const char * to_string(StorageKind kind) noexcept
{
  switch (kind) {
    case StorageKind::Vmm:
      return "vmm";
    case StorageKind::NvSci:
      return "nvsci";
  }
  return "unknown";
}

std::optional<StorageKind> parse_storage_kind(std::string_view name) noexcept
{
  if (name == "vmm") {
    return StorageKind::Vmm;
  }
  if (name == "nvsci") {
    return StorageKind::NvSci;
  }
  return std::nullopt;
}

bool is_storage_available(StorageKind kind) noexcept
{
  switch (kind) {
    case StorageKind::Vmm: {
        static const bool available = [] {
            try {
              return get_or_create_global_pool() != nullptr;
            } catch (const std::exception & e) {
              RCUTILS_LOG_WARN_NAMED(
                "cuda_buffer_backend", "VMM storage unavailable: %s", e.what());
              return false;
            }
          }();
        return available;
      }
    case StorageKind::NvSci:
#ifdef CUDA_BUFFER_ENABLE_NVSCI
      return NvSciRuntime::is_available();
#else
      return false;
#endif
  }
  return false;
}

StorageKindMask importable_storage_kinds() noexcept
{
  StorageKindMask mask = 0;
  if (vmm_ipc_supported()) {
    mask |= storage_kind_bit(StorageKind::Vmm);
  }
#ifdef CUDA_BUFFER_ENABLE_NVSCI
  if (NvSciRuntime::is_available() && NvSciRuntime::get().ipc_available()) {
    mask |= storage_kind_bit(StorageKind::NvSci);
  }
#endif
  return mask;
}

StorageKind default_storage_kind() noexcept
{
  int kind = g_default_kind.load(std::memory_order_acquire);
  if (kind >= 0) {
    return static_cast<StorageKind>(kind);
  }
  // Resolved once. A set_default_storage_kind() that races with this wins.
  static const StorageKind from_environment = default_from_environment();
  int unresolved = -1;
  g_default_kind.compare_exchange_strong(
    unresolved, static_cast<int>(from_environment), std::memory_order_acq_rel);
  return static_cast<StorageKind>(g_default_kind.load(std::memory_order_acquire));
}

void set_default_storage_kind(StorageKind kind)
{
  if (!is_storage_available(kind)) {
    throw CudaError(
            std::string("set_default_storage_kind: storage '") + to_string(kind) +
            "' is not available in this process");
  }
  g_default_kind.store(static_cast<int>(kind), std::memory_order_release);
}

// -- CudaBuffer --

CudaBuffer::~CudaBuffer()
{
  // A subclass normally takes these in its own destructor so that it can keep
  // its storage alive until they complete. Anything left is waited for here.
  for (cudaEvent_t event : read_events_) {
    cudaEventSynchronize(event);
    cudaEventDestroy(event);
  }
  if (!read_events_.empty()) {
    (void)cudaGetLastError();
  }
}

std::unique_ptr<CudaBuffer> CudaBuffer::allocate(size_t byte_size, StorageKind kind)
{
  // An empty buffer owns no storage, so its backend does not matter.
  if (byte_size == 0) {
    return VmmBuffer::allocate(0);
  }
  // No default case: -Wswitch points here when a backend is added.
  switch (kind) {
    case StorageKind::Vmm:
      return VmmBuffer::allocate(byte_size);
    case StorageKind::NvSci:
#ifdef CUDA_BUFFER_ENABLE_NVSCI
      return NvSciBuffer::allocate(byte_size);
#else
      throw CudaError("CudaBuffer::allocate: cuda_buffer was built without NvSci support");
#endif
  }
  throw CudaError("CudaBuffer::allocate: unknown StorageKind");
}

ReadHandle CudaBuffer::get_read_handle(cudaStream_t stream) const
{
  (void)reap_completed_reads();

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      finalize_write_handle_locked();
    }
  }

  wait_for_write(stream);

  return ReadHandle(device_ptr(), this, stream);
}

WriteHandle CudaBuffer::get_write_handle(cudaStream_t stream)
{
  if (!is_writable()) {
    throw CudaError("CudaBuffer: buffer is shared read-only by another process");
  }

  std::lock_guard<std::mutex> lg(write_acquire_mutex_);
  const size_t pending_reads = reap_completed_reads();

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      throw CudaError("CudaBuffer: write handle already in use; cannot acquire second handle");
    }
    if (handle_state_->state == HandleState::State::Finalized) {
      throw CudaError("CudaBuffer: write already finalized; cannot acquire write handle");
    }
  }

  if (pending_reads > 0) {
    throw CudaError("CudaBuffer: read events exist; cannot re-acquire write handle");
  }

  if (!handle_state_) {handle_state_ = std::make_shared<HandleState>();}
  handle_state_->state = HandleState::State::InUse;
  handle_state_->write_stream = stream;
  return WriteHandle(device_ptr(), this, stream, handle_state_);
}

void CudaBuffer::finalize_write_handle() const
{
  if (handle_state_ == nullptr) {return;}
  std::lock_guard<std::mutex> lock(handle_state_->mtx);
  finalize_write_handle_locked();
}

void CudaBuffer::finalize_write_handle_locked() const
{
  if (handle_state_->state != HandleState::State::InUse) {return;}
  if (cuda_is_stream_usable(handle_state_->write_stream)) {
    signal_write(handle_state_->write_stream);
  }
  handle_state_->state = HandleState::State::Finalized;
  handle_state_->write_stream = nullptr;
}

void CudaBuffer::signal_read(cudaStream_t stream) const noexcept
{
  // Read events are only queried and synchronized in this process, never
  // exported with cudaIpcGetEventHandle. cudaEventInterprocess is therefore
  // not needed, and it makes the CUDA driver create a per-process
  // /dev/shm/cuda.shm.* segment that it does not remove at process exit.
  cudaEvent_t event{nullptr};
  CUDA_CHECK_NOTHROW(
    cudaEventCreateWithFlags(&event, CUDA_BUFFER_MINIMUM_EVENT_FLAGS),
    return);
  CUDA_CHECK_NOTHROW(cudaEventRecord(event, stream), {cudaEventDestroy(event); return;});
  std::lock_guard<std::mutex> lock(read_events_mutex_);
  read_events_.push_back(event);
}

size_t CudaBuffer::reap_completed_reads() const
{
  std::lock_guard<std::mutex> lock(read_events_mutex_);
  auto event = read_events_.begin();
  while (event != read_events_.end()) {
    const cudaError_t status = cudaEventQuery(*event);
    if (status == cudaSuccess) {
      CUDA_CHECK_NOTHROW(cudaEventDestroy(*event), (void)0);
      event = read_events_.erase(event);
      continue;
    }
    if (status != cudaErrorNotReady) {
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend", "cudaEventQuery failed: %s", cudaGetErrorName(status));
    }
    (void)cudaGetLastError();
    ++event;
  }
  return read_events_.size();
}

std::vector<cudaEvent_t> CudaBuffer::take_read_events() const
{
  std::lock_guard<std::mutex> lock(read_events_mutex_);
  std::vector<cudaEvent_t> events;
  events.swap(read_events_);
  return events;
}

// -- ReadHandle --

void ReadHandle::release() noexcept
{
  if (!owner_ || !cuda_is_stream_usable(stream_)) {return;}
  owner_->signal_read(stream_);
}

// -- WriteHandle --

void WriteHandle::release() noexcept
{
  if (!state_) {return;}
  std::lock_guard<std::mutex> lock(state_->mtx);
  // A finalized write no longer refers to its buffer, which may be gone by now.
  if (state_->state == HandleState::State::Finalized) {return;}
  cudaStream_t ws = state_->write_stream ? state_->write_stream : stream_;
  if (owner_ && cuda_is_stream_usable(ws)) {
    owner_->signal_write(ws);
  }
  state_->state = HandleState::State::Finalized;
  state_->write_stream = nullptr;
}

}  // namespace cuda_buffer_backend
