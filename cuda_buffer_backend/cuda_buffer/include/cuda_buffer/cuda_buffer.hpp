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

#ifndef CUDA_BUFFER__CUDA_BUFFER_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_handle.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/visibility_control.h"

namespace cuda_buffer_backend
{

/// \brief The storage backend behind a CudaBuffer.
/// \details Code that treats one backend specially, such as the plugin's
/// transport, switches on CudaBuffer::storage_kind() and then static_casts to
/// the class the enumerator names. The values travel in CudaBufferDescriptor
/// and in the host endpoint registry, so they must never be renumbered. Each
/// new backend adds an enumerator here.
enum class StorageKind : uint8_t
{
  /// VmmBuffer: CUDA Virtual Memory Management.
  Vmm = 0,
  /// NvSciBuffer: NvSciBuf memory ordered by NvSciSync fences.
  NvSci = 1,
};

/// \brief A set of storage backends, one bit per StorageKind value.
using StorageKindMask = uint32_t;

constexpr StorageKindMask storage_kind_bit(StorageKind kind) noexcept
{
  return StorageKindMask{1} << static_cast<uint8_t>(kind);
}

/// \brief The backend used when nothing else is specified.
inline constexpr StorageKind kDefaultStorageKind = StorageKind::Vmm;

/// \brief "vmm" or "nvsci": the name used by CUDA_BUFFER_STORAGE and the Python API.
CUDA_BUFFER_PUBLIC const char * to_string(StorageKind kind) noexcept;

/// \brief Parse a backend name as returned by to_string(). Case-sensitive.
CUDA_BUFFER_PUBLIC std::optional<StorageKind> parse_storage_kind(std::string_view name) noexcept;

/// \brief Whether buffers of \p kind can be allocated in this process.
/// \details True when the backend was compiled in and works on this machine.
/// The first call for a backend initializes its runtime (for NvSci, its
/// modules); later calls return the cached answer. Never throws.
CUDA_BUFFER_PUBLIC bool is_storage_available(StorageKind kind) noexcept;

/// \brief The backends whose buffers another process on this host can send here.
/// \details This is what the plugin advertises to its peers. It does not
/// depend on what this process allocates: a process that never allocates
/// still imports whatever its publishers send. Never throws.
CUDA_BUFFER_PUBLIC StorageKindMask importable_storage_kinds() noexcept;

/// \brief The backend used by allocations that do not name one.
/// \details Resolved on the first call, which is normally the first
/// allocation: set_default_storage_kind() if it was called, otherwise the
/// CUDA_BUFFER_STORAGE environment variable, otherwise kDefaultStorageKind.
/// An unknown or unavailable name in CUDA_BUFFER_STORAGE logs one warning and
/// falls back to kDefaultStorageKind.
CUDA_BUFFER_PUBLIC StorageKind default_storage_kind() noexcept;

/// \brief Change the default for allocations made from now on.
/// \details Buffers that already exist keep their backend.
/// \throws CudaError if \p kind is not available in this process.
CUDA_BUFFER_PUBLIC void set_default_storage_kind(StorageKind kind);

/// \brief Abstract CUDA storage object owned by CudaBufferImpl.
///
/// The public rosidl buffer hierarchy is:
///   rosidl::Buffer<T> -> rosidl::BufferImplBase<T> -> CudaBufferImpl<T>.
/// CudaBuffer is not part of that hierarchy. It is the byte-level storage
/// that CudaBufferImpl owns, and the interface through which read and write
/// handles are acquired.
///
/// This base class owns everything that is the same for every storage
/// backend: the size, the device id, the write-handle state machine (one
/// writer, no write after a read, finalize on publish), and the CUDA events
/// that track reads still in flight. How the device memory is obtained and
/// released, and how a reader's stream is ordered after the producer's write,
/// is left to the storage backend through the protected hooks below.
class CUDA_BUFFER_PUBLIC CudaBuffer
{
public:
  virtual ~CudaBuffer();

  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer & operator=(const CudaBuffer &) = delete;
  CudaBuffer(CudaBuffer &&) = delete;
  CudaBuffer & operator=(CudaBuffer &&) = delete;

  /// \brief Allocate \p byte_size bytes from the storage backend \p kind.
  /// \details Buffers received from another process are constructed by the
  /// plugin. A \p byte_size of zero returns an empty buffer without touching
  /// the device, whatever \p kind is.
  /// \throws CudaError if \p kind is not available or the allocation fails.
  static std::unique_ptr<CudaBuffer> allocate(
    size_t byte_size, StorageKind kind = default_storage_kind());

  /// \brief The storage backend this buffer belongs to.
  /// \details Every subclass returns its own enumerator, so a static_cast to
  /// the class that enumerator names is valid.
  virtual StorageKind storage_kind() const noexcept = 0;

  /// \brief Whether this process may write the buffer.
  /// \details False for storage that another process owns and shares
  /// read-only. get_write_handle() throws for such a buffer.
  virtual bool is_writable() const noexcept {return true;}

  ReadHandle get_read_handle(cudaStream_t stream) const;
  WriteHandle get_write_handle(cudaStream_t stream);

  void finalize_write_handle() const;

  size_t size() const {return size_;}
  int get_device_id() const {return device_id_;}
  uint8_t * get_device_ptr() {return device_ptr();}
  const uint8_t * get_device_ptr() const {return device_ptr();}

protected:
  CudaBuffer() = default;
  CudaBuffer(size_t size, int device_id)
  : size_(size), device_id_(device_id) {}

  /// \brief Device pointer of the storage, or nullptr for an empty buffer.
  virtual uint8_t * device_ptr() const noexcept = 0;

  /// \brief Make work submitted to \p stream wait for the producer's write.
  /// \details Called before a read handle is handed out. A null \p stream is
  /// the legacy default stream. May throw CudaError.
  virtual void wait_for_write(cudaStream_t stream) const = 0;

  /// \brief Mark the end of the producer's work on \p stream.
  /// \details Called once per write, either when the write handle is released
  /// or when the write is finalized on publish. \p stream is usable.
  virtual void signal_write(cudaStream_t stream) const noexcept = 0;

  /// \brief Hand over the events of reads that may still be in flight.
  /// \details A subclass calls this from its destructor and keeps its storage
  /// alive until every returned event has completed, usually by passing both
  /// to detail::release_after(). The caller owns the returned events.
  std::vector<cudaEvent_t> take_read_events() const;

private:
  friend class ReadHandle;
  friend class WriteHandle;

  void finalize_write_handle_locked() const;

  /// Record the end of a consumer's work on \p stream.
  void signal_read(cudaStream_t stream) const noexcept;

  /// Drop the events of reads that have completed.
  /// \return the number of reads that are still in flight.
  size_t reap_completed_reads() const;

  size_t size_{0};
  int device_id_{0};

  /// Serializes write-handle acquisition.
  std::mutex write_acquire_mutex_;
  std::shared_ptr<HandleState> handle_state_{nullptr};

  mutable std::vector<cudaEvent_t> read_events_;
  mutable std::mutex read_events_mutex_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_HPP_
