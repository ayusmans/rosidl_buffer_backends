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

#ifndef CUDA_BUFFER__NVSCI_BUFFER_HPP_
#define CUDA_BUFFER__NVSCI_BUFFER_HPP_

#include <cuda_runtime.h>
#include <nvscisync.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/visibility_control.h"

namespace cuda_buffer_backend
{

struct NvSciBlock;
struct NvSciSyncChannel;

/// \brief CudaBuffer backed by an NvSciBufObj and ordered by NvSciSync fences.
///
/// A buffer allocated in this process wraps a block of the NvSciRuntime pool
/// and is writable. A buffer received from a publisher, in this process or
/// another one, wraps that publisher's block and is read-only: is_writable()
/// is false and get_write_handle() throws.
///
/// Readers wait on the producer's fence on their own stream, so neither
/// get_read_handle() nor publishing blocks the CPU.
///
/// This header is for storage and transport code. Code that only reads or
/// writes buffers should use the CudaBuffer interface and cuda_buffer_api.hpp.
class CUDA_BUFFER_PUBLIC NvSciBuffer final : public CudaBuffer
{
public:
  /// \brief Create an empty buffer that owns no device memory.
  NvSciBuffer();

  /// \param ptr device pointer of the mapped storage.
  /// \param channel sync object the producer's writes are signaled on; may be
  /// null for a buffer that is never written.
  /// \param writable false for a buffer received from a publisher.
  /// \param block the pooled block, for a buffer allocated in this process.
  /// \param on_release runs once every read and write recorded against the
  /// buffer has completed; it gives the storage back.
  /// \param fence the producer's write to wait for, or null. Duplicated.
  NvSciBuffer(
    uint8_t * ptr, size_t size, int device_id,
    std::shared_ptr<NvSciSyncChannel> channel, bool writable,
    std::shared_ptr<NvSciBlock> block, std::function<void()> on_release,
    const NvSciSyncFence * fence = nullptr);

  ~NvSciBuffer() override;

  /// \brief Allocate \p byte_size bytes from the NvSciRuntime block pool.
  /// \details A \p byte_size of zero returns an empty buffer without
  /// initializing NvSci.
  static std::unique_ptr<NvSciBuffer> allocate(size_t byte_size);

  StorageKind storage_kind() const noexcept override {return StorageKind::NvSci;}
  bool is_writable() const noexcept override {return writable_;}

  // -- Transport interface, used by NvSciRuntime --

  /// The pooled block, or null for a buffer received from a publisher.
  const std::shared_ptr<NvSciBlock> & block() const {return block_;}
  const std::shared_ptr<NvSciSyncChannel> & write_channel() const {return channel_;}

  /// \brief Duplicate the fence of the last write into \p fence.
  /// \return false if the buffer was never written.
  bool dup_write_fence(NvSciSyncFence * fence) const;

protected:
  uint8_t * device_ptr() const noexcept override {return ptr_;}
  void wait_for_write(cudaStream_t stream) const override;
  void signal_write(cudaStream_t stream) const noexcept override;

private:
  uint8_t * ptr_{nullptr};
  std::shared_ptr<NvSciSyncChannel> channel_;
  bool writable_{false};
  std::shared_ptr<NvSciBlock> block_;
  std::function<void()> on_release_;

  mutable std::mutex sync_mutex_;
  mutable NvSciSyncFence write_fence_ = NvSciSyncFenceInitializer;
  mutable bool has_write_fence_{false};
  /// Recorded after the producer's write, so release waits for it.
  mutable cudaEvent_t write_done_{nullptr};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__NVSCI_BUFFER_HPP_
