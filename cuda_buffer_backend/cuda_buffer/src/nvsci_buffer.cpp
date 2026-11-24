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

#include "cuda_buffer/nvsci_buffer.hpp"

#include <rcutils/logging_macros.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/nvsci_runtime.hpp"
#include "deferred_release.hpp"

namespace cuda_buffer_backend
{

NvSciBuffer::NvSciBuffer() = default;

NvSciBuffer::NvSciBuffer(
  uint8_t * ptr, size_t size, int device_id,
  std::shared_ptr<NvSciSyncChannel> channel, bool writable,
  std::shared_ptr<NvSciBlock> block, std::function<void()> on_release,
  const NvSciSyncFence * fence)
: CudaBuffer(size, device_id),
  ptr_(ptr),
  channel_(std::move(channel)),
  writable_(writable),
  block_(std::move(block)),
  on_release_(std::move(on_release))
{
  if (fence != nullptr) {
    NVSCI_CHECK(NvSciSyncFenceDup(fence, &write_fence_));
    has_write_fence_ = true;
  }
}

NvSciBuffer::~NvSciBuffer()
{
  // The storage outlives the write and every read recorded against it. The
  // fence refers to the sync object, so the channel is kept until it is cleared.
  std::vector<cudaEvent_t> events = take_read_events();
  if (write_done_) {
    events.push_back(write_done_);
    write_done_ = nullptr;
  }
  NvSciSyncFence fence = write_fence_;
  write_fence_ = NvSciSyncFenceInitializer;
  detail::release_after(
    std::move(events),
    [fence, channel = channel_, release = std::move(on_release_)]() mutable {
      NvSciSyncFenceClear(&fence);
      channel.reset();
      if (release) {
        release();
      }
    });
}

std::unique_ptr<NvSciBuffer> NvSciBuffer::allocate(size_t byte_size)
{
  if (byte_size == 0) {
    return std::make_unique<NvSciBuffer>();
  }
  return NvSciRuntime::get().allocate(byte_size);
}

bool NvSciBuffer::dup_write_fence(NvSciSyncFence * fence) const
{
  std::lock_guard<std::mutex> lock(sync_mutex_);
  if (!has_write_fence_) {
    return false;
  }
  NVSCI_CHECK(NvSciSyncFenceDup(&write_fence_, fence));
  return true;
}

void NvSciBuffer::wait_for_write(cudaStream_t stream) const
{
  std::lock_guard<std::mutex> lock(sync_mutex_);
  if (!has_write_fence_ || !channel_) {
    return;
  }
  // A null stream is the legacy default stream, which is a valid target.
  cudaExternalSemaphoreWaitParams params{};
  params.params.nvSciSync.fence = &write_fence_;
  CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&channel_->semaphore, &params, 1, stream));
}

void NvSciBuffer::signal_write(cudaStream_t stream) const noexcept
{
  if (!writable_ || !channel_) {
    return;
  }

  std::lock_guard<std::mutex> lock(sync_mutex_);
  NvSciSyncFenceClear(&write_fence_);
  has_write_fence_ = false;
  cudaExternalSemaphoreSignalParams params{};
  params.params.nvSciSync.fence = &write_fence_;
  CUDA_CHECK_NOTHROW(
    cudaSignalExternalSemaphoresAsync(&channel_->semaphore, &params, 1, stream),
    return);
  has_write_fence_ = true;

  if (block_) {
    try {
      block_->record_write(channel_, write_fence_);
    } catch (const std::exception & e) {
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend", "Could not record an NvSci write fence: %s", e.what());
    }
  }

  if (!write_done_) {
    CUDA_CHECK_NOTHROW(
      cudaEventCreateWithFlags(&write_done_, CUDA_BUFFER_MINIMUM_EVENT_FLAGS),
      {write_done_ = nullptr; return;});
  }
  CUDA_CHECK_NOTHROW(cudaEventRecord(write_done_, stream), (void)0);
}

}  // namespace cuda_buffer_backend
