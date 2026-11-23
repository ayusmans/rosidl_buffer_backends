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

#include "cuda_buffer/vmm_buffer.hpp"

#include <rcutils/logging_macros.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"
#include "deferred_release.hpp"

namespace cuda_buffer_backend
{

// -- VmmBuffer --

VmmBuffer::VmmBuffer() = default;

VmmBuffer::VmmBuffer(
  void * ptr, size_t size, int device_id,
  std::function<void(uint8_t *)> custom_deleter)
: CudaBuffer(size, device_id),
  device_ptr_(static_cast<uint8_t *>(ptr), std::move(custom_deleter))
{
}

VmmBuffer::~VmmBuffer()
{
  // The storage outlives the write and every read recorded against it.
  std::vector<cudaEvent_t> events = take_read_events();
  if (owns_write_event_ && write_event_) {
    events.push_back(write_event_);
    write_event_ = nullptr;
  }
  if (events.empty()) {
    return;
  }
  auto storage = std::make_shared<decltype(device_ptr_)>(std::move(device_ptr_));
  detail::release_after(std::move(events), [storage]() {storage->reset();});
}

std::unique_ptr<VmmBuffer> VmmBuffer::allocate(size_t byte_size)
{
  if (byte_size == 0) {
    return std::make_unique<VmmBuffer>();
  }

  auto pool = get_or_create_global_pool();

  VmmBlock * block = pool->allocate(byte_size);

  cudaEvent_t ev = nullptr;
  cudaError_t ev_err = cudaEventCreateWithFlags(&ev, CUDA_BUFFER_DEFAULT_EVENT_FLAGS);
  if (ev_err != cudaSuccess) {
    const cudaError_t interprocess_event_error = ev_err;
    (void)cudaGetLastError();
    ev_err = cudaEventCreateWithFlags(&ev, CUDA_BUFFER_MINIMUM_EVENT_FLAGS);
    if (ev_err != cudaSuccess) {
      ev = nullptr;
      (void)cudaGetLastError();
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
        "Failed to create CUDA event; stream ordering disabled for this buffer");
    } else {
      RCUTILS_LOG_WARN_ONCE_NAMED(
        "cuda_buffer_backend",
        "Failed to create interprocess CUDA event (%s); falling back to a local event",
        cudaGetErrorName(interprocess_event_error));
    }
  }

  auto buffer = std::make_unique<VmmBuffer>(
    reinterpret_cast<void *>(block->va), byte_size, pool->get_device_id(),
    pool->deleter(block));

  if (ev) {
    buffer->set_write_event(ev, true);
  }

  return buffer;
}

void VmmBuffer::wait_for_write(cudaStream_t stream) const
{
  if (write_event_ != nullptr && stream != nullptr) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, write_event_, 0));
  }
}

void VmmBuffer::signal_write(cudaStream_t stream) const noexcept
{
  if (write_event_) {
    CUDA_CHECK_NOTHROW(cudaEventRecord(write_event_, stream), (void)0);
  }
}

void VmmBuffer::default_cuda_free(uint8_t * p)
{
  if (p) {
    cudaError_t e = cudaFree(p);
    if (!cuda_error_is_safe(e)) {
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
        "cudaFree failed: %s", cudaGetErrorName(e));
    }
    (void)cudaGetLastError();
  }
}

}  // namespace cuda_buffer_backend
