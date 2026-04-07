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

#ifndef CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"
#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/buffer_impl_base.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

namespace cuda_buffer_backend
{

template<typename T>
class CudaBufferImpl : public rosidl::BufferImplBase<T>
{
public:
  CudaBufferImpl()
  : size_(0) {}

  explicit CudaBufferImpl(size_t size)
  : size_(size)
  {
    if (size_ > 0) {
      allocate_buffer(size_);
    }
  }

  explicit CudaBufferImpl(CudaBuffer && buffer, size_t size)
  : size_(size), cuda_buffer_(std::move(buffer)) {}

  ~CudaBufferImpl() = default;

  CudaBufferImpl(const CudaBufferImpl &) = delete;
  CudaBufferImpl & operator=(const CudaBufferImpl &) = delete;
  CudaBufferImpl(CudaBufferImpl &&) = delete;
  CudaBufferImpl & operator=(CudaBufferImpl &&) = delete;

  std::string get_backend_type() const override {return "cuda";}

  size_t size() const override {return size_;}

  void resize(size_t n)
  {
    if (n == size_) {
      return;
    }

    if (n == 0) {
      clear();
      return;
    }

    CudaBuffer new_buffer;
    allocate_buffer_internal(new_buffer, n);

    if (size_ > 0 && cuda_buffer_.size() > 0) {
      size_t copy_size = std::min(n, size_) * sizeof(T);
      CUDA_CHECK(cudaMemcpyAsync(
        new_buffer.get_device_ptr(), cuda_buffer_.get_device_ptr(),
        copy_size, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cuda_buffer_ = std::move(new_buffer);
    size_ = n;
  }

  void clear()
  {
    cuda_buffer_ = CudaBuffer();
    size_ = 0;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    auto cpu = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu->get_storage().resize(size_);

    if (size_ > 0 && cuda_buffer_.size() > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
        cpu->get_storage().data(), cuda_buffer_.get_device_ptr(),
        size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    return cpu;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    auto copy = std::make_unique<CudaBufferImpl<T>>(size_);

    if (size_ > 0 && cuda_buffer_.size() > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
        copy->cuda_buffer_.get_device_ptr(), cuda_buffer_.get_device_ptr(),
        size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    return copy;
  }

  CudaBuffer & get_cuda_buffer() {return cuda_buffer_;}
  const CudaBuffer & get_cuda_buffer() const {return cuda_buffer_;}

  void set_stream(cudaStream_t stream) {stream_ = stream;}
  cudaStream_t get_stream() const {return stream_;}

  static std::shared_ptr<CudaMemoryPool> get_or_create_global_pool()
  {
    static std::shared_ptr<CudaMemoryPool> global_pool;
    static std::mutex pool_mutex;

    std::lock_guard<std::mutex> lock(pool_mutex);

    if (!global_pool) {
      global_pool = std::make_shared<CudaMemoryPool>();
      CUresult r = global_pool->create();
      if (r != CUDA_SUCCESS) {
        global_pool.reset();
        throw CudaError(__FILE__, __LINE__, "CudaMemoryPool::create", r);
      }
    }

    return global_pool;
  }

  static bool is_pool_ipc_capable()
  {
    auto pool = get_or_create_global_pool();
    return pool && pool->is_ipc_capable();
  }

private:
  void allocate_buffer(size_t n)
  {
    allocate_buffer_internal(cuda_buffer_, n);
  }

  void allocate_buffer_internal(CudaBuffer & buffer, size_t n)
  {
    auto pool = get_or_create_global_pool();
    size_t byte_size = n * sizeof(T);

    VmmBlock * block = pool->allocate(byte_size);

    cudaEvent_t ev = nullptr;
    cudaError_t ev_err = cudaEventCreateWithFlags(&ev, CUDA_BUFFER_EVENT_FLAGS);
    if (ev_err != cudaSuccess) {
      (void)cudaGetLastError();
      ev_err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
      if (ev_err != cudaSuccess) {
        ev = nullptr;
        (void)cudaGetLastError();
        RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
          "Failed to create CUDA event; stream ordering disabled for this buffer");
      }
    }

    buffer = CudaBuffer(
      reinterpret_cast<void *>(block->va), byte_size, pool->deleter(block));

    if (ev) {
      buffer.set_write_event(ev, true);
    }
  }

  size_t size_;
  CudaBuffer cuda_buffer_;
  cudaStream_t stream_{nullptr};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_
