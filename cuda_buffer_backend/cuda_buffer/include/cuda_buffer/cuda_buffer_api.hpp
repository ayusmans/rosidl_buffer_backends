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

#ifndef CUDA_BUFFER__CUDA_BUFFER_API_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_API_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace cuda_buffer_backend
{

namespace detail
{

template<typename T>
CudaBufferImpl<T> * get_cuda_impl(rosidl::Buffer<T> & buffer)
{
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_buffer called on buffer with null implementation");
  }
  auto * cuda_impl = const_cast<CudaBufferImpl<T> *>(
    dynamic_cast<const CudaBufferImpl<T> *>(impl));
  if (!cuda_impl) {
    throw CudaError(
            "from_buffer: buffer is not CUDA-backed (backend: " +
            buffer.get_backend_type() + ")");
  }
  return cuda_impl;
}

template<typename T>
const CudaBufferImpl<T> * get_cuda_impl(const rosidl::Buffer<T> & buffer)
{
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_buffer called on buffer with null implementation");
  }
  const auto * cuda_impl = dynamic_cast<const CudaBufferImpl<T> *>(impl);
  if (!cuda_impl) {
    throw CudaError(
            "from_buffer: buffer is not CUDA-backed (backend: " +
            buffer.get_backend_type() + ")");
  }
  return cuda_impl;
}

}  // namespace detail

/// \brief Allocate a ROS message with a CUDA-backed buffer of \p count elements.
template<typename MsgT>
MsgT allocate_msg(size_t count)
{
  auto impl = std::make_unique<CudaBufferImpl<uint8_t>>(count);
  MsgT msg;
  msg.data = rosidl::Buffer<uint8_t>(std::move(impl));
  return msg;
}

/// \brief Acquire a write handle for a CUDA-backed buffer.
/// \throws CudaError if the buffer is not CUDA-backed.
template<typename T>
WriteHandle from_buffer(
  rosidl::Buffer<T> & buffer,
  cudaStream_t stream)
{
  auto * cuda_impl = detail::get_cuda_impl(buffer);
  cuda_impl->set_stream(stream);
  return cuda_impl->get_cuda_buffer().get_write_handle(stream);
}

/// \brief Acquire a read handle for a CUDA-backed buffer.
/// \throws CudaError if the buffer is not CUDA-backed.
template<typename T>
ReadHandle from_buffer(
  const rosidl::Buffer<T> & buffer,
  cudaStream_t stream)
{
  const auto * cuda_impl = detail::get_cuda_impl(buffer);
  return cuda_impl->get_cuda_buffer().get_read_handle(stream);
}

/// \brief Create a new CUDA-backed buffer from a raw pointer.
/// Allocates GPU memory and copies data via the given stream.
inline rosidl::Buffer<uint8_t> to_buffer(
  const void * src,
  size_t byte_count,
  cudaStream_t stream,
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice)
{
  auto impl = std::make_unique<CudaBufferImpl<uint8_t>>(byte_count);
  if (byte_count > 0 && src) {
    auto wh = impl->get_cuda_buffer().get_write_handle(stream);
    CUDA_CHECK(cudaMemcpyAsync(wh.get_ptr(), src, byte_count, kind, stream));
  }
  return rosidl::Buffer<uint8_t>(std::move(impl));
}

/// \brief Create a new CUDA-backed buffer from any rosidl::Buffer.
/// If the source is already CUDA-backed, performs a device-to-device copy.
/// Otherwise copies from host.
template<typename T>
rosidl::Buffer<T> to_buffer(
  const rosidl::Buffer<T> & src,
  cudaStream_t stream)
{
  size_t byte_count = src.size() * sizeof(T);
  auto impl = std::make_unique<CudaBufferImpl<T>>(byte_count);
  if (byte_count > 0) {
    auto wh = impl->get_cuda_buffer().get_write_handle(stream);
    if (src.get_backend_type() == "cuda") {
      const auto * src_impl = detail::get_cuda_impl(src);
      auto rh = src_impl->get_cuda_buffer().get_read_handle(stream);
      CUDA_CHECK(cudaMemcpyAsync(
        wh.get_ptr(), rh.get_ptr(), byte_count, cudaMemcpyDeviceToDevice, stream));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
        wh.get_ptr(), src.data(), byte_count, cudaMemcpyHostToDevice, stream));
    }
  }
  return rosidl::Buffer<T>(std::move(impl));
}

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_API_HPP_
