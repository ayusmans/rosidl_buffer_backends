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

#include <memory>

#include <cuda_runtime.h>

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
  if (buffer.get_backend_type() != "cuda") {
    throw CudaError(
            "from_buffer called on non-CUDA buffer (backend: " +
            buffer.get_backend_type() + ")");
  }
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_buffer called on buffer with null implementation");
  }
  auto * cuda_impl = const_cast<CudaBufferImpl<T> *>(
    dynamic_cast<const CudaBufferImpl<T> *>(impl));
  if (!cuda_impl) {
    throw CudaError("from_buffer: failed to cast buffer impl to CudaBufferImpl");
  }
  return cuda_impl;
}

template<typename T>
const CudaBufferImpl<T> * get_cuda_impl(const rosidl::Buffer<T> & buffer)
{
  if (buffer.get_backend_type() != "cuda") {
    throw CudaError(
            "from_buffer called on non-CUDA buffer (backend: " +
            buffer.get_backend_type() + ")");
  }
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_buffer called on buffer with null implementation");
  }
  const auto * cuda_impl = dynamic_cast<const CudaBufferImpl<T> *>(impl);
  if (!cuda_impl) {
    throw CudaError("from_buffer: failed to cast buffer impl to CudaBufferImpl");
  }
  return cuda_impl;
}

}  // namespace detail

/// \brief Allocate a ROS message with a CUDA-backed buffer of \p count elements.
/// \tparam MsgT ROS message type whose `data` field is a `rosidl::Buffer<uint8_t>`.
/// \param count Number of uint8_t elements to allocate on the GPU.
/// \return A message with `data` backed by a CUDA memory pool allocation.
template<typename MsgT>
MsgT allocate_msg(size_t count)
{
  auto impl = std::make_unique<CudaBufferImpl<uint8_t>>(count);
  MsgT msg;
  msg.data = rosidl::Buffer<uint8_t>(std::move(impl));
  return msg;
}

/// \brief Acquire a write handle for a CUDA-backed buffer.
/// \tparam T Element type of the buffer.
/// \param buffer Mutable buffer to write to.
/// \param stream CUDA stream for GPU operations.
/// \return WriteHandle whose destructor records a write event on \p stream.
/// \throws CudaError if \p buffer is not CUDA-backed or a handle is already active.
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
/// \tparam T Element type of the buffer.
/// \param buffer Const buffer to read from.
/// \param stream CUDA stream; will wait on the publisher's write event.
/// \return ReadHandle whose destructor records a read event on \p stream.
/// \throws CudaError if \p buffer is not CUDA-backed.
template<typename T>
ReadHandle from_buffer(
  const rosidl::Buffer<T> & buffer,
  cudaStream_t stream)
{
  const auto * cuda_impl = detail::get_cuda_impl(buffer);
  return cuda_impl->get_cuda_buffer().get_read_handle(stream);
}

/// \brief Copy data into a CUDA buffer through a write handle.
/// \param src Source pointer (device or host, depending on \p kind).
/// \param byte_count Number of bytes to copy.
/// \param wh Active write handle for the destination buffer.
/// \param stream CUDA stream for the async memcpy.
/// \param kind Copy direction (default: device-to-device).
inline void to_buffer(
  const void * src,
  size_t byte_count,
  WriteHandle & wh,
  cudaStream_t stream,
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice)
{
  if (!wh.get_ptr()) {
    throw CudaError("to_buffer: WriteHandle has null pointer");
  }
  CUDA_CHECK(cudaMemcpyAsync(wh.get_ptr(), src, byte_count, kind, stream));
}

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_API_HPP_
