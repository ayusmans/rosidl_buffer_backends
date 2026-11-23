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
#include <optional>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace cuda_buffer_backend
{

/// \brief Allocate a fresh CUDA-backed \c rosidl::Buffer<uint8_t> of \p count
/// bytes. Pure allocation: no handle is acquired, no data is copied. The
/// caller owns the returned buffer and assigns it wherever the schema needs
/// it (e.g. `msg.data = allocate_buffer(n)` for messages that follow the
/// `data` convention, or an arbitrary field on other messages).
/// \param kind the storage backend; default_storage_kind() unless given.
/// \throws CudaError if \p kind is not available in this process.
inline rosidl::Buffer<uint8_t> allocate_buffer(
  size_t count, StorageKind kind = default_storage_kind())
{
  return rosidl::Buffer<uint8_t>(
    std::make_unique<CudaBufferImpl<uint8_t>>(count, kind));
}

/// \brief The storage backend of \p buffer, or std::nullopt if it is not
/// CUDA-backed (for example, it arrived over the CPU fallback path).
template<typename T>
std::optional<StorageKind> storage_kind_of(const rosidl::Buffer<T> & buffer)
{
  const auto * impl = dynamic_cast<const CudaBufferImpl<T> *>(buffer.get_impl());
  if (!impl) {
    return std::nullopt;
  }
  return impl->get_cuda_buffer().size() > 0 ?
         impl->get_cuda_buffer().storage_kind() : impl->storage_kind();
}

namespace detail
{

/// \brief Heap-allocate a fresh CUDA-backed rosidl::Buffer<uint8_t>, held via
/// shared_ptr so it can ride along with a promoted read/write handle.
inline std::shared_ptr<rosidl::Buffer<uint8_t>> allocate_cuda_buffer_shared(
  size_t byte_count, StorageKind kind)
{
  auto cuda_impl = std::make_unique<CudaBufferImpl<uint8_t>>(byte_count, kind);
  return std::make_shared<rosidl::Buffer<uint8_t>>(std::move(cuda_impl));
}

inline CudaBufferImpl<uint8_t> * cuda_impl_of(rosidl::Buffer<uint8_t> & buffer)
{
  return dynamic_cast<CudaBufferImpl<uint8_t> *>(buffer.get_impl());
}

}  // namespace detail

/// \brief Acquire a write handle for a CUDA-backed buffer.
/// \details If \p buffer is already CUDA-backed, returns a write handle
/// directly. If the buffer is non-CUDA (e.g. CPU-backed), a fresh
/// CUDA-backed \c rosidl::Buffer<uint8_t> is allocated (no H2D copy; the
/// caller is about to overwrite it) and a write handle for the new buffer
/// is returned. The promoted buffer is attached to the handle via
/// \c WriteHandle::get_promoted_buffer() so the caller can substitute the
/// buffer back into the message they're publishing.
/// \param kind backend of the promoted buffer; default_storage_kind() unless
/// given. Ignored when \p buffer is already CUDA-backed.
template<typename T>
WriteHandle from_output_buffer(
  rosidl::Buffer<T> & buffer,
  cudaStream_t stream,
  std::optional<StorageKind> kind = std::nullopt)
{
  auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_output_buffer called on buffer with null implementation");
  }
  if (buffer.size() == 0) {
    throw CudaError("from_output_buffer called on empty buffer");
  }
  auto * cuda_impl = dynamic_cast<CudaBufferImpl<T> *>(impl);
  if (cuda_impl) {
    cuda_impl->set_stream(stream);
    return cuda_impl->get_cuda_buffer().get_write_handle(stream);
  }

  size_t byte_count = buffer.size() * sizeof(T);
  auto promoted = detail::allocate_cuda_buffer_shared(
    byte_count, kind.value_or(default_storage_kind()));
  auto * promoted_impl = detail::cuda_impl_of(*promoted);
  promoted_impl->set_stream(stream);
  auto wh = promoted_impl->get_cuda_buffer().get_write_handle(stream);
  wh.set_promoted_buffer(std::move(promoted));
  return wh;
}

/// \brief Acquire a read handle for a CUDA-backed buffer.
/// \details If \p buffer is already CUDA-backed, returns a read handle
/// directly. If the buffer is non-CUDA (e.g. CPU-backed), a new CUDA-backed
/// \c rosidl::Buffer<uint8_t> is allocated, the source contents are copied
/// host-to-device, and a read handle for the new buffer is returned.
/// \param kind backend of the promoted buffer; default_storage_kind() unless
/// given. Ignored when \p buffer is already CUDA-backed.
template<typename T>
ReadHandle from_input_buffer(
  const rosidl::Buffer<T> & buffer,
  cudaStream_t stream,
  std::optional<StorageKind> kind = std::nullopt)
{
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_input_buffer called on buffer with null implementation");
  }
  if (buffer.size() == 0) {
    throw CudaError("from_input_buffer called on empty buffer");
  }
  const auto * cuda_impl = dynamic_cast<const CudaBufferImpl<T> *>(impl);
  if (cuda_impl) {
    return cuda_impl->get_cuda_buffer().get_read_handle(stream);
  }

  size_t byte_count = buffer.size() * sizeof(T);
  auto promoted = detail::allocate_cuda_buffer_shared(
    byte_count, kind.value_or(default_storage_kind()));
  auto * promoted_impl = detail::cuda_impl_of(*promoted);
  {
    auto wh = promoted_impl->get_cuda_buffer().get_write_handle(stream);
    CUDA_CHECK(cudaMemcpyAsync(
      wh.get_ptr(), buffer.data(), byte_count, cudaMemcpyHostToDevice, stream));
  }
  auto rh = promoted_impl->get_cuda_buffer().get_read_handle(stream);
  rh.set_promoted_buffer(std::move(promoted));
  return rh;
}

/// \brief Copy \p byte_count bytes from \p src into the memory referenced by \p wh
/// using \p stream and the given \p kind. Does not allocate.
inline void to_buffer(
  const void * src,
  size_t byte_count,
  WriteHandle & wh,
  cudaStream_t stream,
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice)
{
  if (byte_count == 0 || !src || !wh.get_ptr()) {
    return;
  }
  CUDA_CHECK(cudaMemcpyAsync(wh.get_ptr(), src, byte_count, kind, stream));
}

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_API_HPP_
