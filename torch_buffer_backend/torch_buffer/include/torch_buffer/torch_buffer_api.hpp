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

#ifndef TORCH_BUFFER__TORCH_BUFFER_API_HPP_
#define TORCH_BUFFER__TORCH_BUFFER_API_HPP_

#include <torch/torch.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <rcutils/logging_macros.h>

#include "rosidl_buffer/buffer.hpp"
#include "torch_buffer/torch_buffer_impl.hpp"
#include "torch_buffer/torch_buffer_utils.hpp"

#include <c10/core/StreamGuard.h>

#if __has_include("cuda_buffer/cuda_buffer_api.hpp")
#include <c10/cuda/CUDAStream.h>
#include "cuda_buffer/cuda_buffer_api.hpp"
#define TORCH_BUFFER_DEVICE_CUDA
#endif

namespace torch_buffer_backend
{

namespace detail
{

inline std::optional<c10::Stream> get_non_default_stream()
{
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (torch::cuda::is_available()) {
    return c10::cuda::getStreamFromPool();
  }
#endif
  return std::nullopt;
}

inline c10::DeviceType default_device()
{
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (torch::cuda::is_available()) {
    return c10::kCUDA;
  }
#endif
  return c10::kCPU;
}

}  // namespace detail

/// \brief RAII guard that sets a non-default CUDA stream for the current scope.
class StreamGuard
{
public:
  StreamGuard()
  : guard_(detail::get_non_default_stream())
  {
  }

private:
  c10::OptionalStreamGuard guard_;
};

/// \brief Create a StreamGuard that activates a non-default CUDA stream.
inline StreamGuard set_stream()
{
  return StreamGuard();
}

/// \brief Allocate a ROS message with a torch-backed buffer.
/// \tparam MsgT ROS message type whose `data` field is a `rosidl::Buffer<uint8_t>`.
/// \param shape Tensor shape.
/// \param dtype Scalar type (e.g. `at::kByte`, `at::kFloat`).
/// \param device Target device (defaults to CUDA if available, otherwise CPU).
template<typename MsgT>
MsgT allocate_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt)
{
  c10::DeviceType dev = device.value_or(detail::default_device());

  int64_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }
  size_t byte_count = static_cast<size_t>(numel) * scalar_type_size(dtype);

  std::vector<int64_t> contiguous_strides(shape.size());
  int64_t stride_val = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    contiguous_strides[i] = stride_val;
    stride_val *= shape[i];
  }

  rosidl::Buffer<uint8_t> device_buffer;

#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (dev == c10::kCUDA) {
    auto cuda_impl = std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    device_buffer = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
  } else
#endif
  if (dev == c10::kCPU) {
    device_buffer.resize(byte_count);
  } else {
    throw std::runtime_error(
      "allocate_msg: unsupported device type " + std::to_string(static_cast<int>(dev)));
  }

  auto torch_impl = std::make_unique<TorchBufferImpl<uint8_t>>(
    std::move(device_buffer), shape, contiguous_strides,
    scalar_type_to_string(dtype));

  MsgT msg;
  msg.data = rosidl::Buffer<uint8_t>(std::move(torch_impl));
  return msg;
}

/// \brief Copy a PyTorch tensor into a pre-allocated torch buffer.
/// \param buffer Destination buffer (must be allocated via allocate_msg).
/// \param tensor Source tensor; will be made contiguous if needed.
inline void to_buffer(rosidl::Buffer<uint8_t> & buffer, const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }

  at::Tensor contig = tensor.contiguous();
  size_t byte_count = contig.numel() * contig.element_size();

  auto * torch_impl = const_cast<TorchBufferImpl<uint8_t> *>(
    static_cast<const TorchBufferImpl<uint8_t> *>(buffer.get_impl()));

  if (byte_count > torch_impl->byte_size()) {
    throw std::runtime_error(
      "to_buffer: tensor size (" + std::to_string(byte_count) +
      " bytes) exceeds allocated buffer (" + std::to_string(torch_impl->byte_size()) + " bytes)");
  }

  const std::string & backend = torch_impl->get_device_buffer().get_backend_type();

#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (backend == "cuda") {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto wh = cuda_buffer_backend::from_buffer(
      torch_impl->get_device_buffer(), stream);
    cudaMemcpyKind kind = contig.is_cuda() ?
      cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cuda_buffer_backend::to_buffer(
      contig.data_ptr(), byte_count, wh, stream, kind);
  } else
#endif
  if (backend == "cpu") {
    void * dst = torch_impl->get_device_buffer().data();
    std::memcpy(dst, contig.data_ptr(), byte_count);
  } else {
    throw std::runtime_error("to_buffer: unsupported backend '" + backend + "'");
  }

  torch_impl->set_metadata(
    contig.sizes().vec(), contig.strides().vec(),
    scalar_type_to_string(contig.scalar_type()));
}

namespace detail
{

// Returned tensor borrows memory from impl; caller must ensure
// the owning rosidl::Buffer outlives the tensor.
inline at::Tensor cpu_wrap(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  auto * ptr = const_cast<void *>(
    static_cast<const void *>(impl->get_device_buffer().data()));
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  return strides.empty() ?
         torch::from_blob(ptr, shape, opts) :
         torch::from_blob(ptr, shape, strides, opts);
}

#ifdef TORCH_BUFFER_DEVICE_CUDA

inline at::Tensor cuda_wrap_writable(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  auto * cuda_impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      impl->get_device_buffer().get_impl()));
  if (!cuda_impl) {
    throw std::runtime_error("from_buffer (write): device buffer is not a CudaBufferImpl");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (stream == nullptr) {
    RCUTILS_LOG_WARN_NAMED(
      "torch_buffer",
      "from_buffer (write): current CUDA stream is the default stream (nullptr). "
      "Event-based synchronization is disabled; all operations will be synchronous. "
      "Set a non-default stream with c10::cuda::CUDAStreamGuard before calling from_buffer.");
  }
  cuda_impl->set_stream(stream);
  auto wh = std::make_shared<cuda_buffer_backend::WriteHandle>(
    cuda_impl->get_cuda_buffer().get_write_handle(stream));
  void * ptr = static_cast<void *>(wh->get_ptr());
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  return strides.empty() ?
         torch::from_blob(ptr, shape, [wh](void *) {}, opts) :
         torch::from_blob(ptr, shape, strides, [wh](void *) {}, opts);
}

inline at::Tensor cuda_wrap_readable(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  const auto * cuda_impl =
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    impl->get_device_buffer().get_impl());
  if (!cuda_impl) {
    throw std::runtime_error("from_buffer (read): device buffer is not a CudaBufferImpl");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (stream == nullptr) {
    RCUTILS_LOG_WARN_NAMED(
      "torch_buffer",
      "from_buffer (read): current CUDA stream is the default stream (nullptr). "
      "Event-based synchronization is disabled; all operations will be synchronous. "
      "Set a non-default stream with c10::cuda::CUDAStreamGuard before calling from_buffer.");
  }
  auto rh = std::make_shared<cuda_buffer_backend::ReadHandle>(
    cuda_impl->get_cuda_buffer().get_read_handle(stream));
  void * ptr = const_cast<void *>(static_cast<const void *>(rh->get_ptr()));
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  return strides.empty() ?
         torch::from_blob(ptr, shape, [rh](void *) {}, opts) :
         torch::from_blob(ptr, shape, strides, [rh](void *) {}, opts);
}

#endif  // TORCH_BUFFER_DEVICE_CUDA

inline at::Tensor wrap_writable(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  const std::string & backend = impl->get_device_buffer().get_backend_type();
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (backend == "cuda") {
    return cuda_wrap_writable(impl, shape, strides, dtype);
  }
#endif
  if (backend == "cpu") {
    return cpu_wrap(impl, shape, strides, dtype);
  }
  throw std::runtime_error("from_buffer: unsupported backend '" + backend + "'");
}

inline at::Tensor wrap_readable(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  const std::string & backend = impl->get_device_buffer().get_backend_type();
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (backend == "cuda") {
    return cuda_wrap_readable(impl, shape, strides, dtype);
  }
#endif
  if (backend == "cpu") {
    return cpu_wrap(impl, shape, strides, dtype);
  }
  throw std::runtime_error("from_buffer: unsupported backend '" + backend + "'");
}

}  // namespace detail

/// \brief Get a writable tensor view of a torch buffer (uses stored metadata).
inline at::Tensor from_buffer(rosidl::Buffer<uint8_t> & buffer)
{
  if (buffer.empty()) {return {};}
  const auto * impl = static_cast<const TorchBufferImpl<uint8_t> *>(buffer.get_impl());
  at::ScalarType dtype = string_to_scalar_type(impl->dtype());
  return detail::wrap_writable(impl, impl->shape(), impl->strides(), dtype);
}

/// \brief Get a read-only tensor view of a torch buffer (uses stored metadata).
inline at::Tensor from_buffer(const rosidl::Buffer<uint8_t> & buffer)
{
  if (buffer.empty()) {return {};}
  const auto * impl = static_cast<const TorchBufferImpl<uint8_t> *>(buffer.get_impl());
  at::ScalarType dtype = string_to_scalar_type(impl->dtype());
  return detail::wrap_readable(impl, impl->shape(), impl->strides(), dtype);
}

/// \brief Get a writable tensor view with explicit shape, strides, and dtype.
inline at::Tensor from_buffer(
  rosidl::Buffer<uint8_t> & buffer,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides = {},
  at::ScalarType dtype = at::kByte)
{
  if (buffer.empty()) {return {};}
  const auto * impl = static_cast<const TorchBufferImpl<uint8_t> *>(buffer.get_impl());
  return detail::wrap_writable(impl, shape, strides, dtype);
}

/// \brief Get a read-only tensor view with explicit shape, strides, and dtype.
inline at::Tensor from_buffer(
  const rosidl::Buffer<uint8_t> & buffer,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides = {},
  at::ScalarType dtype = at::kByte)
{
  if (buffer.empty()) {return {};}
  const auto * impl = static_cast<const TorchBufferImpl<uint8_t> *>(buffer.get_impl());
  return detail::wrap_readable(impl, shape, strides, dtype);
}

}  // namespace torch_buffer_backend

#endif  // TORCH_BUFFER__TORCH_BUFFER_API_HPP_
