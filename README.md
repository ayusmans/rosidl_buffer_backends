# rosidl_buffer_backends

CUDA buffer backend implementation for `rosidl::Buffer`, enabling zero-copy
GPU memory sharing between ROS 2 publishers and subscribers, plus a
PyTorch-side helper library that builds on the same buffer infrastructure.

## Packages

- **cuda_buffer** -- Core CUDA buffer library (VMM-backed IPC memory pool,
  host endpoint manager, ReadHandle/WriteHandle with CUDA event sync).
- **cuda_buffer_backend** -- BufferBackend plugin for CUDA IPC transport.
- **cuda_buffer_backend_msgs** -- ROS 2 message definitions for CUDA buffer
  descriptors.
- **libtorch_vendor** -- Vendor package that downloads and installs the
  pre-built LibTorch C++ distribution.
- **tensor_msgs** -- DLPack-aligned `ExperimentalTensor.msg` definition.
- **torch_conversions** -- Header-only helper library that converts between
  `tensor_msgs/ExperimentalTensor` and `at::Tensor` and exposes DLPack import /
  export. Replaces the older `torch_buffer_backend` plugin approach with a
  plain message + bridge library that rides on top of whichever
  `rosidl::Buffer` backend is registered (CUDA when available, CPU
  otherwise).

## Prerequisites

- A ROS 2 Rolling development environment. See the upstream
  [Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
  guide for the canonical source-build flow, or use the pixi workflow
  shipped by the [`ros2/ros2`](https://github.com/ros2/ros2) meta-repo.
- CUDA Toolkit (>= 11.8) on the host.
- LibTorch: provided automatically by `libtorch_vendor` at build time if a
  system LibTorch isn't already visible.

Per-package build, test, and run details live in each package's README:

- [`cuda_buffer_backend/README.md`](cuda_buffer_backend/README.md)
- [`torch_conversions/README.md`](torch_conversions/README.md)

## API overview

### CUDA buffer backend (`cuda_buffer_backend`)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

// Publisher: allocate + write directly via kernel.
sensor_msgs::msg::Image msg;
msg.data = cuda_buffer_backend::allocate_buffer(byte_count);
{
  auto wh = cuda_buffer_backend::from_write_buffer(msg.data, stream);
  my_kernel<<<...>>>(wh.get_ptr(), ...);  // wh.get_ptr() returns uint8_t *
}  // wh destructor records the write event on `stream`

// Publisher: copy from an existing host/device pointer into a pre-allocated buffer.
{
  auto wh = cuda_buffer_backend::from_write_buffer(msg.data, stream);
  cuda_buffer_backend::to_buffer(host_ptr, byte_count, wh, stream,
    cudaMemcpyHostToDevice);
}

// Subscriber: read handle (waits on publisher's write event).
// from_read_buffer takes `const Buffer &` and accepts both const and mutable
// arguments; no `const Buffer & data = ...` alias is needed.
auto rh = cuda_buffer_backend::from_read_buffer(msg->data, stream);
use_data<<<...>>>(rh.get_ptr(), ...);  // rh.get_ptr() returns const uint8_t *

// Auto-promotion: passing a non-CUDA buffer allocates a fresh CUDA buffer
// and (for reads) copies H2D; the handle owns the new buffer via
// get_promoted_buffer().
auto rh_any = cuda_buffer_backend::from_read_buffer(cpu_or_other_buf, stream);
std::shared_ptr<rosidl::Buffer<uint8_t>> promoted = rh_any.get_promoted_buffer();
```

### Torch tensor API (`torch_conversions`)

```cpp
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

// Publisher: allocate a Tensor message (CUDA-backed if available).
auto msg = torch_conversions::allocate_tensor_msg(
  /*shape=*/{1080, 1920, 3}, torch::kUInt8, torch::kCUDA);

// Wrap as at::Tensor without copying and write into it.
at::Tensor t_out = torch_conversions::from_output_tensor_msg(msg);
my_pipeline(t_out);

// Subscriber: zero-copy read view of the received message.
at::Tensor t_in = torch_conversions::from_input_tensor_msg(msg);
```

The message schema carries DLPack's dtype / shape / stride / offset
metadata, while device placement is derived from the underlying
`rosidl::Buffer` backend. Any DLPack-compatible framework (PyTorch,
TensorFlow, JAX, CuPy, ONNX Runtime, ...) can interoperate over the wire by
converting to / from its own DLPack representation.

## License

Apache-2.0
