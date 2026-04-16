# rosidl_buffer_backends

CUDA and PyTorch buffer backend implementations for `rosidl::Buffer`,
enabling zero-copy GPU memory sharing between ROS 2 publishers and
subscribers.

## Packages

- **cuda_buffer** -- Core CUDA buffer library (VMM-backed IPC memory pool,
  host endpoint manager, ReadHandle/WriteHandle with CUDA event sync).
- **cuda_buffer_backend** -- BufferBackend plugin for CUDA IPC transport.
- **cuda_buffer_backend_msgs** -- ROS 2 message definitions for CUDA buffer
  descriptors.
- **libtorch_vendor** -- Vendor package that downloads and installs the
  pre-built LibTorch C++ distribution.
- **torch_buffer** -- Device-agnostic PyTorch buffer library wrapping device
  backends with tensor metadata (shape, strides, dtype).
- **torch_buffer_backend** -- BufferBackend plugin for PyTorch tensors.
- **torch_buffer_backend_msgs** -- ROS 2 message definitions for Torch buffer
  descriptors.

## Prerequisites

- CUDA Toolkit (>= 11.8)
- LibTorch is provided automatically by `libtorch_vendor` at build time

## Build

```bash
# Build CUDA backend
pixi run build cuda_buffer_backend

# Build Torch backend (includes libtorch_vendor download)
pixi run build torch_buffer_backend
```

## API overview

### CUDA buffer backend (`cuda_buffer_backend`)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

// Publisher: allocate + write directly via kernel
auto msg = cuda_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(byte_count);
auto wh = cuda_buffer_backend::from_buffer(msg.data, stream);
my_kernel<<<...>>>(wh.get_ptr(), ...);

// Publisher: create a CUDA buffer from a host pointer
msg.data = cuda_buffer_backend::to_buffer(host_ptr, byte_count, stream);

// Subscriber: get read handle (waits on write event)
auto rh = cuda_buffer_backend::from_buffer(msg->data, stream);
use_data<<<...>>>(rh.get_ptr(), ...);

// Promote any buffer to CUDA (e.g. CPU fallback)
auto cuda_buf = cuda_buffer_backend::to_buffer(msg->data, stream);
```

### Torch buffer backend (`torch_buffer_backend`)

```cpp
#include "torch_buffer/torch_buffer_api.hpp"

// Publisher: create a torch buffer from a tensor
msg.data = torch_buffer_backend::to_buffer(tensor);

// Subscriber: get a tensor view
at::Tensor t = torch_buffer_backend::from_buffer(msg->data);

// Promote any buffer to torch (e.g. CPU fallback)
auto torch_buf = torch_buffer_backend::to_buffer(msg->data);
at::Tensor t = torch_buffer_backend::from_buffer(torch_buf);
```

## License

Apache-2.0
