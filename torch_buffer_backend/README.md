# torch_buffer_backend

PyTorch buffer backend plugin for the ROS 2 Buffer system. Wraps CUDA/CPU device buffers with tensor metadata (shape, strides, dtype) and provides `allocate_msg`, `from_buffer`, and `to_buffer` APIs for zero-copy GPU tensor sharing.

For setup instructions, see the [ros2 meta repo](https://github.com/yuanknv/ros2).

## Prerequisites

LibTorch (the C++ distribution of PyTorch) must be installed before building.

1. Download from <https://pytorch.org/get-started/locally/> (select LibTorch, C++, your CUDA version)
2. Extract and set `CMAKE_PREFIX_PATH`:

```bash
unzip libtorch-cxx11-abi-shared-with-deps-*.zip -d /opt/
export CMAKE_PREFIX_PATH="/opt/libtorch:$CMAKE_PREFIX_PATH"
```

For pixi-based workspaces, add to `pixi.toml` activation env:

```toml
[target.linux-64.activation.env]
CMAKE_PREFIX_PATH="$PIXI_PROJECT_ROOT/path/to/libtorch:$CMAKE_PREFIX_PATH"
```

## Build

If you want GPU acceleration, build your device backend packages first
(e.g. `cuda_buffer_backend`). The torch packages auto-detect available
device backends at compile time.

```bash
# 1. (Optional) Build device backends for GPU support
pixi run build cuda_buffer_backend

# 2. Build torch packages
pixi run build torch_buffer_backend
```

If no device backend is installed, torch_buffer falls back to CPU
automatically.

> **Note:** Tests validate the CPU backend path only, since
> `torch_buffer_backend` has no explicit dependency on any device backend
> package. To verify GPU IPC end-to-end, use the
> [torch_backend_demo](https://github.com/yuanknv/torch_backend_demo).

## Packages

| Package | Description |
|---|---|
| `torch_buffer` | Core TorchBufferImpl wrapping device buffers with tensor metadata, and user-facing `allocate_msg`/`from_buffer`/`to_buffer` APIs |
| `torch_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `torch_buffer_backend_msgs` | ROS 2 message definition for `TorchBufferDescriptor` |

## Usage

### Stream Setup

The user is responsible for setting a non-default CUDA stream before calling
`allocate_msg`, `to_buffer`, or `from_buffer`. This ensures the same stream is
shared between buffer operations and the user's own operators (model inference,
custom kernels, etc.), enabling correct event-based asynchronous synchronization.

A convenience helper is provided:

```cpp
torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();
// All buffer operations AND user operators in this scope share the same stream.
```

This auto-detects CUDA/ROCm and sets a non-default stream from the pool. On
CPU-only builds it is a no-op. If no stream is set, a warning is logged and
operations fall back to synchronous execution on the default stream.

### Device Selection

`allocate_msg` accepts an optional device parameter. If omitted, it auto-detects
the best available device backend at runtime:

```cpp
// Auto-detect the best available device backend
auto msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte);

// Explicit device override (when the caller needs a specific device)
auto msg_gpu = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte, c10::kCUDA);
```

### Publisher (direct write, zero-copy)

```cpp
#include "torch_buffer/torch_buffer_api.hpp"
#include "sensor_msgs/msg/image.hpp"

torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

{
  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  my_pipeline(output);  // user operators run on the same stream
}  // tensor destroyed -> WriteHandle records write event on stream

publisher->publish(msg);
```

### Publisher (copy from existing tensor)

Use `to_buffer` when you have a tensor in PyTorch's own memory and need to copy it
into the pre-allocated buffer. This triggers a device-to-device memcpy from the
tensor's memory to the VMM pool buffer:

```cpp
torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte);

at::Tensor result = model(input);  // runs on same stream as buffer ops
torch_buffer_backend::to_buffer(msg.data, result);  // D2D memcpy into VMM pool buffer

publisher->publish(msg);
```

For zero-copy (no D2D memcpy), use `from_buffer` to write directly into the pool
buffer instead:

```cpp
{
  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  model.forward_into(output);  // writes directly to VMM pool memory
}
```

### Subscriber (read input tensor)

```cpp
#include "torch_buffer/torch_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  const rosidl::Buffer<uint8_t> & data = msg->data;
  at::Tensor input = torch_buffer_backend::from_buffer(data);
  // ReadHandle waits on publisher's write event, then records read event on destroy

  at::Tensor result = model(input);  // user inference on same stream
}
```

## Event Lifecycle

All CUDA synchronization is managed by the backend via WriteHandle/ReadHandle RAII:

| Stage | Mechanism |
|---|---|
| Publisher writes to buffer | WriteHandle in tensor deleter records write event on destroy |
| Subscriber reads from buffer | ReadHandle waits on write event at construction, records read event on destroy |
| Buffer recycled to pool | CudaBuffer destructor waits on all read events before returning to pool |

No `cudaStreamSynchronize` in the pipeline.

## Inter-process Behavior

For inter-process communication, the TorchBufferDescriptor carries the inner device buffer via endpoint-aware serialization:

| Condition | Path |
|---|---|
| Same host, CUDA VMM available | Zero-copy via CUDA IPC (nested CudaBufferDescriptor with IPC handles) |
| CPU buffer or VMM unavailable | CPU serialization via CDR |

## License

Apache-2.0
