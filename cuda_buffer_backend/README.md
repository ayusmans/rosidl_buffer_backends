# cuda_buffer_backend

CUDA buffer backend plugin for the ROS 2 Buffer system. Enables zero-copy GPU memory sharing between publishers and subscribers on the same host using CUDA VMM (Virtual Memory Management).

## Build

```bash
pixi run build cuda_buffer_backend
```

## Packages

| Package | Description |
|---|---|
| `cuda_buffer` | Core CUDA buffer implementation: memory pool, IPC manager, host endpoint manager, and user-facing `allocate_msg`/`from_buffer`/`to_buffer` APIs |
| `cuda_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `cuda_buffer_backend_msgs` | ROS 2 message definition for `CudaBufferDescriptor` |

## Usage

### Publisher (direct write, zero-copy)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "sensor_msgs/msg/image.hpp"

const size_t data_size = 640 * 480 * 3;

sensor_msgs::msg::Image msg =
  cuda_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(data_size);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

cuda_buffer_backend::WriteHandle wh =
  cuda_buffer_backend::from_buffer(msg.data, stream);
my_kernel<<<...>>>(wh.get_ptr(), ...);

publisher->publish(msg);
// wh destructor records write_event on stream when it goes out of scope
```

### Publisher (from existing pointer)

Use `to_buffer` to create a new CUDA-backed buffer from a raw pointer:

```cpp
sensor_msgs::msg::Image msg;
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

msg.data = cuda_buffer_backend::to_buffer(gpu_ptr, data_size, stream);

publisher->publish(msg);
```

### Subscriber (read from buffer)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  const rosidl::Buffer<uint8_t> & data = msg->data;
  cuda_buffer_backend::ReadHandle rh =
    cuda_buffer_backend::from_buffer(data, stream);
  // ReadHandle constructor waits on publisher's write_event

  my_kernel<<<...>>>(rh.get_ptr(), ...);
}  // ReadHandle destructor signals publisher that GPU work is complete
```

### Subscriber (promote non-CUDA buffer)

Use `to_buffer` to promote a buffer from any backend (e.g. CPU fallback)
to CUDA:

```cpp
void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  auto gpu_data = cuda_buffer_backend::to_buffer(msg->data, stream);
  auto rh = cuda_buffer_backend::from_buffer(gpu_data, stream);
  my_kernel<<<...>>>(rh.get_ptr(), ...);
}
```

If the buffer is already CUDA-backed, `to_buffer` performs a D2D copy.
For CPU buffers, it copies H2D. In both cases the returned buffer is
a fresh CUDA allocation owned by the caller.

### `from_buffer` handle rules

`from_buffer` returns a **WriteHandle** when called with a non-const buffer, or a
**ReadHandle** when called with a const buffer. The overload is selected at compile
time based on const-ness of the reference:

```cpp
// Write path (publisher):
cuda_buffer_backend::WriteHandle wh = cuda_buffer_backend::from_buffer(msg.data, stream);

// Read path (subscriber):
const rosidl::Buffer<uint8_t> & data = msg->data;
cuda_buffer_backend::ReadHandle rh = cuda_buffer_backend::from_buffer(data, stream);
```

- A **WriteHandle** can only be acquired once per buffer. Attempting to acquire
  a second WriteHandle (or acquiring one after finalization) throws `CudaError`.
- To read a received buffer, always pass a **const reference**.

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, same GPU, same user | Zero-copy via CUDA VMM IPC |
| Different GPU, different user, different host, or VMM unavailable | CPU fallback via `to_cpu()` |

The publisher's pool checks a shared-memory refcount before recycling a block, ensuring all IPC subscribers have released their handles.

## License

Apache-2.0
