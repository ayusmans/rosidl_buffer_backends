# cuda_buffer_backend

CUDA buffer backend plugin for the ROS 2 Buffer system. Enables zero-copy GPU memory sharing between publishers and subscribers on the same host using CUDA VMM (Virtual Memory Management).

For setup instructions, see the [ros2 meta repo](https://github.com/yuanknv/ros2).

## Build

```bash
pixi run build "cuda_buffer cuda_buffer_backend"
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
cudaMemsetAsync(wh.get_ptr(), 0, data_size, stream);
my_kernel<<<...>>>(wh.get_ptr(), ...);  // user operators on same stream

publisher->publish(msg);
// wh destructor records write_event on stream when it goes out of scope
```

### Publisher (copy from existing pointer)

Use `to_buffer` when you have a pre-existing device pointer and need to
copy it into the pre-allocated buffer. This triggers a device-to-device memcpy.

```cpp
const size_t data_size = 640 * 480 * 3;

sensor_msgs::msg::Image msg =
  cuda_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(data_size);

cuda_buffer_backend::WriteHandle wh =
  cuda_buffer_backend::from_buffer(msg.data, stream);
// D2D memcpy from gpu_ptr into the pool buffer
cuda_buffer_backend::to_buffer(gpu_ptr, data_size, wh, stream);

publisher->publish(msg);
// wh destructor records write_event on stream when it goes out of scope
```

### Subscriber (read from buffer)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  const rosidl::Buffer<uint8_t> & data = msg->data;
  cuda_buffer_backend::ReadHandle rh =
    cuda_buffer_backend::from_buffer(data, stream);
  // ReadHandle constructor waits on publisher's write_event automatically

  cudaMemcpyAsync(host_buf, rh.get_ptr(), msg->data.size(),
    cudaMemcpyDeviceToHost, stream);
}  // ReadHandle destructor signals publisher that GPU work is complete
```

### `from_buffer` handle rules

`from_buffer` returns a **WriteHandle** when called with a non-const buffer, or a
**ReadHandle** when called with a const buffer. The overload is selected at compile
time based on const-ness of the reference you pass:

```cpp
// Write path -- use on a freshly allocated message before publish:
cuda_buffer_backend::WriteHandle wh = cuda_buffer_backend::from_buffer(msg.data, stream);

// Read path -- use const reference in the subscriber callback:
const rosidl::Buffer<uint8_t> & data = msg->data;
cuda_buffer_backend::ReadHandle rh = cuda_buffer_backend::from_buffer(data, stream);
```

- A **WriteHandle** can only be acquired once per buffer. It is intended for the
  publisher to fill a freshly allocated message. Attempting to acquire a second
  WriteHandle (or acquiring one after the write has been finalized) throws
  `CudaError`.
- To read a received buffer, always pass a **const reference**. If your subscriber
  callback takes a non-const `SharedPtr` or `UniquePtr`, cast to const before
  calling `from_buffer`:

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, same GPU, same user | Zero-copy via CUDA VMM IPC |
| Different GPU, different user, different host, or VMM unavailable | CPU fallback via `to_cpu()` |

The publisher's pool checks a shared-memory refcount before recycling a block, ensuring all IPC subscribers have released their handles.

## License

Apache-2.0
