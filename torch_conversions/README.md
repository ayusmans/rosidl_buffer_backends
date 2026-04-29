# torch_conversions (DLPack-aligned)

Header-only helper library that converts between a DLPack-shaped ROS 2
message (`tensor_msgs/Tensor`) and an `at::Tensor`, riding on top of
whichever `rosidl::Buffer` storage backend is registered at runtime.

The message schema follows [DLPack](https://dmlc.github.io/dlpack/latest/)
exactly, so any DLPack-compatible framework (PyTorch, TensorFlow, JAX,
CuPy, ONNX Runtime, TensorRT, MXNet, RAPIDS, ...) can plug in via a thin
wrapper and interoperate over the wire without re-encoding metadata.

## Packages

| Package | Description |
|---|---|
| `tensor_msgs` | `Tensor.msg` definition: DLPack-aligned `{dtype_code, dtype_bits, dtype_lanes}`, `{device_type, device_id}`, `shape[]`, `strides[]`, `byte_offset`, `data[]`. |
| `torch_conversions` | Header-only library: allocation, `at::Tensor` ↔ `Tensor.msg` conversion, DLPack export, and CUDA stream helpers. |

There is no pluginlib plugin, no `BufferImplBase` subclass, and no custom
descriptor. The `uint8[] data` field maps to `rosidl::Buffer<uint8_t>`,
which transparently uses `cuda_buffer_backend` for GPU zero-copy when
both peers support it and falls back to CPU CDR otherwise.

## The `Tensor.msg` schema

```
# DLDataType
uint8  dtype_code        # DLPack DLDataTypeCode: 0=Int, 1=UInt, 2=Float, 4=BFloat, 6=Bool, ...
uint8  dtype_bits        # 8, 16, 32, 64, ...
uint16 dtype_lanes       # SIMD lanes; 1 for plain scalar

# DLDevice
int32 device_type        # DLPack DLDeviceType: 1=CPU, 2=CUDA, 3=CUDAHost, 10=ROCm, 13=CUDAManaged, ...
int32 device_id          # device ordinal

# DLTensor
int64[] shape
int64[] strides          # empty = contiguous (DLPack nullptr convention)
uint64  byte_offset      # view offset into `data`

# Underlying storage (may be larger than numel * element_size for views)
uint8[] data
```

Field-for-field transcription of `DLTensor`. Producing or consuming a
`DLManagedTensor` on either side is one helper call.

## Build

```bash
# CUDA path (recommended): build cuda_buffer_backend first.
colcon build --symlink-install --packages-up-to cuda_buffer_backend
source install/setup.sh

colcon build --symlink-install --packages-up-to torch_conversions
source install/setup.sh
```

## API reference

All entry points are in namespace `torch_conversions`. `TensorMsg` is a
type alias for `tensor_msgs::msg::Tensor`.

### Allocation

```cpp
TensorMsg allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt);
```

Returns a `Tensor` message with DLPack metadata populated and `data`
sized exactly to `prod(shape) * dtype.bytesize` bytes. `byte_offset` is
0; the buffer is contiguous (`strides` set to row-major).

`device` defaults to CUDA when LibTorch reports CUDA available, otherwise
CPU. The `data` buffer is a `cuda_buffer_backend::CudaBufferImpl` for
CUDA messages (zero-copy IPC eligible) and a CPU `rosidl::Buffer` for
CPU messages.

### `TensorMsg` ↔ `at::Tensor`

```cpp
// Mutable: returns a tensor that aliases msg.data and is safe to write to.
at::Tensor from_tensor_msg(TensorMsg & msg);

// Const: zero-copy view by default; set clone=true for an independent copy.
at::Tensor from_tensor_msg(const TensorMsg & msg, bool clone = true);

// Copy a torch tensor's data into an existing TensorMsg (also fills metadata).
void to_tensor_msg(TensorMsg & msg, const at::Tensor & tensor);
```

`from_tensor_msg(const &)` is the path subscribers use; `clone=true` (the
default) protects against mutating the producer's storage when the buffer
is shared via CUDA IPC. Pass `clone=false` only when the caller is
certain the tensor will be treated as read-only.

### CUDA stream guard

```cpp
class StreamGuard;                   // RAII over c10::cuda::CUDAStreamGuard
StreamGuard set_stream();            // uses c10::cuda::getStreamFromPool()
```

Wrap the publisher/subscriber callback body in a `StreamGuard` when
running on CUDA; this routes torch ops onto a non-default stream so the
event-based handshake in `cuda_buffer_backend` (write_event +
`cudaStreamWaitEvent`) actually has work to synchronize against. On CPU
builds the guard is a no-op.

### DLPack export

For interop with non-torch frameworks, the bridge can hand out a
`DLManagedTensor*` directly:

```cpp
DLManagedTensor * make_dlpack_read(const TensorMsg & msg, cudaStream_t = nullptr);
DLManagedTensor * make_dlpack_write(TensorMsg & msg, cudaStream_t = nullptr);

// Direction inferred from const-ness of msg.
DLManagedTensor * to_dlpack(const TensorMsg & msg, cudaStream_t = nullptr);
DLManagedTensor * to_dlpack(      TensorMsg & msg, cudaStream_t = nullptr);

// RAII variants for callers that don't immediately hand the pointer to
// a framework's `from_dlpack` (which would take ownership itself).
using DlpackPtr = std::unique_ptr<DLManagedTensor, DlpackDeleter>;
DlpackPtr to_dlpack_owned(const TensorMsg & msg, cudaStream_t = nullptr);
DlpackPtr to_dlpack_owned(      TensorMsg & msg, cudaStream_t = nullptr);
```

The returned `DLManagedTensor` keeps a `Read`/`WriteHandle` alive for its
lifetime, so the `cuda_buffer_backend` event-sync machinery stays valid
until the importing framework destroys the tensor. The `byte_offset` of
`Tensor.msg` is baked into the returned `data` pointer (and the DLTensor
field is set to 0) for portability across DLPack importers.

### DLPack dtype / device helpers

```cpp
DLDataType   dl_dtype_from_scalar(at::ScalarType);
at::ScalarType scalar_from_dl_dtype(DLDataType);
size_t       dl_dtype_bytesize(DLDataType);
void         set_dtype(TensorMsg & m, DLDataType d);   // sets the 3 dtype fields
```

`device_type` / `device_id` follow DLPack's `DLDeviceType` enum
(`kDLCPU=1`, `kDLCUDA=2`, `kDLCUDAHost=3`, `kDLROCm=10`,
`kDLCUDAManaged=13`, ...). `allocate_tensor_msg` sets these for you.

## Examples

### Publisher

```cpp
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/tensor.hpp"

void timer_cb()
{
  auto guard = torch_conversions::set_stream();

  auto msg = torch_conversions::allocate_tensor_msg(
    {height, width, 3}, torch::kByte);   // CUDA-backed when available

  {
    at::Tensor t = torch_conversions::from_output_tensor_msg(msg);
    render_pipeline(t);                  // runs on the guarded stream
  }

  publisher_->publish(msg);
}
```

### Subscriber

```cpp
void cb(const tensor_msgs::msg::Tensor::SharedPtr msg)
{
  auto guard = torch_conversions::set_stream();

  // Default clone=true: independent tensor, safe to mutate.
  at::Tensor in = torch_conversions::from_input_tensor_msg(*msg);
  auto out = model_(in);

  // Or zero-copy view (caller guarantees no in-place writes).
  at::Tensor view = torch_conversions::from_input_tensor_msg(*msg, /*clone=*/false);
}
```

### Filling metadata from an existing tensor

```cpp
at::Tensor t = compute_something();             // arbitrary at::Tensor

tensor_msgs::msg::Tensor msg;
torch_conversions::to_tensor_msg(msg, t);        // copies data, fills shape/strides/dtype/device
publisher_->publish(msg);
```

`to_tensor_msg` copies tensor data into `msg.data` and sets all DLPack
metadata fields to match `t`.

## Testing

```bash
colcon test --packages-select tensor_msgs torch_conversions
colcon test-result --verbose
```

Coverage:

- `test_torch_conversions.cpp` — gtest covering CPU and CUDA paths,
  DLPack round-trips, dtype helpers, `byte_offset` view materialization.
- `test_torch_tensor_intra_pubsub_fastrtps_launch.py` — single-process
  pub/sub through RMW, exercises CUDA IPC where available.
- `test_torch_tensor_inter_pubsub_fastrtps_launch.py` — separate-process
  pub/sub, exercises the FD handshake or CPU fallback.
- `test_torch_tensor_multi_pub_fastrtps_launch.py` — N-to-1 fan-in.
- `test_torch_tensor_multi_sub_fastrtps_launch.py` — 1-to-N fan-out.

## License

Apache-2.0
