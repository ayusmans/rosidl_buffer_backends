# torch_conversions (DLPack-aligned)

Header-only helper library that converts between a DLPack-shaped ROS 2
message (`tensor_msgs/ExperimentalTensor`) and an `at::Tensor`, riding on top of
whichever `rosidl::Buffer` storage backend is registered at runtime.

The message schema follows [DLPack](https://dmlc.github.io/dlpack/latest/)
tensor metadata, so any DLPack-compatible framework (PyTorch, TensorFlow,
JAX, CuPy, ONNX Runtime, MXNet, RAPIDS, ...) can plug in via a thin wrapper
and interoperate over the wire without re-encoding shape / dtype metadata.

> **Status: experimental.** The message is named `ExperimentalTensor` on
> purpose. The schema is used internally to validate the buffer-backend
> design and may change before it is renamed to `Tensor` and stabilized.

## Packages

| Package | Description |
|---|---|
| `tensor_msgs` | `ExperimentalTensor.msg` definition: DLPack-aligned `{dtype_code, dtype_bits, dtype_lanes}`, `shape[]`, `strides[]`, `byte_offset`, `data[]`. |
| `torch_conversions` | Header-only library: allocation, `at::Tensor` ↔ `ExperimentalTensor.msg` conversion, DLPack export, and CUDA stream helpers. |

There is no pluginlib plugin, no `BufferImplBase` subclass, and no custom
descriptor. The `uint8[] data` field maps to `rosidl::Buffer<uint8_t>`,
which transparently uses `cuda_buffer_backend` for GPU zero-copy when
both peers support it and falls back to CPU CDR otherwise.

## The `ExperimentalTensor.msg` schema

```
# DLDataType
uint8  dtype_code        # DLPack DLDataTypeCode: 0=Int, 1=UInt, 2=Float, 4=BFloat, 6=Bool, ...
uint8  dtype_bits        # 8, 16, 32, 64, ...
uint16 dtype_lanes       # SIMD lanes; 1 for plain scalar

# DLTensor
int64[] shape
int64[] strides          # empty = contiguous (DLPack nullptr convention)
uint64  byte_offset      # view offset into `data`

# Underlying storage (may be larger than numel * element_size for views)
uint8[] data
```

The message carries DLPack's dtype / shape / stride / offset metadata. The
`DLDevice` fields are derived from the underlying `msg.data` buffer backend
when exporting to DLPack (`cpu` -> `kDLCPU`, `cuda` -> `kDLCUDA` plus the
pointer's CUDA device).

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
type alias for `tensor_msgs::msg::ExperimentalTensor`.

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
// Write path (publisher): aliases msg.data; the tensor is safe to write to.
// Requires a mutable msg so the write intent is explicit at the call site.
at::Tensor from_output_tensor_msg(TensorMsg & msg);

// Read path (subscriber): zero-copy view by default; clone=true gives an
// independent copy. Takes const &, so both const and mutable args are accepted.
at::Tensor from_input_tensor_msg(const TensorMsg & msg, bool clone = true);

// Copy an at::Tensor's data into a pre-sized TensorMsg and fill all DLPack
// metadata from the tensor.
void to_tensor_msg(TensorMsg & msg, const at::Tensor & tensor);
```

`from_input_tensor_msg` defaults `clone=true` so the subscriber gets an
independent tensor that is safe to mutate even when the underlying buffer
is shared via CUDA IPC. Pass `clone=false` when the caller guarantees
read-only use.

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

## Examples

### Publisher

```cpp
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

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
void cb(const tensor_msgs::msg::ExperimentalTensor::SharedPtr msg)
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

tensor_msgs::msg::ExperimentalTensor msg;
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

## License

Apache-2.0
