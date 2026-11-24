# cuda_buffer_examples

A talker and a listener that send CUDA buffers allocated from the two storage
backends, CUDA VMM and NvSci, side by side.

- `BufferTalker` (`buffer_talker_node`) publishes a `sensor_msgs/Image` whose
  data is a CUDA buffer from the backend named by its `storage` parameter,
  filled on the GPU with a byte that changes with every message.
- `BufferListener` (`buffer_listener_node`) subscribes to several topics, reads
  every payload back from the GPU, checks every byte, and reports how each
  message arrived: zero-copy `vmm`, zero-copy `nvsci`, or `cpu` (serialized,
  because the subscriber cannot import that backend). It also publishes the
  counts as JSON on `buffer_stats`.

## Run

```bash
ros2 launch cuda_buffer_examples demo.launch.py                       # listener in another process
ros2 launch cuda_buffer_examples demo.launch.py layout:=same_process  # everything in one process
```

Two talkers publish on `/buffer_vmm` (VMM) and `/buffer_nvsci` (NvSci) from one
process, so both backends are used in the same process. The listener prints
one line per topic every two seconds:

```text
[listener]: /buffer_vmm: 40 received (zero-copy vmm=40 nvsci=0, cpu=0), 0 corrupt, mean latency 0.40 ms
[listener]: /buffer_nvsci: 40 received (zero-copy vmm=0 nvsci=40, cpu=0), 0 corrupt, mean latency 0.25 ms
```

Launch arguments: `layout` (`cross_process` or `same_process`), `size_bytes`
(default 1 MiB) and `rate_hz` (default 10).

The nodes can also be run on their own:

```bash
ros2 run cuda_buffer_examples buffer_talker_node --ros-args -p storage:=nvsci -p topic:=buffer_nvsci
ros2 run cuda_buffer_examples buffer_listener_node --ros-args -p topics:="['buffer_nvsci']"
```

A talker with an empty `storage` uses the process default, which
`CUDA_BUFFER_STORAGE` sets (`vmm` if unset). A talker whose backend is not
available in its process logs a warning and uses the default instead.

## What to expect

| | `/buffer_vmm` | `/buffer_nvsci` |
|---|---|---|
| `same_process`, any machine with NvSci | zero-copy `vmm` | zero-copy `nvsci` |
| `cross_process`, NvSciIpc endpoint pool set up (Jetson) | zero-copy `vmm` | zero-copy `nvsci`, after the first few messages |
| `cross_process`, no NvSciIpc (e.g. x86 without `/dev/nvsciipc`) | zero-copy `vmm` | `cpu` |
| No NvSci at all | zero-copy `vmm` | zero-copy `vmm` (the NvSci talker falls back) |

The first few NvSci messages to a new subscriber process take the CPU path:
NvSci can only share a buffer with a process whose attributes it was
allocated with, and those arrive just after the two processes discover each
other.

## Setting up NvSci across processes

Each pair of processes that share NvSci buffers needs an NvSciIpc
`INTER_PROCESS` endpoint pair, claimed at runtime from the host's endpoint pool
inventory. Append
[`config/nvsciipc_cuda_buffer.cfg`](config/nvsciipc_cuda_buffer.cfg) to
`/etc/nvsciipc.cfg` (on Jetson, drop the last column), restart the NvSciIpc
service, and install the inventory that lists the same pairs,
[`config/nvsci_endpoint_pool.json`](config/nvsci_endpoint_pool.json):

```bash
sudo sh -c 'cat config/nvsciipc_cuda_buffer.cfg >> /etc/nvsciipc.cfg'
sudo systemctl restart nv_nvsciipc_init
sudo install -D -m 644 config/nvsci_endpoint_pool.json /etc/cuda_buffer/nvsci_endpoint_pool.json
```

At startup each process logs either
`NvSci cross-process sharing enabled with 16 endpoint pairs` or the reason it
is disabled.

## Test

```bash
colcon test --packages-select cuda_buffer_examples
```

The launch test runs both layouts and checks that no payload is corrupt, that
VMM buffers arrive zero-copy, that NvSci buffers never arrive as VMM, and, in
one process, that NvSci buffers arrive zero-copy.
