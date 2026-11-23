# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

uint64 size                    # Allocation size in bytes
string element_type_name       # Type name for validation (e.g., "h" for uint8_t)
int32 device_id                # CUDA device ID of the source GPU

# IPC metadata for VMM-based zero-copy sharing
bool use_ipc                   # True if IPC handles are valid
int32 vmm_pid                  # Publisher process ID
uint32 vmm_block_id            # Block ID in the publisher's pool
uint64 vmm_block_size          # Block size (for import mapping)
string vmm_socket_path         # Unix socket path to receive block FD
uint64 ipc_uid                 # Unique ID per publish for staleness detection

# CUDA IPC event for inter-process GPU synchronization
uint8[64] ipc_event_handle     # cudaIpcEventHandle_t (64 bytes)

# Which storage backend produced this descriptor
# (cuda_buffer_backend::StorageKind). The vmm_* fields and ipc_event_handle are
# VMM only; ipc_uid is used by every backend.
uint8 STORAGE_KIND_VMM=0
uint8 STORAGE_KIND_NVSCI=1
uint8 storage_kind

# IPC metadata for NvSci-based zero-copy sharing. The buffer and sync objects
# themselves travel over the NvSciIpc channel between the two processes; the
# ids say which of them this message refers to. A descriptor for another
# process is valid for that one subscriber only.
int32 nvsci_pid                # Publisher process ID
uint64 nvsci_buffer_id         # Block id in the publisher process
uint32 nvsci_sync_channel_id   # Sync object the fence belongs to
bool nvsci_has_fence           # False if the buffer was never written
uint8[56] nvsci_fence          # NvSciSyncFenceIpcExportDescriptor (56 bytes)
# The publisher's id of the NvSciIpc connection the objects were sent on (0
# within a process). A subscriber refuses a descriptor of another connection.
uint64 nvsci_connection
# The publisher numbers the descriptors it sends to each subscriber; the
# subscriber reports back the highest one it has handled, and the publisher
# reuses a block once nobody it was sent to can still import it.
uint32 nvsci_stream
uint64 nvsci_seq
