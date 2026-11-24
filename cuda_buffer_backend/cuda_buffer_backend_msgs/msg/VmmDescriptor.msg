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

# IPC metadata for VMM-based zero-copy sharing (StorageKind::Vmm): where to
# fetch the file descriptor of one block of the publisher's pool.

int32 pid                      # Publisher process ID
uint32 block_id                # Block ID in the publisher's pool
uint64 block_size              # Block size (for import mapping)
string socket_path             # Unix socket path to receive block FD

# CUDA IPC event for inter-process GPU synchronization; all zero if the
# producer recorded no event.
uint8[64] event_handle         # cudaIpcEventHandle_t (64 bytes)
