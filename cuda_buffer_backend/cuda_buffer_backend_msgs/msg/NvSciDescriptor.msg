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

# IPC metadata for NvSci-based zero-copy sharing (StorageKind::NvSci). The
# buffer and sync objects themselves travel over the NvSciIpc channel between
# the two processes; the ids say which of them this message refers to. A
# descriptor for another process is valid for that one subscriber only.

int32 pid                      # Publisher process ID
uint64 buffer_id               # Block id in the publisher process
uint32 sync_channel_id         # Sync object the fence belongs to
bool has_fence                 # False if the buffer was never written
uint8[56] fence                # NvSciSyncFenceIpcExportDescriptor (56 bytes)

# The publisher's id of the NvSciIpc connection the objects were sent on (0
# within a process). A subscriber refuses a descriptor of another connection.
uint64 connection

# The publisher numbers the descriptors it sends to each subscriber; the
# subscriber reports back the highest one it has handled, and the publisher
# reuses a block once nobody it was sent to can still import it.
uint32 stream
uint64 seq
