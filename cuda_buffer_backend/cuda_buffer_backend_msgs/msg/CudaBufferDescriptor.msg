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
uint64 ipc_uid                 # Unique ID per publish for staleness detection

# Which storage backend produced this descriptor
# (cuda_buffer_backend::StorageKind). The field of that backend below holds
# exactly one entry and every other backend's field is empty, so a descriptor
# carries only its own backend's metadata.
#
# Adding a backend adds an enumerator here, a <Backend>Descriptor message, a
# bounded field below, and a new CudaBufferBackend::kDescriptorVersion: peers
# that disagree on this layout must not exchange descriptors.
uint8 STORAGE_KIND_VMM=0
uint8 STORAGE_KIND_NVSCI=1
uint8 storage_kind

VmmDescriptor[<=1] vmm
NvSciDescriptor[<=1] nvsci
