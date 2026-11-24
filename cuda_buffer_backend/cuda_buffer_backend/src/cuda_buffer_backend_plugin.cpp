// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cuda_buffer_backend/cuda_buffer_backend.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_ipc_manager.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"
#include "cuda_buffer/vmm_buffer.hpp"
#include "cuda_buffer_backend_msgs/msg/cuda_buffer_descriptor.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rcutils/logging_macros.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"

#ifdef CUDA_BUFFER_ENABLE_NVSCI
#include "cuda_buffer/nvsci_buffer.hpp"
#include "cuda_buffer/nvsci_runtime.hpp"
#endif


namespace cuda_buffer_backend
{

namespace
{

using cuda_buffer_backend_msgs::msg::CudaBufferDescriptor;
using cuda_buffer_backend_msgs::msg::NvSciDescriptor;
using cuda_buffer_backend_msgs::msg::VmmDescriptor;

// -- CUDA VMM transport --
//
// Everything that is specific to sharing a VmmBuffer between processes lives
// in these two functions: exporting a pool block and its write event into a
// descriptor, and mapping a described block back into a VmmBuffer.

/// \brief Fill ipc_uid and the vmm entry of \p descriptor for \p buffer.
/// \param peer_is_same_process queried only once the block is known to be shareable.
/// \return false if \p buffer cannot be shared over VMM IPC.
bool export_vmm_buffer(
  VmmBuffer & buffer,
  const std::function<bool()> & peer_is_same_process,
  CudaBufferDescriptor & descriptor)
{
  auto pool = get_or_create_global_pool();
  if (!pool || !pool->is_ipc_capable()) {
    return false;
  }

  CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(buffer.get_device_ptr());

  VmmBlock * block = pool->find_block_for_va(dev_ptr);
  if (!block || block->exported_fd < 0) {
    return false;
  }

  std::string socket_path = pool->register_block_for_ipc(block);
  if (socket_path.empty()) {
    return false;
  }

  VmmDescriptor vmm;
  vmm.pid = static_cast<int32_t>(getpid());
  vmm.block_id = block->block_id;
  vmm.block_size = block->size;
  vmm.socket_path = socket_path;
  const uint64_t uid = pool->assign_uid(block);

  buffer.finalize_write_handle();

  cudaEvent_t write_event = buffer.get_write_event();

  if (peer_is_same_process() && write_event) {
    RCUTILS_LOG_WARN_ONCE_NAMED("cuda_buffer_backend",
      "Same-process CUDA IPC requires cudaEventSynchronize on the publish path. "
      "Enable intra-process communication to avoid this overhead.");
    cudaEventSynchronize(write_event);
  }

  if (write_event) {
    cudaIpcEventHandle_t event_handle;
    CUDA_CHECK(cudaIpcGetEventHandle(&event_handle, write_event));
    std::memcpy(vmm.event_handle.data(), &event_handle, sizeof(cudaIpcEventHandle_t));
  } else {
    std::memset(vmm.event_handle.data(), 0, sizeof(cudaIpcEventHandle_t));
  }

  descriptor.ipc_uid = uid;
  descriptor.vmm.push_back(std::move(vmm));
  return true;
}

/// \brief Map the VMM block described by \p descriptor into this process.
/// \return nullptr if \p descriptor has no usable vmm entry.
/// \throws std::exception if the block cannot be imported.
std::unique_ptr<CudaBuffer> import_vmm_buffer(
  const CudaBufferDescriptor & descriptor,
  int local_device_id)
{
  if (descriptor.vmm.empty() || descriptor.vmm[0].socket_path.empty()) {
    return nullptr;
  }
  const VmmDescriptor & vmm = descriptor.vmm[0];

  if (descriptor.device_id != local_device_id) {
    throw CudaError("Cross-device IPC not implemented");
  }

  auto import_result = CudaVmmIPCManager::import_block(
    vmm.socket_path,
    vmm.pid,
    vmm.block_id,
    vmm.block_size,
    descriptor.ipc_uid);

  IPCMetadata * meta = import_result.ipc_meta;
  auto deleter = [meta](uint8_t *) {
      if (meta) {
        meta->refcount.fetch_sub(1, std::memory_order_release);
      }
    };
  auto imported_buffer = std::make_unique<VmmBuffer>(
    reinterpret_cast<void *>(import_result.va), descriptor.size * sizeof(uint8_t),
    descriptor.device_id, deleter);

  const bool has_event = std::any_of(
    vmm.event_handle.begin(), vmm.event_handle.end(), [](uint8_t b) {return b != 0;});

  if (has_event) {
    cudaIpcEventHandle_t event_handle;
    std::memcpy(&event_handle, vmm.event_handle.data(), sizeof(cudaIpcEventHandle_t));

    cudaEvent_t imported_event = nullptr;
    cudaError_t ev_err = cudaIpcOpenEventHandle(&imported_event, event_handle);
    if (ev_err == cudaSuccess) {
      imported_buffer->set_write_event(imported_event, true);
    } else {
      // Same-process: IPC event handles can't be opened in the originating
      // process.
      (void)cudaGetLastError();
    }
  }

  return imported_buffer;
}

#ifdef CUDA_BUFFER_ENABLE_NVSCI

// -- NvSci transport --
//
// The NvSci counterparts of export_vmm_buffer and import_vmm_buffer. The work
// is in NvSciRuntime: unlike a VMM block, an NvSci object cannot be described
// in a message for whoever reads it. It is pushed to one peer process over
// that peer's NvSciIpc channel, and the descriptor only names it.

/// \brief Fill ipc_uid and the nvsci entry of \p descriptor for \p subscriber.
/// \return false if \p buffer cannot be shared with that subscriber.
bool export_nvsci_buffer(
  const NvSciBuffer & buffer, const NvSciSubscriber & subscriber,
  CudaBufferDescriptor & descriptor)
{
  NvSciExport out;
  try {
    if (!NvSciRuntime::get().export_buffer(buffer, subscriber, &out)) {
      return false;
    }
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
      "NvSci export failed, falling back to CPU: %s", e.what());
    return false;
  }
  static_assert(
    sizeof(out.fence) == std::tuple_size<NvSciDescriptor::_fence_type>::value,
    "NvSciDescriptor.fence does not match NvSciSyncFenceIpcExportDescriptor");
  NvSciDescriptor nvsci;
  nvsci.pid = static_cast<int32_t>(out.pid);
  nvsci.buffer_id = out.buffer_id;
  nvsci.sync_channel_id = out.sync_channel_id;
  nvsci.has_fence = out.has_fence;
  std::memcpy(nvsci.fence.data(), out.fence.data(), out.fence.size());
  nvsci.connection = out.connection;
  nvsci.stream = out.stream;
  nvsci.seq = out.seq;
  descriptor.ipc_uid = out.uid;
  descriptor.nvsci.push_back(std::move(nvsci));
  return true;
}

/// \brief Map the NvSci buffer named by \p descriptor.
/// \return nullptr if \p descriptor has no nvsci entry.
/// \throws std::exception if it cannot be mapped.
std::unique_ptr<CudaBuffer> import_nvsci_buffer(const CudaBufferDescriptor & descriptor)
{
  if (descriptor.nvsci.empty()) {
    return nullptr;
  }
  const NvSciDescriptor & nvsci = descriptor.nvsci[0];

  NvSciExport in;
  in.pid = static_cast<pid_t>(nvsci.pid);
  in.buffer_id = nvsci.buffer_id;
  in.uid = descriptor.ipc_uid;
  in.sync_channel_id = nvsci.sync_channel_id;
  in.has_fence = nvsci.has_fence;
  std::memcpy(in.fence.data(), nvsci.fence.data(), in.fence.size());
  in.connection = nvsci.connection;
  in.stream = nvsci.stream;
  in.seq = nvsci.seq;
  return NvSciRuntime::get().import_buffer(in, descriptor.size * sizeof(uint8_t));
}

#endif  // CUDA_BUFFER_ENABLE_NVSCI

}  // namespace

CudaBufferBackend::CudaBufferBackend()
{
}

const rosidl_message_type_support_t *
CudaBufferBackend::get_descriptor_type_support() const
{
  return rosidl_typesupport_cpp::get_message_type_support_handle<CudaBufferDescriptor>();
}

std::shared_ptr<void>
CudaBufferBackend::create_empty_descriptor() const
{
  return std::make_shared<CudaBufferDescriptor>();
}

std::shared_ptr<host_endpoint_manager::HostEndpointManager>
CudaBufferBackend::get_endpoint_manager() const
{
  std::lock_guard<std::mutex> lock(manager_mutex_);

  if (!endpoint_manager_) {
    size_t domain_id = 0;
    endpoint_manager_ = host_endpoint_manager::HostEndpointManager::get_instance(domain_id);
    endpoint_manager_->set_storage_kinds(importable_storage_kinds());
  }

  return endpoint_manager_;
}

StorageKindMask CudaBufferBackend::shareable_kinds(
  const host_endpoint_manager::LocalityInfo & peer,
  const host_endpoint_manager::HostEndpointManager & manager)
{
  if (!peer.found) {
    return 0;
  }
  if (peer.locality == host_endpoint_manager::EndpointLocality::INTRA_PROCESS) {
    // VMM goes through its IPC path even within a process. NvSci buffers are
    // looked up in this process's pool, which needs no IPC; a process that
    // holds an NvSci buffer has NvSci available.
    StorageKindMask kinds = peer.storage_kinds & storage_kind_bit(StorageKind::Vmm);
#ifdef CUDA_BUFFER_ENABLE_NVSCI
    kinds |= storage_kind_bit(StorageKind::NvSci);
#endif
    return kinds;
  }
  if (peer.locality == host_endpoint_manager::EndpointLocality::INTER_PROCESS_SAME_HOST &&
    peer.device_id == manager.get_device_id() && peer.uid == manager.get_uid())
  {
    return peer.storage_kinds & importable_storage_kinds();
  }
  return 0;
}

void CudaBufferBackend::on_creating_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  auto manager = get_endpoint_manager();

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);

  if (endpoint_info.endpoint_type == RMW_ENDPOINT_PUBLISHER) {
    manager->register_publisher(gid, endpoint_info.topic_type);
  } else if (endpoint_info.endpoint_type == RMW_ENDPOINT_SUBSCRIPTION) {
    // Publishers bound how long they keep a block for this subscription by
    // the depth of its history. DDS does not share it in discovery, so it is
    // shared here, if the RMW gave it.
    const rmw_qos_profile_t & qos = endpoint_info.qos_profile;
    const uint32_t depth =
      qos.history == RMW_QOS_POLICY_HISTORY_KEEP_LAST &&
      qos.depth != RMW_QOS_POLICY_DEPTH_SYSTEM_DEFAULT ?
      static_cast<uint32_t>(std::min<size_t>(qos.depth, UINT32_MAX)) : 0;
    manager->register_subscription(gid, endpoint_info.topic_type, depth);
  }
}

std::pair<bool, std::vector<std::set<uint32_t>>>
CudaBufferBackend::on_discovering_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info,
  const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
  const std::unordered_map<std::string, std::string> & endpoint_supported_backends)
{
  (void)existing_endpoints;

  // The RMW does not act on the result: it still asks for a descriptor for
  // this peer on every publish. A peer that must never get one is therefore
  // cached with no importable kinds, which create_descriptor_with_endpoint
  // turns into the CPU path.
  const auto cuda = endpoint_supported_backends.find(get_backend_type());
  if (cuda == endpoint_supported_backends.end() || cuda->second != kDescriptorVersion) {
    if (cuda != endpoint_supported_backends.end()) {
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
        "Peer on topic type '%s' uses CUDA descriptor layout '%s', this process uses '%s'; "
        "using the CPU path for it",
        endpoint_info.topic_type ? endpoint_info.topic_type : "",
        cuda->second.empty() ? "(unversioned)" : cuda->second.c_str(), kDescriptorVersion);
    }
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    ipc_decision_cache_[host_endpoint_manager::GidKey(endpoint_info.endpoint_gid)] =
      ShareDecision{};
    return {false, {}};
  }

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);

  auto manager = get_endpoint_manager();
  manager->refresh_from_remote();

  auto locality_info = manager->query_endpoint_locality(gid);

  const ShareDecision decision{
    shareable_kinds(locality_info, *manager),
    static_cast<pid_t>(locality_info.remote_instance_id >> 32),
    locality_info.history_depth};

  if (locality_info.found) {
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    host_endpoint_manager::GidKey key(endpoint_info.endpoint_gid);
    ipc_decision_cache_[key] = decision;
  }

#ifdef CUDA_BUFFER_ENABLE_NVSCI
  // Open the NvSci channel now. A publisher can only allocate buffers a
  // subscriber is able to import once it has that subscriber's attributes,
  // so waiting for the first message would be too late.
  if ((decision.kinds & storage_kind_bit(StorageKind::NvSci)) &&
    locality_info.locality == host_endpoint_manager::EndpointLocality::INTER_PROCESS_SAME_HOST)
  {
    NvSciRuntime::get().connect_peer(decision.pid);
  }
#endif

  return {decision.kinds != 0, {}};
}

std::shared_ptr<void> CudaBufferBackend::create_descriptor_with_endpoint(
  const void * impl,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  host_endpoint_manager::GidKey key(endpoint_info.endpoint_gid);

  std::optional<ShareDecision> decision;
  {
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    auto it = ipc_decision_cache_.find(key);
    if (it != ipc_decision_cache_.end()) {
      decision = it->second;
    }
  }

  if (!decision) {
    // The subscriber was never classified, so classify it the way
    // on_discovering_endpoint does. If it is missing from the local endpoint
    // registry it is on another host or has not registered yet; if it is
    // present it may have no usable device, or use another device or uid.
    // None of these can import a CUDA descriptor, so use the host path.
    // A peer found in the registry is cached, as in on_discovering_endpoint.
    // A miss is not cached, so a same-host peer gets a CUDA descriptor once
    // it registers.
    rmw_gid_t peer;
    peer.implementation_identifier = "";
    std::memcpy(peer.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);
    auto peer_manager = get_endpoint_manager();
    peer_manager->refresh_from_remote();
    auto peer_locality = peer_manager->query_endpoint_locality(peer);
    decision = ShareDecision{
      shareable_kinds(peer_locality, *peer_manager),
      static_cast<pid_t>(peer_locality.remote_instance_id >> 32),
      peer_locality.history_depth};
    if (peer_locality.found) {
      std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
      ipc_decision_cache_[key] = *decision;
    }
  }
  if (decision->kinds == 0) {
    return nullptr;
  }

  auto * cuda_impl = dynamic_cast<CudaBufferImpl<uint8_t> *>(
    const_cast<rosidl::BufferImplBase<uint8_t> *>(
      static_cast<const rosidl::BufferImplBase<uint8_t> *>(impl)));
  if (!cuda_impl) {
    return nullptr;
  }

  CudaBuffer & storage = cuda_impl->get_cuda_buffer();
  if (!(decision->kinds & storage_kind_bit(storage.storage_kind()))) {
    // The subscriber cannot import this backend: CPU path for this message.
    return nullptr;
  }

  auto descriptor = std::make_shared<CudaBufferDescriptor>();

  descriptor->size = cuda_impl->size();
  descriptor->element_type_name = typeid(uint8_t).name();

  descriptor->device_id = cuda_impl->get_device_id();

  auto peer_is_same_process = [this, &endpoint_info]() {
      rmw_gid_t gid;
      gid.implementation_identifier = "";
      std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);
      auto locality = get_endpoint_manager()->query_endpoint_locality(gid);
      return locality.found &&
             locality.locality == host_endpoint_manager::EndpointLocality::INTRA_PROCESS;
    };

  // The subscriber has no CudaBuffer to ask, so the kind travels with the descriptor.
  descriptor->storage_kind = static_cast<uint8_t>(storage.storage_kind());

  // The storage backend of the buffer decides how it is shared. A backend
  // whose storage cannot be exported falls back to the CPU path. There is no
  // default case, so -Wswitch points here when a backend is added.
  bool exported = false;
  switch (storage.storage_kind()) {
    case StorageKind::Vmm:
      exported = export_vmm_buffer(
        static_cast<VmmBuffer &>(storage), peer_is_same_process, *descriptor);
      break;
    case StorageKind::NvSci:
#ifdef CUDA_BUFFER_ENABLE_NVSCI
      {
        // The RMW calls this once per descriptor it sends to the subscriber,
        // in sending order, which the subscriber's reservations rely on.
        NvSciSubscriber subscriber;
        subscriber.pid = decision->pid;
        subscriber.key.assign(
          reinterpret_cast<const char *>(endpoint_info.endpoint_gid), RMW_GID_STORAGE_SIZE);
        subscriber.history_depth = decision->history_depth;
        exported = export_nvsci_buffer(
          static_cast<const NvSciBuffer &>(storage), subscriber, *descriptor);
      }
#endif
      break;
  }
  if (!exported) {
    return nullptr;
  }

  return descriptor;
}

std::unique_ptr<void, void (*)(void *)> CudaBufferBackend::from_descriptor_with_endpoint(
  const void * descriptor_ptr,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  // Every backend's descriptor says where it came from.
  (void)endpoint_info;

  const auto * descriptor = static_cast<const CudaBufferDescriptor *>(descriptor_ptr);

  int local_device_id = 0;
  CUDA_CHECK(cudaGetDevice(&local_device_id));

  if (descriptor->element_type_name != typeid(uint8_t).name()) {
    throw CudaError(
            "CudaBufferDescriptor element type mismatch: expected " +
            std::string(typeid(uint8_t).name()) + ", got " +
            descriptor->element_type_name);
  }

  // A descriptor is only sent once its backend's entry is filled in, and the
  // import functions return nullptr when that entry is missing.
  try {
    std::unique_ptr<CudaBuffer> imported_buffer;
    switch (descriptor->storage_kind) {
      case CudaBufferDescriptor::STORAGE_KIND_VMM:
        imported_buffer = import_vmm_buffer(*descriptor, local_device_id);
        break;
      case CudaBufferDescriptor::STORAGE_KIND_NVSCI:
#ifdef CUDA_BUFFER_ENABLE_NVSCI
        imported_buffer = import_nvsci_buffer(*descriptor);
#endif
        break;
      default:
        break;
    }
    if (!imported_buffer) {
      throw CudaError("descriptor is for a storage backend this process cannot import");
    }

    auto result = std::make_unique<CudaBufferImpl<uint8_t>>(
      std::move(imported_buffer), descriptor->size);
    return {result.release(), [](void * p) {
        delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
      }};
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
      "IPC import failed, falling back to CPU: %s", e.what());
  }

  RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
    "Dropping stale descriptor: IPC failed");
  auto empty = std::make_unique<CudaBufferImpl<uint8_t>>();
  return {empty.release(), [](void * p) {
      delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
    }};
}
}  // namespace cuda_buffer_backend

PLUGINLIB_EXPORT_CLASS(
  cuda_buffer_backend::CudaBufferBackend,
  rosidl::BufferBackend)
