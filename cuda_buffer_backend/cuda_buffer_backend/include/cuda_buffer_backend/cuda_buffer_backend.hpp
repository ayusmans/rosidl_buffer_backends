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

#ifndef CUDA_BUFFER_BACKEND__CUDA_BUFFER_BACKEND_HPP_
#define CUDA_BUFFER_BACKEND__CUDA_BUFFER_BACKEND_HPP_

#include <sys/types.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/host_endpoint_manager.hpp"
#include "rosidl_buffer_backend/buffer_backend.hpp"

namespace cuda_buffer_backend
{

/// \brief CUDA buffer backend plugin for zero-copy GPU memory sharing.
/// \details Every buffer carries its storage backend (CUDA VMM or NvSci), and
/// each one is shared the way its backend allows. A subscriber that cannot
/// import a buffer's backend gets that message over the CPU path.
class CudaBufferBackend : public rosidl::BufferBackend
{
public:
  /// \brief Layout version of CudaBufferDescriptor, advertised as this
  /// backend's metadata.
  /// \details The RMW sends a descriptor as the backend name and the message,
  /// with no type hash, so two builds that disagree on the layout would
  /// misread each other's descriptors. A peer that advertises "cuda" with any
  /// other metadata, including the empty metadata of builds from before this
  /// version, gets every message over the CPU path. Change it whenever
  /// CudaBufferDescriptor or a message it contains changes. It travels in the
  /// endpoint's user data, so it must not contain ',', ';' or '='.
  static constexpr const char * kDescriptorVersion = "descriptor-v2";

  CudaBufferBackend();
  ~CudaBufferBackend() override = default;

  std::string get_backend_type() const override
  {
    return "cuda";
  }

  const rosidl_message_type_support_t * get_descriptor_type_support() const override;

  std::shared_ptr<void> create_empty_descriptor() const override;

  std::shared_ptr<void> create_descriptor_with_endpoint(
    const void * impl,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::unique_ptr<void, void (*)(void *)> from_descriptor_with_endpoint(
    const void * descriptor,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  void on_creating_endpoint(
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::pair<bool, std::vector<std::set<uint32_t>>> on_discovering_endpoint(
    const rmw_topic_endpoint_info_t & endpoint_info,
    const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
    const std::unordered_map<std::string, std::string> & endpoint_supported_backends) override;

  std::string get_backend_metadata() const override
  {
    return kDescriptorVersion;
  }

private:
  std::shared_ptr<host_endpoint_manager::HostEndpointManager> get_endpoint_manager() const;

  /// What can be sent to one subscriber, decided once per endpoint.
  struct ShareDecision
  {
    /// Storage backends whose buffers the subscriber can import from here.
    StorageKindMask kinds{0};
    /// Its process.
    pid_t pid{0};
    /// Depth of its KEEP_LAST history, or 0 if unknown.
    uint32_t history_depth{0};
  };

  /// The storage backends a peer can import from this process: those it is
  /// able to import, if it is in this process, or on this host with the same
  /// device and uid.
  static StorageKindMask shareable_kinds(
    const host_endpoint_manager::LocalityInfo & peer,
    const host_endpoint_manager::HostEndpointManager & manager);

  mutable std::shared_ptr<host_endpoint_manager::HostEndpointManager> endpoint_manager_;
  mutable std::mutex manager_mutex_;

  mutable std::unordered_map<
    host_endpoint_manager::GidKey, ShareDecision,
    host_endpoint_manager::GidKeyHash> ipc_decision_cache_;
  mutable std::mutex ipc_cache_mutex_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER_BACKEND__CUDA_BUFFER_BACKEND_HPP_
