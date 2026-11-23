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

#include <cuda_runtime.h>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "pattern.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace cuda_buffer_examples
{

/// \brief Publishes CUDA buffers allocated from one storage backend.
///
/// Parameters:
///   storage     "vmm", "nvsci", or "" for the process default
///               (CUDA_BUFFER_STORAGE, else "vmm")
///   topic       topic to publish on
///   size_bytes  payload size of each message
///   rate_hz     publish rate
///
/// Each message is a mono8 sensor_msgs/Image whose data is filled on the GPU
/// with pattern_for(sequence); header.frame_id carries the sequence number.
/// Several talkers in one process may use different backends.
class BufferTalker : public rclcpp::Node
{
public:
  explicit BufferTalker(const rclcpp::NodeOptions & options)
  : Node("buffer_talker", options)
  {
    const std::string storage = declare_parameter<std::string>("storage", "");
    const std::string topic = declare_parameter<std::string>("topic", "buffer");
    size_bytes_ = static_cast<size_t>(declare_parameter<int64_t>("size_bytes", 1 << 20));
    const double rate_hz = declare_parameter<double>("rate_hz", 10.0);
    if (size_bytes_ == 0 || rate_hz <= 0.0) {
      throw std::invalid_argument("size_bytes and rate_hz must be positive");
    }

    kind_ = cuda_buffer_backend::default_storage_kind();
    if (!storage.empty()) {
      const auto requested = cuda_buffer_backend::parse_storage_kind(storage);
      if (!requested) {
        throw std::invalid_argument(
                "storage must be 'vmm', 'nvsci' or empty, not '" + storage + "'");
      }
      if (cuda_buffer_backend::is_storage_available(*requested)) {
        kind_ = *requested;
      } else {
        RCLCPP_WARN(
          get_logger(), "Storage '%s' is not available in this process; using '%s'",
          storage.c_str(), cuda_buffer_backend::to_string(kind_));
      }
    }

    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("cannot create a CUDA stream");
    }
    publisher_ = create_publisher<sensor_msgs::msg::Image>(topic, 10);
    report_every_ = std::max<uint64_t>(1, static_cast<uint64_t>(rate_hz * 5.0));
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz), [this]() {publish();});

    RCLCPP_INFO(
      get_logger(), "Publishing %zu-byte buffers on '%s' at %.1f Hz using '%s' storage",
      size_bytes_, publisher_->get_topic_name(), rate_hz, cuda_buffer_backend::to_string(kind_));
  }

  ~BufferTalker() override
  {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

private:
  void publish()
  {
    sensor_msgs::msg::Image msg;
    msg.header.stamp = now();
    msg.header.frame_id = std::to_string(sequence_);
    msg.height = 1;
    msg.width = static_cast<uint32_t>(size_bytes_);
    msg.step = static_cast<uint32_t>(size_bytes_);
    msg.encoding = "mono8";

    // Allocate from this talker's backend and fill on the GPU. The write is
    // ordered before every reader by the handle; nothing waits on the CPU.
    msg.data = cuda_buffer_backend::allocate_buffer(size_bytes_, kind_);
    {
      cuda_buffer_backend::WriteHandle handle =
        cuda_buffer_backend::from_output_buffer(msg.data, stream_);
      if (cudaMemsetAsync(handle.get_ptr(), pattern_for(sequence_), size_bytes_, stream_) !=
        cudaSuccess)
      {
        RCLCPP_ERROR(get_logger(), "cudaMemsetAsync failed");
        return;
      }
    }

    publisher_->publish(msg);
    ++sequence_;
    if (sequence_ % report_every_ == 0) {
      RCLCPP_INFO(
        get_logger(), "Published %" PRIu64 " buffers on '%s' (%s)", sequence_,
        publisher_->get_topic_name(), cuda_buffer_backend::to_string(kind_));
    }
  }

  cuda_buffer_backend::StorageKind kind_{cuda_buffer_backend::kDefaultStorageKind};
  size_t size_bytes_{0};
  uint64_t sequence_{0};
  uint64_t report_every_{1};
  cudaStream_t stream_{nullptr};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace cuda_buffer_examples

RCLCPP_COMPONENTS_REGISTER_NODE(cuda_buffer_examples::BufferTalker)
