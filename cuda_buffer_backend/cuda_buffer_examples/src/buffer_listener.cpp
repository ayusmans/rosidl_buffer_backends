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
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "pattern.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

namespace cuda_buffer_examples
{

/// \brief Receives CUDA buffers on several topics and reports how each arrived.
///
/// Parameters:
///   topics                      topics to subscribe to
///   report_period_s             how often to log and publish the statistics
///   acceptable_buffer_backends  "cuda" (default) to receive CUDA buffers
///                               zero-copy, or "cpu" to force the CPU path
///
/// For every message it reads the payload back from the GPU, checks every byte
/// against pattern_for(sequence), and counts which path delivered it: "vmm" or
/// "nvsci" (zero-copy, the subscriber maps the publisher's memory) or "cpu"
/// (the buffer was serialized because this subscriber cannot import it).
/// The statistics are also published as one line of JSON per topic on
/// "buffer_stats".
class BufferListener : public rclcpp::Node
{
public:
  explicit BufferListener(const rclcpp::NodeOptions & options)
  : Node("buffer_listener", options)
  {
    const auto topics = declare_parameter<std::vector<std::string>>(
      "topics", std::vector<std::string>{"buffer"});
    const double report_period_s = declare_parameter<double>("report_period_s", 2.0);
    const std::string accept = declare_parameter<std::string>("acceptable_buffer_backends", "cuda");
    if (topics.empty() || report_period_s <= 0.0) {
      throw std::invalid_argument("topics must not be empty and report_period_s must be positive");
    }

    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("cannot create a CUDA stream");
    }

    // A subscription receives buffers zero-copy only from the backends it
    // accepts; everything else arrives over the CPU path.
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = accept;

    stats_.resize(topics.size());
    for (size_t i = 0; i < topics.size(); ++i) {
      subscriptions_.push_back(
        create_subscription<sensor_msgs::msg::Image>(
          topics[i], 10,
          [this, i](const sensor_msgs::msg::Image::SharedPtr msg) {on_message(i, *msg);},
          subscription_options));
      stats_[i].topic = subscriptions_.back()->get_topic_name();
    }
    stats_publisher_ = create_publisher<std_msgs::msg::String>("buffer_stats", 10);
    report_timer_ = create_wall_timer(
      std::chrono::duration<double>(report_period_s), [this]() {report();});
  }

  ~BufferListener() override
  {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

private:
  struct TopicStats
  {
    std::string topic;
    uint64_t received{0};
    uint64_t vmm{0};
    uint64_t nvsci{0};
    uint64_t cpu{0};
    uint64_t corrupt{0};
    double latency_ms_sum{0.0};
  };

  void on_message(size_t index, const sensor_msgs::msg::Image & msg)
  {
    TopicStats & stats = stats_[index];
    ++stats.received;
    stats.latency_ms_sum += (now() - rclcpp::Time(msg.header.stamp)).seconds() * 1e3;

    // Which path delivered this message.
    const auto kind = cuda_buffer_backend::storage_kind_of(msg.data);
    if (!kind) {
      ++stats.cpu;
    } else if (*kind == cuda_buffer_backend::StorageKind::NvSci) {
      ++stats.nvsci;
    } else {
      ++stats.vmm;
    }

    if (!payload_is_valid(msg)) {
      ++stats.corrupt;
      RCLCPP_ERROR(
        get_logger(), "Corrupt payload in message '%s' on '%s'",
        msg.header.frame_id.c_str(), stats.topic.c_str());
    }
  }

  bool payload_is_valid(const sensor_msgs::msg::Image & msg)
  {
    uint64_t sequence = 0;
    try {
      sequence = std::stoull(msg.header.frame_id);
    } catch (const std::exception &) {
      return false;
    }
    const size_t size = msg.data.size();
    if (size == 0 || size != static_cast<size_t>(msg.step) * msg.height) {
      return false;
    }

    // from_input_buffer orders this stream after the publisher's write. A
    // buffer that arrived over the CPU path is copied to the GPU first, so
    // the same code reads either kind.
    host_.resize(size);
    try {
      cuda_buffer_backend::ReadHandle handle =
        cuda_buffer_backend::from_input_buffer(msg.data, stream_);
      if (cudaMemcpyAsync(host_.data(), handle.get_ptr(), size, cudaMemcpyDeviceToHost, stream_) !=
        cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
      {
        return false;
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Cannot read the payload: %s", e.what());
      return false;
    }

    const uint8_t expected = pattern_for(sequence);
    for (uint8_t byte : host_) {
      if (byte != expected) {
        return false;
      }
    }
    return true;
  }

  void report()
  {
    for (const TopicStats & stats : stats_) {
      const double latency_ms =
        stats.received ? stats.latency_ms_sum / static_cast<double>(stats.received) : 0.0;
      RCLCPP_INFO(
        get_logger(),
        "%s: %" PRIu64 " received (zero-copy vmm=%" PRIu64 " nvsci=%" PRIu64 ", cpu=%" PRIu64
        "), %" PRIu64 " corrupt, mean latency %.2f ms",
        stats.topic.c_str(), stats.received, stats.vmm, stats.nvsci, stats.cpu, stats.corrupt,
        latency_ms);

      std::ostringstream json;
      json << "{\"topic\": \"" << stats.topic << "\", \"received\": " << stats.received <<
        ", \"vmm\": " << stats.vmm << ", \"nvsci\": " << stats.nvsci <<
        ", \"cpu\": " << stats.cpu << ", \"corrupt\": " << stats.corrupt << "}";
      std_msgs::msg::String out;
      out.data = json.str();
      stats_publisher_->publish(out);
    }
  }

  cudaStream_t stream_{nullptr};
  std::vector<uint8_t> host_;
  std::vector<TopicStats> stats_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_publisher_;
  rclcpp::TimerBase::SharedPtr report_timer_;
};

}  // namespace cuda_buffer_examples

RCLCPP_COMPONENTS_REGISTER_NODE(cuda_buffer_examples::BufferListener)
