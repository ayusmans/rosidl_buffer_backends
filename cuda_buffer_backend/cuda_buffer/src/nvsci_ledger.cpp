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

#include "nvsci_ledger.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cuda_buffer_backend
{
namespace detail
{

NvSciLedger::NvSciLedger(Clock::duration hold)
: hold_(hold)
{
}

NvSciLedger::Ticket NvSciLedger::record(
  uint64_t block_id, const std::string & subscriber, pid_t pid, uint32_t history_depth,
  Clock::time_point now)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::shared_ptr<Stream> & stream = streams_by_subscriber_[subscriber];
  if (!stream || stream->pid != pid) {
    // A new subscriber, or (with a reused GID) a new process behind it.
    stream = std::make_shared<Stream>();
    stream->id = next_stream_++;
    if (next_stream_ == 0) {
      next_stream_ = 1;  // 0 means "no stream" in a descriptor.
    }
    stream->pid = pid;
    streams_[stream->id] = stream;
  }
  stream->history_depth = history_depth;
  const uint64_t seq = ++stream->last_sent;
  reservations_[block_id].push_back(Reservation{stream, seq, now, now + hold_});
  return Ticket{stream->id, seq};
}

void NvSciLedger::acknowledge(pid_t pid, uint32_t stream_id, uint64_t seq)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = streams_.find(stream_id);
  if (it == streams_.end() || it->second->pid != pid) {
    return;
  }
  Stream & stream = *it->second;
  // A number never sent cannot release anything sent later.
  stream.acknowledged = std::max(stream.acknowledged, std::min(seq, stream.last_sent));
}

void NvSciLedger::release_process(pid_t pid)
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & entry : streams_) {
    if (entry.second->pid == pid) {
      entry.second->acknowledged = entry.second->last_sent;
    }
  }
}

bool NvSciLedger::ended(const Reservation & reservation, Clock::time_point now) const
{
  const Stream & stream = *reservation.stream;
  if (reservation.seq <= stream.acknowledged || now >= reservation.deadline) {
    return true;
  }
  // The subscriber's history keeps its newest history_depth samples. Once
  // that many newer ones (plus some still in flight) were sent, this one has
  // been dropped from it, unless it was taken, and so imported, already.
  return stream.history_depth != 0 &&
         stream.last_sent - reservation.seq >= stream.history_depth + kDepthSlack;
}

bool NvSciLedger::settled(uint64_t block_id, Clock::time_point now)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = reservations_.find(block_id);
  if (it == reservations_.end()) {
    return true;
  }
  auto & list = it->second;
  list.erase(
    std::remove_if(
      list.begin(), list.end(),
      [this, now](const Reservation & r) {return ended(r, now);}),
    list.end());
  if (list.empty()) {
    reservations_.erase(it);
    return true;
  }
  return false;
}

std::optional<NvSciLedger::Clock::time_point> NvSciLedger::oldest(uint64_t block_id) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = reservations_.find(block_id);
  if (it == reservations_.end() || it->second.empty()) {
    return std::nullopt;
  }
  Clock::time_point made = it->second.front().made;
  for (const Reservation & r : it->second) {
    made = std::min(made, r.made);
  }
  return made;
}

std::vector<pid_t> NvSciLedger::reclaim(uint64_t block_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<pid_t> pids;
  auto it = reservations_.find(block_id);
  if (it == reservations_.end()) {
    return pids;
  }
  for (const Reservation & r : it->second) {
    if (std::find(pids.begin(), pids.end(), r.stream->pid) == pids.end()) {
      pids.push_back(r.stream->pid);
    }
  }
  reservations_.erase(it);
  return pids;
}

void NvSciLedger::forget(uint64_t block_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  reservations_.erase(block_id);
}

size_t NvSciLedger::reserved_blocks() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return reservations_.size();
}

}  // namespace detail
}  // namespace cuda_buffer_backend
