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

// Private to the cuda_buffer library. Free of NvSci, so that it can be tested
// anywhere.

#ifndef NVSCI_LEDGER_HPP_
#define NVSCI_LEDGER_HPP_

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuda_buffer_backend
{
namespace detail
{

/// \brief Which subscribers still owe an import of each published block.
///
/// A descriptor travels over the RMW, which may drop it without telling the
/// publisher: a full KEEP_LAST history, a lost best-effort sample, a
/// subscriber that never takes. The shared-memory refcount only covers
/// subscribers that already imported. This ledger covers the gap: every
/// descriptor sent to a subscriber is a reservation of its block, and the
/// block is not reused while one is outstanding.
///
/// Each subscriber endpoint has a stream, and the descriptors sent to it are
/// numbered in the order they are sent. A subscriber takes one publisher's
/// samples in that order, so once it reports that it has handled number k,
/// every earlier one has been imported or will never be. A reservation ends
/// at the first of:
///  - the subscriber reported a number at least as high as its own;
///  - its process can no longer import it (release_process());
///  - the subscriber keeps a KEEP_LAST history of depth N, and more than
///    N + kDepthSlack newer descriptors were sent to it since, so its history
///    has dropped this one;
///  - the hold time has passed: the fallback for a subscriber that stopped
///    taking, or whose depth is unknown;
///  - the pool reclaims it (reclaim()).
///
/// Ending a reservation early is never unsafe: an importer still takes its
/// reference and checks the per-publish uid, so the worst case is a stale
/// drop. Thread-safe.
class NvSciLedger
{
public:
  using Clock = std::chrono::steady_clock;

  /// Newer descriptors allowed in flight, beyond the history depth, before a
  /// reservation counts as dropped by the subscriber's history.
  static constexpr uint64_t kDepthSlack = 2;

  /// Where one descriptor stands in its subscriber's stream.
  struct Ticket
  {
    uint32_t stream{0};
    uint64_t seq{0};
  };

  explicit NvSciLedger(Clock::duration hold);

  /// \brief Reserve \p block_id for a descriptor sent to \p subscriber.
  /// \param subscriber identifies the subscriber endpoint; its stream is
  /// created on first use.
  /// \param pid the subscriber's process.
  /// \param history_depth the subscriber's KEEP_LAST depth, or 0 if unknown.
  Ticket record(
    uint64_t block_id, const std::string & subscriber, pid_t pid, uint32_t history_depth,
    Clock::time_point now);

  /// \brief Process \p pid handled every descriptor of \p stream up to \p seq.
  /// \details Ignored unless the stream leads to \p pid.
  void acknowledge(pid_t pid, uint32_t stream, uint64_t seq);

  /// \brief Process \p pid will import none of the descriptors sent so far.
  void release_process(pid_t pid);

  /// \brief True if no reservation of \p block_id is outstanding at \p now.
  /// \details Forgets the reservations that ended.
  bool settled(uint64_t block_id, Clock::time_point now);

  /// \brief When the oldest outstanding reservation of \p block_id was made.
  std::optional<Clock::time_point> oldest(uint64_t block_id) const;

  /// \brief End every reservation of \p block_id now.
  /// \return the processes that still owed an import of it, once each.
  std::vector<pid_t> reclaim(uint64_t block_id);

  /// \brief Forget \p block_id, which no longer exists.
  void forget(uint64_t block_id);

  /// \brief Number of blocks with outstanding reservations, for tests.
  size_t reserved_blocks() const;

private:
  struct Stream
  {
    uint32_t id{0};
    pid_t pid{0};
    /// Number of the last descriptor sent; 0 before the first.
    uint64_t last_sent{0};
    /// Highest number the subscriber reported.
    uint64_t acknowledged{0};
    uint32_t history_depth{0};
  };

  struct Reservation
  {
    std::shared_ptr<Stream> stream;
    uint64_t seq{0};
    Clock::time_point made;
    Clock::time_point deadline;
  };

  bool ended(const Reservation & reservation, Clock::time_point now) const;

  const Clock::duration hold_;
  mutable std::mutex mutex_;
  uint32_t next_stream_{1};
  std::unordered_map<std::string, std::shared_ptr<Stream>> streams_by_subscriber_;
  std::unordered_map<uint32_t, std::shared_ptr<Stream>> streams_;
  std::unordered_map<uint64_t, std::vector<Reservation>> reservations_;
};

}  // namespace detail
}  // namespace cuda_buffer_backend

#endif  // NVSCI_LEDGER_HPP_
