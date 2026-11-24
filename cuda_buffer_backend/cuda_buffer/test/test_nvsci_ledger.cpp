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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "nvsci_ledger.hpp"

using cuda_buffer_backend::detail::NvSciLedger;
using Clock = NvSciLedger::Clock;
using std::chrono::milliseconds;

namespace
{

constexpr pid_t kSubPid = 1000;
constexpr pid_t kOtherPid = 2000;

class LedgerTest : public ::testing::Test
{
protected:
  NvSciLedger ledger_{milliseconds(1000)};
  const Clock::time_point t0_ = Clock::now();
};

}  // namespace

TEST_F(LedgerTest, UnpublishedBlockIsSettled)
{
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_FALSE(ledger_.oldest(1).has_value());
}

TEST_F(LedgerTest, ReservedUntilTheSubscriberHandlesIt)
{
  const auto ticket = ledger_.record(1, "sub", kSubPid, 0, t0_);
  EXPECT_NE(0u, ticket.stream);
  EXPECT_EQ(1u, ticket.seq);
  EXPECT_FALSE(ledger_.settled(1, t0_));
  ledger_.acknowledge(kSubPid, ticket.stream, ticket.seq);
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_EQ(0u, ledger_.reserved_blocks());
}

TEST_F(LedgerTest, LaterAcknowledgementReleasesEarlierDescriptors)
{
  // Descriptors 1 and 2 were dropped by the RMW; the subscriber took 3.
  const auto a = ledger_.record(1, "sub", kSubPid, 0, t0_);
  const auto b = ledger_.record(2, "sub", kSubPid, 0, t0_);
  const auto c = ledger_.record(3, "sub", kSubPid, 0, t0_);
  EXPECT_EQ(a.stream, b.stream);
  EXPECT_EQ(b.stream, c.stream);
  ledger_.acknowledge(kSubPid, c.stream, c.seq);
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_TRUE(ledger_.settled(2, t0_));
  EXPECT_TRUE(ledger_.settled(3, t0_));
}

TEST_F(LedgerTest, EarlierAcknowledgementKeepsLaterDescriptors)
{
  const auto a = ledger_.record(1, "sub", kSubPid, 0, t0_);
  ledger_.record(2, "sub", kSubPid, 0, t0_);
  ledger_.acknowledge(kSubPid, a.stream, a.seq);
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_FALSE(ledger_.settled(2, t0_));
  // An acknowledgement that goes backwards changes nothing.
  ledger_.acknowledge(kSubPid, a.stream, 0);
  EXPECT_TRUE(ledger_.settled(1, t0_));
}

TEST_F(LedgerTest, EverySubscriberMustHandleIt)
{
  const auto a = ledger_.record(1, "sub_a", kSubPid, 0, t0_);
  const auto b = ledger_.record(1, "sub_b", kOtherPid, 0, t0_);
  EXPECT_NE(a.stream, b.stream);
  EXPECT_EQ(1u, b.seq);  // Each subscriber has its own numbering.
  ledger_.acknowledge(kSubPid, a.stream, a.seq);
  EXPECT_FALSE(ledger_.settled(1, t0_));
  ledger_.acknowledge(kOtherPid, b.stream, b.seq);
  EXPECT_TRUE(ledger_.settled(1, t0_));
}

TEST_F(LedgerTest, SameBlockPublishedTwiceNeedsBoth)
{
  ledger_.record(1, "sub", kSubPid, 0, t0_);
  const auto second = ledger_.record(1, "sub", kSubPid, 0, t0_);
  ledger_.acknowledge(kSubPid, second.stream, second.seq - 1);
  EXPECT_FALSE(ledger_.settled(1, t0_));
  ledger_.acknowledge(kSubPid, second.stream, second.seq);
  EXPECT_TRUE(ledger_.settled(1, t0_));
}

TEST_F(LedgerTest, AcknowledgementFromAnotherProcessIsIgnored)
{
  const auto ticket = ledger_.record(1, "sub", kSubPid, 0, t0_);
  ledger_.acknowledge(kOtherPid, ticket.stream, ticket.seq);
  EXPECT_FALSE(ledger_.settled(1, t0_));
}

TEST_F(LedgerTest, AcknowledgementOfUnsentNumberDoesNotReleaseLaterOnes)
{
  const auto ticket = ledger_.record(1, "sub", kSubPid, 0, t0_);
  ledger_.acknowledge(kSubPid, ticket.stream, 100);
  EXPECT_TRUE(ledger_.settled(1, t0_));
  ledger_.record(2, "sub", kSubPid, 0, t0_);
  EXPECT_FALSE(ledger_.settled(2, t0_));
}

TEST_F(LedgerTest, ReservationExpiresAtTheHoldTime)
{
  ledger_.record(1, "sub", kSubPid, 0, t0_);
  EXPECT_FALSE(ledger_.settled(1, t0_ + milliseconds(999)));
  EXPECT_TRUE(ledger_.settled(1, t0_ + milliseconds(1000)));
}

TEST_F(LedgerTest, ReleasedProcessOwesNothingSentBefore)
{
  ledger_.record(1, "sub", kSubPid, 0, t0_);
  ledger_.record(2, "other", kOtherPid, 0, t0_);
  ledger_.release_process(kSubPid);
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_FALSE(ledger_.settled(2, t0_));
  // Descriptors sent after a reconnect are reserved again.
  ledger_.record(3, "sub", kSubPid, 0, t0_);
  EXPECT_FALSE(ledger_.settled(3, t0_));
}

TEST_F(LedgerTest, HistoryDepthBoundsReservationsOfAStalledSubscriber)
{
  constexpr uint32_t kDepth = 3;
  constexpr uint64_t kLimit = kDepth + NvSciLedger::kDepthSlack;
  ledger_.record(1, "sub", kSubPid, kDepth, t0_);
  for (uint64_t i = 1; i < kLimit; ++i) {
    ledger_.record(100 + i, "sub", kSubPid, kDepth, t0_);
  }
  // kLimit - 1 newer descriptors: block 1 may still be in the history.
  EXPECT_FALSE(ledger_.settled(1, t0_));
  ledger_.record(200, "sub", kSubPid, kDepth, t0_);
  // kLimit newer ones: the history dropped it.
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_FALSE(ledger_.settled(200, t0_));
}

TEST_F(LedgerTest, UnknownDepthKeepsReservationsUntilTheHoldTime)
{
  ledger_.record(1, "sub", kSubPid, 0, t0_);
  for (uint64_t i = 0; i < 100; ++i) {
    ledger_.record(100 + i, "sub", kSubPid, 0, t0_);
  }
  EXPECT_FALSE(ledger_.settled(1, t0_));
}

TEST_F(LedgerTest, DepthOfOneSubscriberDoesNotReleaseAnother)
{
  ledger_.record(1, "deep", kSubPid, 0, t0_);
  ledger_.record(1, "shallow", kOtherPid, 1, t0_);
  for (uint64_t i = 0; i < 10; ++i) {
    ledger_.record(100 + i, "shallow", kOtherPid, 1, t0_);
  }
  EXPECT_FALSE(ledger_.settled(1, t0_));
}

TEST_F(LedgerTest, ReclaimEndsReservationsAndNamesTheirProcesses)
{
  ledger_.record(1, "a", kSubPid, 0, t0_ + milliseconds(5));
  ledger_.record(1, "b", kOtherPid, 0, t0_);
  ledger_.record(1, "a", kSubPid, 0, t0_ + milliseconds(7));
  ASSERT_TRUE(ledger_.oldest(1).has_value());
  EXPECT_EQ(t0_, *ledger_.oldest(1));
  auto pids = ledger_.reclaim(1);
  std::sort(pids.begin(), pids.end());
  EXPECT_EQ((std::vector<pid_t>{kSubPid, kOtherPid}), pids);
  EXPECT_TRUE(ledger_.settled(1, t0_));
  EXPECT_FALSE(ledger_.oldest(1).has_value());
}

TEST_F(LedgerTest, NewProcessBehindTheSameSubscriberKeyGetsANewStream)
{
  const auto old_ticket = ledger_.record(1, "sub", kSubPid, 0, t0_);
  const auto new_ticket = ledger_.record(2, "sub", kOtherPid, 0, t0_);
  EXPECT_NE(old_ticket.stream, new_ticket.stream);
  EXPECT_EQ(1u, new_ticket.seq);
  // The old process cannot acknowledge the new stream.
  ledger_.acknowledge(kSubPid, new_ticket.stream, new_ticket.seq);
  EXPECT_FALSE(ledger_.settled(2, t0_));
}

TEST_F(LedgerTest, ForgottenBlockIsSettled)
{
  ledger_.record(1, "sub", kSubPid, 0, t0_);
  ledger_.forget(1);
  EXPECT_TRUE(ledger_.settled(1, t0_));
}

// Publishers and subscribers on several threads. A subscriber takes
// descriptors in order, skips some (RMW drops), and acknowledges each one it
// takes. A block must stay reserved while one of its descriptors could still
// be taken, and every block must be settled once every subscriber has caught
// up.
TEST(LedgerStress, ReservedWhileImportableAndSettledAfterwards)
{
  constexpr int kSubscribers = 4;
  constexpr uint64_t kBlocks = 16;
  constexpr int kPublishes = 4000;
  NvSciLedger ledger(std::chrono::hours(1));

  struct Sent
  {
    uint64_t block;
    uint32_t stream;
    uint64_t seq;
  };
  struct Queue
  {
    std::mutex mutex;
    std::vector<Sent> items;
    size_t next{0};
  };
  std::vector<Queue> queues(kSubscribers);
  std::atomic<bool> done{false};
  std::atomic<int> violations{0};

  std::thread publisher([&] {
      std::mt19937 rng(1);
      for (int i = 0; i < kPublishes; ++i) {
        const uint64_t block = rng() % kBlocks;
        for (int s = 0; s < kSubscribers; ++s) {
          const auto t = ledger.record(
            block, "sub" + std::to_string(s), 100 + s, 0, Clock::now());
          std::lock_guard<std::mutex> lock(queues[s].mutex);
          queues[s].items.push_back({block, t.stream, t.seq});
        }
      }
      done = true;
    });

  std::vector<std::thread> subscribers;
  for (int s = 0; s < kSubscribers; ++s) {
    subscribers.emplace_back(
      [&, s] {
        std::mt19937 rng(10 + s);
        for (;; ) {
          Sent item{};
          bool have = false;
          {
            std::lock_guard<std::mutex> lock(queues[s].mutex);
            if (queues[s].next < queues[s].items.size()) {
              item = queues[s].items[queues[s].next++];
              have = true;
            }
          }
          if (!have) {
            if (done) {
              std::lock_guard<std::mutex> lock(queues[s].mutex);
              if (queues[s].next == queues[s].items.size()) {
                return;
              }
            }
            std::this_thread::yield();
            continue;
          }
          if (rng() % 4 == 0) {
            continue;  // Dropped by the RMW: never acknowledged by itself.
          }
          // About to import: the block must still be reserved for it.
          if (ledger.settled(item.block, Clock::now())) {
            ++violations;
          }
          ledger.acknowledge(100 + s, item.stream, item.seq);
        }
      });
  }
  publisher.join();
  for (auto & t : subscribers) {
    t.join();
  }
  EXPECT_EQ(0, violations.load());

  // Catch up the streams whose last descriptors were dropped.
  for (int s = 0; s < kSubscribers; ++s) {
    const Sent & last = queues[s].items.back();
    ledger.acknowledge(100 + s, last.stream, last.seq);
  }
  for (uint64_t b = 0; b < kBlocks; ++b) {
    EXPECT_TRUE(ledger.settled(b, Clock::now())) << "block " << b;
  }
}
