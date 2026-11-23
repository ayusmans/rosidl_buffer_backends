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
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nvsci_endpoint_pool.hpp"

using cuda_buffer_backend::detail::NvSciEndpointPool;
using Pool = NvSciEndpointPool;

namespace
{

/// One pool configuration of its own per test: inventory, NvSciIpc
/// configuration and claim region, all removed afterwards.
class PoolFiles
{
public:
  explicit PoolFiles(const std::string & tag)
  : base_("/tmp/cbpool_" + tag + "_" + std::to_string(getpid())),
    region_("/cbpool_" + tag + "_" + std::to_string(getpid()))
  {
    shm_unlink(region_.c_str());
  }

  ~PoolFiles()
  {
    shm_unlink(region_.c_str());
    std::remove(inventory().c_str());
    std::remove((base_ + "_other.json").c_str());
    std::remove(nvsciipc_cfg().c_str());
  }

  std::string inventory() const {return base_ + ".json";}
  std::string nvsciipc_cfg() const {return base_ + ".cfg";}
  const std::string & region() const {return region_;}

  /// An inventory of \p count pairs "<tag>_<i>_a"/"<tag>_<i>_b", declared in
  /// the NvSciIpc configuration too.
  void write_pairs(size_t count, const std::string & path = "") const
  {
    std::string json = "{\"version\": 1, \"pool\": [";
    std::string cfg;
    for (size_t i = 0; i < count; ++i) {
      const std::string a = "ep_" + std::to_string(i) + "_a";
      const std::string b = "ep_" + std::to_string(i) + "_b";
      json += (i ? ", " : "") + std::string("{\"endpointA\": \"") + a +
        "\", \"endpointB\": \"" + b + "\"}";
      cfg += "INTER_PROCESS " + a + " " + b + " 16 4096\n";
    }
    json += "]}";
    write(path.empty() ? inventory() : path, json);
    write(nvsciipc_cfg(), cfg);
  }

  static void write(const std::string & path, const std::string & text)
  {
    FILE * f = std::fopen(path.c_str(), "w");
    ASSERT_NE(nullptr, f);
    std::fputs(text.c_str(), f);
    std::fclose(f);
  }

  Pool::Config config(const std::string & inventory_path = "") const
  {
    Pool::Config c;
    c.inventory_path = inventory_path.empty() ? inventory() : inventory_path;
    c.region_name = region_;
    c.nvsciipc_config_path = nvsciipc_cfg();
    c.attach_wait = std::chrono::milliseconds(50);
    return c;
  }

private:
  std::string base_;
  std::string region_;
};

/// Run \p body in a child process; it reports one string back through a
/// pipe. The child stays alive until finish() or kill_now(). Children inherit
/// each other's pipe ends, so they are told to exit with a byte, not by the
/// pipe closing.
struct Child
{
  pid_t pid{-1};
  int result_fd{-1};
  int release_fd{-1};

  template<typename Body>
  static Child start(Body body)
  {
    int result[2];
    int release[2];
    if (pipe(result) != 0 || pipe(release) != 0) {
      return {};
    }
    const pid_t pid = fork();
    if (pid == 0) {
      close(result[0]);
      close(release[1]);
      const std::string out = body(release[0]) + "\n";
      if (::write(result[1], out.data(), out.size()) < 0) {
        _exit(1);
      }
      char c;
      if (read(release[0], &c, 1) < 0) {
        _exit(1);
      }
      _exit(0);
    }
    close(result[1]);
    close(release[0]);
    return {pid, result[0], release[1]};
  }

  std::string result() const
  {
    std::string out;
    char c;
    while (read(result_fd, &c, 1) == 1 && c != '\n') {
      out.push_back(c);
    }
    return out;
  }

  /// Let the child run on: its body may wait for this.
  void signal() const
  {
    const char c = 'x';
    if (::write(release_fd, &c, 1) < 0) {
      // The child is gone already.
    }
  }

  void finish()
  {
    signal();
    close(release_fd);
    close(result_fd);
    int status = 0;
    waitpid(pid, &status, 0);
  }

  void kill_now()
  {
    ::kill(pid, SIGKILL);
    finish();
  }
};

/// Ask for \p peer until a pair comes up, for at most two seconds.
Pool::Result claim_patiently(Pool & pool, pid_t peer, Pool::Claim * claim)
{
  Pool::Result result = Pool::Result::kUnavailable;
  for (int i = 0; i < 200; ++i) {
    result = pool.claim(peer, claim);
    if (result != Pool::Result::kWaitForPeer) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return result;
}

}  // namespace

// -- Inventory --

TEST(NvSciEndpointPoolInventory, ParsesTheHalosIpcFormat)
{
  const auto pairs = Pool::parse_inventory(
    R"({"version": 1, "pool": [
          {"endpointA": "nvscisync_a_0", "endpointB": "nvscisync_a_1"},
          {"endpointA": "itc_0", "endpointB": "itc_1", "nframes": 16, "frameSize": 24576}
        ]})");
  ASSERT_EQ(2u, pairs.size());
  EXPECT_EQ("nvscisync_a_0", pairs[0].endpoint_a);
  EXPECT_EQ("nvscisync_a_1", pairs[0].endpoint_b);
  EXPECT_EQ("itc_0", pairs[1].endpoint_a);
  EXPECT_EQ("itc_1", pairs[1].endpoint_b);
}

TEST(NvSciEndpointPoolInventory, RejectsInvalidInventories)
{
  const char * invalid[] = {
    "not json",
    "[]",
    R"({"pool": []})",
    R"({"version": 2, "pool": []})",
    R"({"version": 1})",
    R"({"version": 1, "pool": {}})",
    R"({"version": 1, "pool": [], "extra": 1})",
    R"({"version": 1, "pool": [{"endpointA": "a"}]})",
    R"({"version": 1, "pool": [{"endpointA": "", "endpointB": "b"}]})",
    R"({"version": 1, "pool": [{"endpointA": "a", "endpointB": 3}]})",
    R"({"version": 1, "pool": [{"endpointA": "a", "endpointB": "b", "reserved": true}]})",
    R"({"version": 1, "pool": [{"endpointA": "a", "endpointB": "b", "nframes": 0}]})",
    // An endpoint listed twice, as the same or the other end.
    R"({"version": 1, "pool": [{"endpointA": "a", "endpointB": "a"}]})",
    R"({"version": 1, "pool": [{"endpointA": "a", "endpointB": "b"},
                               {"endpointA": "c", "endpointB": "a"}]})",
  };
  for (const char * text : invalid) {
    EXPECT_THROW(Pool::parse_inventory(text), std::runtime_error) << text;
  }
}

TEST(NvSciEndpointPoolInventory, ParsesNvSciIpcInterProcessPairs)
{
  const auto pairs = Pool::parse_nvsciipc_config(
    "# comment\n"
    "INTER_PROCESS   ipc_test_0      ipc_test_1      64  1536   30000\n"
    "INTER_THREAD    itc_0           itc_1           16  24576\n"
    "INTER_PROCESS   cuda_buffer_2   cuda_buffer_3   16  24576\n"
    "INTER_PROCESS   broken\n");
  ASSERT_EQ(2u, pairs.size());
  EXPECT_EQ("ipc_test_0", pairs[0].first);
  EXPECT_EQ("cuda_buffer_3", pairs[1].second);
}

TEST(NvSciEndpointPool, WithoutAnInventoryThereIsNoPool)
{
  PoolFiles files("none");
  Pool pool(files.config());
  EXPECT_EQ(Pool::State::kNoInventory, pool.state());
  Pool::Claim claim;
  EXPECT_EQ(Pool::Result::kUnavailable, pool.claim(getppid(), &claim));
}

TEST(NvSciEndpointPool, InvalidInventoryIsAnError)
{
  PoolFiles files("invalid");
  PoolFiles::write(files.inventory(), R"({"version": 1, "pool": [{"endpointA": "a"}]})");
  Pool pool(files.config());
  EXPECT_EQ(Pool::State::kError, pool.state());
  EXPECT_NE(std::string::npos, pool.error().find(files.inventory())) << pool.error();
}

TEST(NvSciEndpointPool, WarnsAboutPairsNvSciIpcDoesNotDeclare)
{
  PoolFiles files("undeclared");
  files.write_pairs(1);
  PoolFiles::write(
    files.inventory(),
    R"({"version": 1, "pool": [{"endpointA": "ep_0_b", "endpointB": "ep_0_a"},
                               {"endpointA": "missing_a", "endpointB": "missing_b"}]})");
  Pool pool(files.config());
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();
  // Either order is the same pair; the missing one is reported.
  ASSERT_EQ(1u, pool.warnings().size());
  EXPECT_NE(std::string::npos, pool.warnings()[0].find("missing_a/missing_b"));
}

// -- Claim region --

TEST(NvSciEndpointPool, CreatesRegionForAllUsersWithoutChangingUmask)
{
  PoolFiles files("mode");
  files.write_pairs(1);
  const mode_t before = umask(022);
  Pool pool(files.config());
  const mode_t during = umask(before);
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();
  EXPECT_EQ(022u, during);

  const int fd = shm_open(files.region().c_str(), O_RDONLY, 0);
  ASSERT_GE(fd, 0) << std::strerror(errno);
  struct stat info {};
  ASSERT_EQ(0, fstat(fd, &info));
  close(fd);
  EXPECT_EQ(0666u, info.st_mode & 0777u);
}

TEST(NvSciEndpointPool, RefusesARegionOfAnotherInventory)
{
  PoolFiles files("mismatch");
  files.write_pairs(2);
  const std::string other = "/tmp/cbpool_mismatch_" + std::to_string(getpid()) + "_other.json";
  files.write_pairs(2, other);
  Pool first(files.config());
  ASSERT_EQ(Pool::State::kReady, first.state()) << first.error();

  // Another file, even with the same content, would number pairs its own way.
  Pool second(files.config(other));
  EXPECT_EQ(Pool::State::kError, second.state());
  EXPECT_NE(std::string::npos, second.error().find("inventory")) << second.error();

  // So would the same file once it changed.
  files.write_pairs(3);
  Pool third(files.config());
  EXPECT_EQ(Pool::State::kError, third.state());
  EXPECT_NE(std::string::npos, third.error().find("3")) << third.error();
}

namespace
{

/// Leave a claim region behind the way a process killed while creating it
/// would: created, \p size bytes of zeros, never published, initialized by
/// \p initializer (0 for nobody).
void leave_unpublished_region(const std::string & name, off_t size, uint32_t initializer)
{
  const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
  ASSERT_GE(fd, 0) << std::strerror(errno);
  ASSERT_EQ(0, ftruncate(fd, size));
  if (size > 0) {
    void * mapped = mmap(nullptr, static_cast<size_t>(size), PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(MAP_FAILED, mapped);
    // The initializer's tag is the header's fourth 32-bit word.
    static_cast<std::atomic<uint32_t> *>(mapped)[3].store(initializer);
    munmap(mapped, static_cast<size_t>(size));
  }
  close(fd);
}

}  // namespace

// A creator that died before sizing the region must not lock the host out.
TEST(NvSciEndpointPool, TakesOverARegionLeftEmptyByACrash)
{
  PoolFiles files("empty");
  files.write_pairs(1);
  leave_unpublished_region(files.region(), 0, 0);
  Pool pool(files.config());
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();
}

// Nor one that died after sizing it, before publishing it.
TEST(NvSciEndpointPool, TakesOverARegionWhoseCreatorDied)
{
  PoolFiles files("unpublished");
  files.write_pairs(1);
  Child dead = Child::start([](int) {return std::string("ready");});
  const uint32_t dead_tag = Pool::process_tag(dead.pid);
  dead.result();
  dead.kill_now();
  ASSERT_NE(0u, dead_tag);
  leave_unpublished_region(files.region(), 1 << 12, dead_tag);
  Pool pool(files.config());
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();
}

// But not from a creator that is still at it.
TEST(NvSciEndpointPool, WaitsForALiveCreator)
{
  PoolFiles files("live");
  files.write_pairs(1);
  leave_unpublished_region(files.region(), 1 << 12, Pool::process_tag(getpid()));
  Pool pool(files.config());
  EXPECT_EQ(Pool::State::kError, pool.state());
  EXPECT_NE(std::string::npos, pool.error().find("still creating")) << pool.error();
}

// -- Claims --

// The lower pid claims the pair and opens endpointA; the other finds it and
// opens endpointB. Each pair goes to one pair of processes at a time, and is
// free again once both have let go or exited.
TEST(NvSciEndpointPool, PairsProcessesAcrossTheHost)
{
  PoolFiles files("pairs");
  files.write_pairs(1);
  const pid_t self = getpid();
  const Pool::Config config = files.config();
  Pool pool(config);
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();

  // Holds the end it finds until it is killed.
  Child first = Child::start(
    [self, config](int) {
      Pool child_pool(config);
      Pool::Claim claim;
      if (claim_patiently(child_pool, self, &claim) != Pool::Result::kClaimed) {
        return std::string("none");
      }
      return claim.endpoint_name + (claim.claimer ? "+claimer" : "");
    });
  ASSERT_GT(first.pid, 0);
  if (first.pid < self) {
    first.kill_now();
    GTEST_SKIP() << "pids wrapped around";
  }
  Pool::Claim claim;
  ASSERT_EQ(Pool::Result::kClaimed, pool.claim(first.pid, &claim));
  EXPECT_TRUE(claim.claimer);
  EXPECT_EQ("ep_0_a", claim.endpoint_name);
  // Asking again gives the same pair.
  Pool::Claim again;
  ASSERT_EQ(Pool::Result::kClaimed, pool.claim(first.pid, &again));
  EXPECT_EQ(claim.index, again.index);
  // The higher pid waited for the claim, then took the other end.
  EXPECT_EQ("ep_0_b", first.result());

  // The only pair is held: another process gets nothing.
  Child second = Child::start([](int) {return std::string("idle");});
  EXPECT_EQ("idle", second.result());
  EXPECT_EQ(Pool::Result::kExhausted, pool.claim(second.pid, &claim));

  // The first child exits without letting go; this process letting go then
  // frees the pair.
  first.kill_now();
  EXPECT_EQ(Pool::Result::kExhausted, pool.claim(second.pid, &claim));
  pool.release(first.pid);
  ASSERT_EQ(Pool::Result::kClaimed, pool.claim(second.pid, &claim));
  EXPECT_EQ("ep_0_a", claim.endpoint_name);
  pool.release(second.pid);
  second.finish();
}

// A pair whose processes both exited is reaped by the next claim that finds
// the pool exhausted.
TEST(NvSciEndpointPool, ReapsPairsOfExitedProcesses)
{
  PoolFiles files("reap");
  files.write_pairs(1);
  const Pool::Config config = files.config();
  Pool pool(config);
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();

  int target_pipe[2];
  ASSERT_EQ(0, pipe(target_pipe));
  Child claimer = Child::start(
    [config, target_pipe](int) {
      pid_t target = 0;
      if (read(target_pipe[0], &target, sizeof(target)) != sizeof(target)) {
        return std::string("setup");
      }
      Pool child_pool(config);
      Pool::Claim claim;
      return child_pool.claim(target, &claim) == Pool::Result::kClaimed ?
      std::string("claimed") : std::string("not claimed");
    });
  Child target = Child::start([](int) {return std::string("target");});
  EXPECT_EQ("target", target.result());
  ASSERT_EQ(
    static_cast<ssize_t>(sizeof(pid_t)), ::write(target_pipe[1], &target.pid, sizeof(pid_t)));
  if (claimer.pid > target.pid || claimer.pid < getpid()) {
    claimer.kill_now();
    target.kill_now();
    GTEST_SKIP() << "pids wrapped around";
  }
  EXPECT_EQ("claimed", claimer.result());

  Child third = Child::start([](int) {return std::string("idle");});
  EXPECT_EQ("idle", third.result());
  Pool::Claim claim;
  EXPECT_EQ(Pool::Result::kExhausted, pool.claim(third.pid, &claim));
  claimer.kill_now();
  target.kill_now();
  EXPECT_EQ(Pool::Result::kClaimed, pool.claim(third.pid, &claim));
  pool.release(third.pid);
  third.finish();
  close(target_pipe[0]);
  close(target_pipe[1]);
}

// A claimer that lets go before its peer opened its end leaves the entry to
// the peer, which clears it the next time it looks for a pair.
TEST(NvSciEndpointPool, PeerClearsAnEntryItsClaimerLeft)
{
  PoolFiles files("orphan");
  files.write_pairs(1);
  const pid_t self = getpid();
  const Pool::Config config = files.config();
  Pool pool(config);
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();

  // Looks for a pair from this process only when told to.
  Child peer = Child::start(
    [self, config](int release_fd) {
      char c;
      if (read(release_fd, &c, 1) != 1) {
        return std::string("setup");
      }
      Pool child_pool(config);
      Pool::Claim claim;
      return child_pool.claim(self, &claim) == Pool::Result::kWaitForPeer ?
      std::string("waiting") : std::string("unexpected");
    });
  if (peer.pid < self) {
    peer.kill_now();
    GTEST_SKIP() << "pids wrapped around";
  }
  Pool::Claim claim;
  ASSERT_EQ(Pool::Result::kClaimed, pool.claim(peer.pid, &claim));
  pool.release(peer.pid);

  Child other = Child::start([](int) {return std::string("idle");});
  EXPECT_EQ("idle", other.result());
  // The entry still names the peer, which is alive.
  EXPECT_EQ(Pool::Result::kExhausted, pool.claim(other.pid, &claim));
  peer.signal();
  EXPECT_EQ("waiting", peer.result());
  EXPECT_EQ(Pool::Result::kClaimed, pool.claim(other.pid, &claim));
  pool.release(other.pid);
  other.finish();
  peer.finish();
}

TEST(NvSciEndpointPool, TheHigherPidOnlyEverWaitsForItsPeer)
{
  PoolFiles files("waits");
  files.write_pairs(4);
  Pool pool(files.config());
  ASSERT_EQ(Pool::State::kReady, pool.state()) << pool.error();
  // The parent of this process started before it.
  const pid_t parent = getppid();
  if (parent <= 1 || parent > getpid()) {
    GTEST_SKIP() << "needs a parent with a lower pid";
  }
  Pool::Claim claim;
  EXPECT_EQ(Pool::Result::kWaitForPeer, pool.claim(parent, &claim));
  EXPECT_EQ(Pool::Result::kUnavailable, pool.claim(getpid(), &claim));
}

// Processes claiming at once never get the same pair, and the pool reports
// itself exhausted only when every pair is held.
TEST(NvSciEndpointPool, ConcurrentClaimsNeverShareAPair)
{
  constexpr size_t kPairs = 5;
  constexpr int kClaimers = 12;
  PoolFiles files("concurrent");
  files.write_pairs(kPairs);
  const Pool::Config config = files.config();
  {
    Pool creator(config);
    ASSERT_EQ(Pool::State::kReady, creator.state()) << creator.error();
  }

  // Claimers first, so that the target they claim for has a higher pid.
  int go[2];
  ASSERT_EQ(0, pipe(go));
  int target_pipe[2];
  ASSERT_EQ(0, pipe(target_pipe));
  std::vector<Child> claimers;
  for (int i = 0; i < kClaimers; ++i) {
    claimers.push_back(
      Child::start(
        [&config, &go, &target_pipe](int) {
          close(go[1]);
          pid_t target = 0;
          char c;
          if (read(target_pipe[0], &target, sizeof(target)) != sizeof(target) ||
          read(go[0], &c, 1) < 0)
          {
            return std::string("setup");
          }
          Pool pool(config);
          Pool::Claim claim;
          switch (pool.claim(target, &claim)) {
            case Pool::Result::kClaimed:
              return std::to_string(claim.index);
            case Pool::Result::kExhausted:
              return std::string("exhausted");
            default:
              return std::string("other");
          }
        }));
  }
  Child target = Child::start(
    [&go](int) {
      close(go[1]);
      return std::string("target");
    });
  for (Child & c : claimers) {
    if (c.pid > target.pid) {
      for (Child & k : claimers) {
        k.kill_now();
      }
      target.kill_now();
      GTEST_SKIP() << "pids wrapped around";
    }
  }
  for (size_t i = 0; i < claimers.size(); ++i) {
    ASSERT_EQ(
      static_cast<ssize_t>(sizeof(pid_t)), ::write(target_pipe[1], &target.pid, sizeof(pid_t)));
  }
  close(go[1]);  // Start them all at once.

  std::set<std::string> indices;
  int exhausted = 0;
  for (Child & c : claimers) {
    const std::string r = c.result();
    if (r == "exhausted") {
      ++exhausted;
    } else {
      EXPECT_TRUE(indices.insert(r).second) << "pair " << r << " was claimed twice";
    }
  }
  EXPECT_EQ(kPairs, indices.size());
  EXPECT_EQ(kClaimers - static_cast<int>(kPairs), exhausted);
  for (Child & c : claimers) {
    c.finish();
  }
  target.finish();
  close(go[0]);
  close(target_pipe[0]);
  close(target_pipe[1]);
}
