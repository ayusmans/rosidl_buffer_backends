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

#include "nvsci_endpoint_pool.hpp"

#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cuda_buffer_backend
{
namespace detail
{

namespace
{

constexpr uint32_t kMagic = 0x43425045;  // "CBPE"
constexpr uint32_t kLayoutVersion = 1;
constexpr size_t kPathMax = 512;

/// Fixed header of the claim region, followed by one claim word per pair.
/// Written once, before magic is published; only the claim words change after.
struct RegionHeader
{
  /// kMagic once the header is complete: stored last (release), loaded first
  /// (acquire). The publication gate.
  std::atomic<uint32_t> magic;
  uint32_t version;
  uint32_t entry_count;
  /// Tag of the process initializing the header, so that another can take
  /// over if it dies before publishing.
  std::atomic<uint32_t> initializer;
  /// Canonical path of the inventory, '\0'-terminated.
  char inventory_path[kPathMax];
};

static_assert(offsetof(RegionHeader, magic) == 0, "magic stays at offset 0");
static_assert(sizeof(RegionHeader) == 528, "layout is shared: bump kLayoutVersion to change it");
static_assert(
  sizeof(RegionHeader) % alignof(std::atomic<uint64_t>) == 0, "claim words must be aligned");
static_assert(
  std::atomic<uint32_t>::is_always_lock_free && std::atomic<uint64_t>::is_always_lock_free,
  "shared atomics must be lock-free");

/// A claim word: the claimer's tag in the high half, its peer's in the low half.
/// 0 is a free pair.
using Word = uint64_t;

Word make_word(uint32_t claimer, uint32_t peer)
{
  return (static_cast<Word>(claimer) << 32) | peer;
}

uint32_t claimer_of(Word w) {return static_cast<uint32_t>(w >> 32);}
uint32_t peer_of(Word w) {return static_cast<uint32_t>(w);}

/// Linux pids are below 2^22 (PID_MAX_LIMIT).
constexpr uint32_t kPidBits = 22;
constexpr uint32_t kPidMask = (1u << kPidBits) - 1;

/// Start time of \p pid in clock ticks since boot, or 0 if it is not running.
uint64_t start_time_of(pid_t pid)
{
  std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
  std::string text;
  if (!std::getline(stat, text)) {
    return 0;
  }
  // The command name may contain spaces; fields resume after the last ')'.
  const size_t paren = text.rfind(')');
  if (paren == std::string::npos || paren + 2 > text.size()) {
    return 0;
  }
  std::istringstream fields(text.substr(paren + 2));
  std::string field;
  // starttime is field 22; field 3 is the first after the command name.
  for (int i = 3; i <= 22; ++i) {
    if (!(fields >> field)) {
      return 0;
    }
  }
  return std::strtoull(field.c_str(), nullptr, 10);
}

/// A process: its pid and, in the top bits, a nonzero tag of its start time,
/// so that a process that reuses an exited one's pid is told apart from it.
/// 0 if it is not running.
uint32_t tag_of(pid_t pid)
{
  if (pid <= 0 || static_cast<uint32_t>(pid) > kPidMask) {
    return 0;
  }
  const uint64_t start = start_time_of(pid);
  if (start == 0) {
    return 0;
  }
  const auto start_tag = static_cast<uint32_t>(start % 1023 + 1);
  return static_cast<uint32_t>(pid) | (start_tag << kPidBits);
}

bool tag_alive(uint32_t tag)
{
  return tag != 0 && tag_of(static_cast<pid_t>(tag & kPidMask)) == tag;
}

std::atomic<Word> * slots(void * region)
{
  return reinterpret_cast<std::atomic<Word> *>(
    static_cast<char *>(region) + sizeof(RegionHeader));
}

std::string env_or(const char * name, const char * fallback)
{
  const char * value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? value : fallback;
}

bool read_file(const std::string & path, std::string * text)
{
  std::ifstream file(path);
  if (!file) {
    return false;
  }
  std::stringstream contents;
  contents << file.rdbuf();
  *text = contents.str();
  return true;
}

}  // namespace

std::vector<NvSciEndpointPool::EndpointPair> NvSciEndpointPool::parse_inventory(
  const std::string & text)
{
  using nlohmann::json;
  json doc;
  try {
    doc = json::parse(text);
  } catch (const json::exception & e) {
    throw std::runtime_error(std::string("not valid JSON: ") + e.what());
  }
  if (!doc.is_object()) {
    throw std::runtime_error("the top level must be an object");
  }
  for (const auto & item : doc.items()) {
    if (item.key() != "version" && item.key() != "pool") {
      throw std::runtime_error("unknown key '" + item.key() + "'");
    }
  }
  const auto version = doc.find("version");
  if (version == doc.end() || !version->is_number_integer() || *version != 1) {
    throw std::runtime_error("\"version\" must be 1");
  }
  const auto pool = doc.find("pool");
  if (pool == doc.end() || !pool->is_array()) {
    throw std::runtime_error("\"pool\" must be an array");
  }
  if (pool->size() > kMaxPairs) {
    throw std::runtime_error("\"pool\" lists more than " + std::to_string(kMaxPairs) + " pairs");
  }

  std::vector<EndpointPair> pairs;
  std::set<std::string> names;
  for (size_t i = 0; i < pool->size(); ++i) {
    const json & entry = (*pool)[i];
    const std::string where = "pool[" + std::to_string(i) + "]";
    if (!entry.is_object()) {
      throw std::runtime_error(where + " must be an object");
    }
    for (const auto & item : entry.items()) {
      const std::string & key = item.key();
      if (key == "endpointA" || key == "endpointB") {
        if (!item.value().is_string() || item.value().get<std::string>().empty()) {
          throw std::runtime_error(where + "." + key + " must be a non-empty string");
        }
      } else if (key == "nframes" || key == "frameSize") {
        // Advisory geometry, as in Halos IPC: the real values are read from
        // the endpoint once it is open.
        if (!item.value().is_number_integer() || item.value().get<int64_t>() < 1 ||
          (key == "nframes" && item.value().get<int64_t>() > 65535))
        {
          throw std::runtime_error(where + "." + key + " is out of range");
        }
      } else {
        throw std::runtime_error(where + ": unknown key '" + key + "'");
      }
    }
    if (!entry.contains("endpointA") || !entry.contains("endpointB")) {
      throw std::runtime_error(where + " needs \"endpointA\" and \"endpointB\"");
    }
    EndpointPair pair{entry["endpointA"].get<std::string>(), entry["endpointB"].get<std::string>()};
    // Each endpoint is one claim word's worth of state: listing it twice would
    // let two processes open it.
    for (const std::string * name : {&pair.endpoint_a, &pair.endpoint_b}) {
      if (!names.insert(*name).second) {
        throw std::runtime_error("endpoint '" + *name + "' is listed more than once");
      }
    }
    pairs.push_back(std::move(pair));
  }
  return pairs;
}

uint32_t NvSciEndpointPool::process_tag(pid_t pid)
{
  return tag_of(pid);
}

std::vector<std::pair<std::string, std::string>> NvSciEndpointPool::parse_nvsciipc_config(
  const std::string & text)
{
  std::vector<std::pair<std::string, std::string>> pairs;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream fields(line);
    std::string backend;
    std::string first;
    std::string second;
    if (fields >> backend >> first >> second && backend == "INTER_PROCESS") {
      pairs.emplace_back(first, second);
    }
  }
  return pairs;
}

NvSciEndpointPool::NvSciEndpointPool()
: NvSciEndpointPool(Config())
{
}

NvSciEndpointPool::NvSciEndpointPool(Config config)
: config_(std::move(config))
{
  inventory_path_ = !config_.inventory_path.empty() ?
    config_.inventory_path : env_or(kInventoryEnv, kDefaultInventoryPath);
  self_tag_ = tag_of(getpid());

  std::string text;
  if (!read_file(inventory_path_, &text)) {
    const int err = errno;
    state_ = err == ENOENT ? State::kNoInventory : State::kError;
    error_ = "cannot read the NvSci endpoint pool inventory " + inventory_path_ + ": " +
      std::strerror(err);
    return;
  }
  try {
    pairs_ = parse_inventory(text);
  } catch (const std::exception & e) {
    error_ = "invalid NvSci endpoint pool inventory " + inventory_path_ + ": " + e.what();
    return;
  }
  if (pairs_.empty()) {
    warnings_.push_back("NvSci endpoint pool inventory " + inventory_path_ + " lists no pairs");
    state_ = State::kReady;
    return;
  }

  // Build-time reconciliation in Halos IPC; here a check at startup: a pair
  // that NvSciIpc does not know cannot be opened.
  const std::string cfg_path = !config_.nvsciipc_config_path.empty() ?
    config_.nvsciipc_config_path : env_or("CUDA_BUFFER_NVSCIIPC_CFG", "/etc/nvsciipc.cfg");
  std::string cfg_text;
  if (read_file(cfg_path, &cfg_text)) {
    std::set<std::pair<std::string, std::string>> declared;
    for (const auto & p : parse_nvsciipc_config(cfg_text)) {
      declared.insert(p);
      declared.insert({p.second, p.first});
    }
    for (const EndpointPair & p : pairs_) {
      if (declared.count({p.endpoint_a, p.endpoint_b}) == 0) {
        warnings_.push_back(
          "NvSci endpoint pool pair " + p.endpoint_a + "/" + p.endpoint_b +
          " is not an INTER_PROCESS pair in " + cfg_path + "; it cannot be opened");
      }
    }
  }

  char canonical[PATH_MAX];
  if (realpath(inventory_path_.c_str(), canonical) == nullptr) {
    error_ = "cannot resolve " + inventory_path_ + ": " + std::strerror(errno);
    return;
  }
  if (std::strlen(canonical) >= kPathMax) {
    // Truncating could make two different paths look the same.
    error_ = "the NvSci endpoint pool inventory path is too long: " + std::string(canonical);
    return;
  }
  try {
    open_region(canonical);
    state_ = State::kReady;
  } catch (const std::exception & e) {
    error_ = e.what();
  }
}

NvSciEndpointPool::~NvSciEndpointPool()
{
  if (region_ != nullptr) {
    munmap(region_, region_size_);
  }
}

void NvSciEndpointPool::open_region(const std::string & canonical_path)
{
  const std::string & name = config_.region_name;
  const size_t total = sizeof(RegionHeader) + pairs_.size() * sizeof(Word);
  auto fail = [&name](const std::string & what, int err) {
      throw std::runtime_error(
              "NvSci endpoint pool region " + name + ": " + what +
              (err != 0 ? std::string(": ") + std::strerror(err) : std::string()));
    };

  // Shared by every process of every user on the host. The creator opens it
  // up with fchmod: the umask is process-wide, so clearing it would loosen the
  // files other threads create meanwhile.
  int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
  const bool created = fd >= 0;
  if (created) {
    (void)fchmod(fd, 0666);
    if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
      const int err = errno;
      close(fd);
      fail("cannot size it", err);
    }
  } else if (errno == EEXIST) {
    fd = shm_open(name.c_str(), O_RDWR, 0);
  }
  if (fd < 0) {
    fail("cannot open it", errno);
  }

  // Attach: wait for the creator to size the region and publish the header.
  // A creator that is overdue is taken to have died, and the first process to
  // notice initializes the region in its place. Nobody uses the region before
  // it is published, so there is nothing of the dead creator's to keep.
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto take_over_after = created ? Clock::duration::zero() : config_.attach_wait;
  const auto give_up = start + config_.attach_wait * 3;
  RegionHeader * header = nullptr;
  auto unmap_header = [&header]() {
      if (header != nullptr) {
        munmap(header, sizeof(RegionHeader));
        header = nullptr;
      }
    };
  for (;; ) {
    const auto now = Clock::now();
    const bool overdue = now - start >= take_over_after;
    struct stat info {};
    if (fstat(fd, &info) != 0) {
      const int err = errno;
      unmap_header();
      close(fd);
      fail("cannot stat it", err);
    }
    if (static_cast<size_t>(info.st_size) < sizeof(RegionHeader)) {
      // Reading past the end of a mapping is a SIGBUS: wait until it is sized.
      if (overdue && ftruncate(fd, static_cast<off_t>(total)) != 0) {
        const int err = errno;
        close(fd);
        fail("cannot size it", err);
      }
      if (!overdue) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      continue;
    }
    if (header == nullptr) {
      void * mapped = mmap(
        nullptr, sizeof(RegionHeader), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (mapped == MAP_FAILED) {
        const int err = errno;
        close(fd);
        fail("cannot map it", err);
      }
      header = static_cast<RegionHeader *>(mapped);
    }
    if (header->magic.load(std::memory_order_acquire) == kMagic) {
      break;
    }
    if (overdue) {
      uint32_t current = header->initializer.load(std::memory_order_acquire);
      if ((current == 0 || !tag_alive(current)) &&
        header->initializer.compare_exchange_strong(
          current, self_tag_, std::memory_order_acq_rel, std::memory_order_acquire))
      {
        if (static_cast<size_t>(info.st_size) < total &&
          ftruncate(fd, static_cast<off_t>(total)) != 0)
        {
          const int err = errno;
          unmap_header();
          close(fd);
          fail("cannot size it", err);
        }
        header->version = kLayoutVersion;
        header->entry_count = static_cast<uint32_t>(pairs_.size());
        std::memset(header->inventory_path, 0, sizeof(header->inventory_path));
        std::memcpy(header->inventory_path, canonical_path.data(), canonical_path.size());
        header->magic.store(kMagic, std::memory_order_release);
        break;
      }
    }
    if (now >= give_up) {
      unmap_header();
      close(fd);
      fail("another process is still creating it", 0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // The numbering of the pairs is the inventory's, so every process must use
  // the same file.
  std::string problem;
  if (header->version != kLayoutVersion) {
    problem = "it has layout version " + std::to_string(header->version);
  } else if (std::strncmp(header->inventory_path, canonical_path.c_str(), kPathMax) != 0) {
    problem = "it was created for the inventory " +
      std::string(header->inventory_path, strnlen(header->inventory_path, kPathMax)) +
      ", not " + canonical_path;
  } else if (header->entry_count != pairs_.size()) {
    problem = "it was created for " + std::to_string(header->entry_count) +
      " pairs, and the inventory now lists " + std::to_string(pairs_.size());
  }
  unmap_header();
  if (!problem.empty()) {
    close(fd);
    fail(
      problem + ". Every process on the host must use the same inventory, which must not "
      "change while any of them runs; to start over, stop them and remove /dev/shm" + name, 0);
  }

  struct stat info {};
  if (fstat(fd, &info) != 0 || static_cast<size_t>(info.st_size) < total) {
    close(fd);
    fail("it is smaller than its header says", 0);
  }
  void * mapped = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int err = errno;
  close(fd);
  if (mapped == MAP_FAILED) {
    fail("cannot map it", err);
  }
  region_ = mapped;
  region_size_ = total;
}

NvSciEndpointPool::Result NvSciEndpointPool::claim(pid_t peer, Claim * out)
{
  if (state_ != State::kReady || region_ == nullptr || self_tag_ == 0) {
    return Result::kUnavailable;
  }
  const pid_t self = getpid();
  const uint32_t peer_tag = tag_of(peer);
  if (peer == self || peer_tag == 0) {
    return Result::kUnavailable;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto held = held_.find(peer);
  if (held != held_.end()) {
    out->index = held->second.index;
    const EndpointPair & pair = pairs_[held->second.index];
    out->endpoint_name = held->second.claimer ? pair.endpoint_a : pair.endpoint_b;
    out->claimer = held->second.claimer;
    return Result::kClaimed;
  }

  std::atomic<Word> * words = slots(region_);
  const size_t count = pairs_.size();
  if (self < peer) {
    // This process claims a pair for both, and opens endpointA.
    const Word desired = make_word(self_tag_, peer_tag);
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t i = 0; i < count; ++i) {
        Word expected = 0;
        // A lost race for one pair moves on to the next; only a sweep that
        // finds none free means the pool is exhausted.
        if (words[i].load(std::memory_order_acquire) == 0 &&
          words[i].compare_exchange_strong(
            expected, desired, std::memory_order_acq_rel, std::memory_order_acquire))
        {
          held_[peer] = Held{i, true};
          *out = Claim{i, pairs_[i].endpoint_a, true};
          return Result::kClaimed;
        }
      }
      if (pass == 0) {
        reap();
      }
    }
    return Result::kExhausted;
  }

  // The peer claims; this process finds the pair it claimed for the two, and
  // opens endpointB.
  const Word wanted = make_word(peer_tag, self_tag_);
  for (size_t i = 0; i < count; ++i) {
    if (words[i].load(std::memory_order_acquire) == wanted) {
      held_[peer] = Held{i, false};
      *out = Claim{i, pairs_[i].endpoint_b, false};
      return Result::kClaimed;
    }
  }
  clear_orphans_locked();
  return Result::kWaitForPeer;
}

void NvSciEndpointPool::release(pid_t peer)
{
  if (region_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto held = held_.find(peer);
  if (held == held_.end()) {
    return;
  }
  const Held h = held->second;
  held_.erase(held);

  std::atomic<Word> & word = slots(region_)[h.index];
  Word current = word.load(std::memory_order_acquire);
  for (;; ) {
    uint32_t claimer = claimer_of(current);
    uint32_t other = peer_of(current);
    uint32_t & mine = h.claimer ? claimer : other;
    uint32_t & theirs = h.claimer ? other : claimer;
    if (mine != self_tag_) {
      return;  // Not ours (any more): leave it alone.
    }
    mine = 0;
    // The pair is free once neither process holds it.
    if (theirs != 0 && !tag_alive(theirs)) {
      theirs = 0;
    }
    if (word.compare_exchange_weak(
        current, make_word(claimer, other), std::memory_order_acq_rel,
        std::memory_order_acquire))
    {
      return;
    }
  }
}

void NvSciEndpointPool::reap()
{
  std::atomic<Word> * words = slots(region_);
  for (size_t i = 0; i < pairs_.size(); ++i) {
    Word current = words[i].load(std::memory_order_acquire);
    if (current == 0) {
      continue;
    }
    uint32_t claimer = claimer_of(current);
    uint32_t peer = peer_of(current);
    if (claimer != 0 && !tag_alive(claimer)) {
      claimer = 0;
    }
    if (peer != 0 && !tag_alive(peer)) {
      peer = 0;
    }
    const Word updated = make_word(claimer, peer);
    if (updated != current) {
      // If the word changed meanwhile, its holders are alive: leave it.
      words[i].compare_exchange_strong(
        current, updated, std::memory_order_acq_rel, std::memory_order_acquire);
    }
  }
}

void NvSciEndpointPool::clear_orphans_locked()
{
  // A claimer that let go before this process opened its end leaves the
  // entry naming this process alone. This process is not using it, so it can
  // clear it; nobody else can tell.
  std::set<size_t> mine;
  for (const auto & entry : held_) {
    mine.insert(entry.second.index);
  }
  std::atomic<Word> * words = slots(region_);
  for (size_t i = 0; i < pairs_.size(); ++i) {
    Word current = words[i].load(std::memory_order_acquire);
    if (peer_of(current) != self_tag_ || mine.count(i) != 0) {
      continue;
    }
    const uint32_t claimer = claimer_of(current);
    if (claimer == 0 || !tag_alive(claimer)) {
      words[i].compare_exchange_strong(
        current, 0, std::memory_order_acq_rel, std::memory_order_acquire);
    }
  }
}

}  // namespace detail
}  // namespace cuda_buffer_backend
