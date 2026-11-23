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

// Private to the cuda_buffer library. Free of NvSci (POSIX shared memory,
// atomics and JSON only), so that it can be tested anywhere.

#ifndef NVSCI_ENDPOINT_POOL_HPP_
#define NVSCI_ENDPOINT_POOL_HPP_

#include <sys/types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cuda_buffer_backend
{
namespace detail
{

/// \brief The NvSciIpc endpoint pairs this host lets processes claim, and the
/// host-wide record of who holds which.
///
/// The same design as Halos IPC's dynamic-mode endpoint pool
/// (discovery/EndpointPool, discovery_dynamic_ddd.md section 3.6):
///
/// - NvSciIpc cannot list or create endpoints at runtime; pairs are declared
///   ahead of time in /etc/nvsciipc.cfg. An explicit, curated inventory file
///   says which of them are this pool's, in the same JSON format as Halos
///   IPC's (version 1, "pool" of {"endpointA", "endpointB"} pairs). One
///   canonical file per host: CUDA_BUFFER_NVSCI_ENDPOINT_POOL, by default
///   /etc/cuda_buffer/nvsci_endpoint_pool.json. Its pairs must not be in any
///   other pool, such as a Halos IPC one.
/// - Claims live in one shared-memory region per host, created with O_EXCL,
///   sized, and published by storing a magic number last (release), which
///   attachers load first (acquire). The region records the inventory's
///   canonical path and entry count, and a process with another inventory
///   refuses to use it rather than claim under a different numbering. It is
///   world read-write (0666), and never unlinked while the host is up.
/// - Entry i of the region is pair i of the inventory, claimed with a
///   compare-and-swap; a sweep reports the pool exhausted only when no entry
///   is free.
///
/// Where it differs, and why:
///
/// - Handoff. A Halos IPC claimer tells its peer which pair it claimed in a
///   PAIR_REQUEST over its discovery socket. There is no such channel here, so
///   an entry names both processes: the one with the lower pid claims a pair
///   for the two and opens endpointA, and the other finds the entry naming
///   it and opens endpointB. Only one of the two ever claims, so they cannot
///   claim two pairs for each other, and no tie-break is needed.
/// - Release. Halos IPC holds a claim until its process exits, since neither
///   peer can tell when the other is done with the pair. Here each process
///   clears its own half of the entry once it has closed its end, and the
///   pair is free when both halves are clear. A half whose process has exited
///   is cleared by whichever process next finds the pool exhausted.
/// - Recovery. If the process creating the region dies before publishing it,
///   another takes over its initialization once it is overdue, instead of
///   the pool staying unusable until reboot.
///
/// Processes are recorded by pid and a tag of their start time, so one that
/// reuses an exited process's pid is not mistaken for it (one time in 1023 it
/// is, and its pair stays claimed until that process exits). Thread-safe.
class NvSciEndpointPool
{
public:
  /// Default location of the inventory.
  static constexpr const char * kDefaultInventoryPath =
    "/etc/cuda_buffer/nvsci_endpoint_pool.json";
  /// Environment variable that overrides it.
  static constexpr const char * kInventoryEnv = "CUDA_BUFFER_NVSCI_ENDPOINT_POOL";
  /// Most pairs an inventory may list.
  static constexpr size_t kMaxPairs = 4096;

  struct EndpointPair
  {
    std::string endpoint_a;
    std::string endpoint_b;
  };

  struct Config
  {
    /// Inventory file. Empty: kInventoryEnv, then kDefaultInventoryPath.
    std::string inventory_path;
    /// Claim region, one per host.
    std::string region_name = "/cuda_buffer_nvsci_endpoint_pool";
    /// NvSciIpc configuration the inventory is checked against. Empty:
    /// CUDA_BUFFER_NVSCIIPC_CFG, then /etc/nvsciipc.cfg.
    std::string nvsciipc_config_path;
    /// How long to wait for another process to publish the region before
    /// taking over its initialization.
    std::chrono::milliseconds attach_wait{500};
  };

  enum class State
  {
    kReady,
    /// There is no inventory file: this host has no pool. Not an error.
    kNoInventory,
    /// The inventory or the claim region is unusable; see error().
    kError,
  };

  /// \brief Load the inventory and attach to the claim region. Never throws.
  explicit NvSciEndpointPool(Config config);
  /// \brief The same, with the default configuration.
  NvSciEndpointPool();
  ~NvSciEndpointPool();

  NvSciEndpointPool(const NvSciEndpointPool &) = delete;
  NvSciEndpointPool & operator=(const NvSciEndpointPool &) = delete;

  State state() const {return state_;}
  /// Why the pool is not ready, for a log message.
  const std::string & error() const {return error_;}
  /// Problems that do not stop the pool, such as inventory pairs missing from
  /// the NvSciIpc configuration, for log messages.
  const std::vector<std::string> & warnings() const {return warnings_;}
  const std::string & inventory_path() const {return inventory_path_;}
  const std::string & region_name() const {return config_.region_name;}
  size_t pair_count() const {return pairs_.size();}

  enum class Result
  {
    /// \p out names the endpoint to open.
    kClaimed,
    /// The peer has the lower pid and has not claimed a pair for the two
    /// processes yet: ask again shortly.
    kWaitForPeer,
    /// Every pair is held.
    kExhausted,
    /// Not ready, or the peer is not running.
    kUnavailable,
  };

  struct Claim
  {
    size_t index{0};
    std::string endpoint_name;
    /// This process claimed the pair (and opens endpointA), rather than its
    /// peer (endpointB).
    bool claimer{false};
  };

  /// \brief The endpoint this process opens to talk to process \p peer.
  /// \details Asking again for a peer this process holds a pair for returns
  /// the same pair.
  Result claim(pid_t peer, Claim * out);

  /// \brief Give back this process's half of the pair it holds for \p peer.
  /// \details Call it only once this process has closed its end, so that the
  /// pair cannot be handed to someone else while that end is open. If its end
  /// could not be opened, only the claimer gives the pair back (and claims
  /// another next time): the peer keeps its half and tries the same end
  /// again, since its claimer is waiting on the other one.
  void release(pid_t peer);

  /// \brief Parse and validate an inventory.
  /// \throws std::runtime_error if it is malformed or invalid.
  static std::vector<EndpointPair> parse_inventory(const std::string & json);

  /// \brief How the region records process \p pid: its pid, and a tag of its
  /// start time in the top bits. 0 if it is not running.
  static uint32_t process_tag(pid_t pid);

  /// \brief The INTER_PROCESS pairs declared in an NvSciIpc configuration.
  static std::vector<std::pair<std::string, std::string>> parse_nvsciipc_config(
    const std::string & text);

private:
  struct Held
  {
    size_t index;
    bool claimer;
  };

  void open_region(const std::string & canonical_path);
  /// Clear the halves held by processes that have exited.
  void reap();
  /// Clear the halves of entries that name this process as the peer of a
  /// claimer that has let go, if this process does not use them.
  void clear_orphans_locked();

  const Config config_;
  State state_{State::kError};
  std::string error_;
  std::vector<std::string> warnings_;
  std::string inventory_path_;
  std::vector<EndpointPair> pairs_;

  void * region_{nullptr};
  size_t region_size_{0};
  uint32_t self_tag_{0};

  std::mutex mutex_;
  /// Pairs this process holds, by peer.
  std::unordered_map<pid_t, Held> held_;
};

}  // namespace detail
}  // namespace cuda_buffer_backend

#endif  // NVSCI_ENDPOINT_POOL_HPP_
