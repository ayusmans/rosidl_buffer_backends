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

#include "nvsci_peer.hpp"

#include <fcntl.h>
#include <rcutils/logging_macros.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_error.hpp"

namespace cuda_buffer_backend
{

namespace
{

/// How long an NvSci object import may wait for its peer's side.
constexpr int64_t kImportTimeoutUs = 1000000;

enum MessageType : uint32_t
{
  kHello = 1,
  kSyncChannel = 2,
  kBlock = 3,
  kRetire = 4,
  kNack = 5,
  kAck = 6,
};

/// A random, nonzero connection id: unique across reconnects and processes.
uint64_t random_connection_id()
{
  std::random_device device;
  std::mt19937_64 rng((static_cast<uint64_t>(device()) << 32) ^ device());
  uint64_t id = 0;
  while (id == 0) {
    id = rng();
  }
  return id;
}

/// Header of every message; the payload follows it.
struct MessageHeader
{
  uint32_t type;
  uint32_t reserved;
  uint64_t id;
  uint64_t arg;
  uint64_t payload_size;
};

/// Each frame starts with the number of message bytes it carries; the rest
/// of the frame is padding.
using FrameHeader = uint32_t;

/// Messages are bounded so that a corrupt length cannot exhaust memory.
constexpr uint64_t kMaxPayload = 16u << 20;

/// Frees an NvSci attribute list at scope exit.
template<typename List, void (*Free)(List)>
struct ScopedList
{
  List list{nullptr};
  ~ScopedList()
  {
    if (list) {Free(list);}
  }
};
using ScopedBufAttrs = ScopedList<NvSciBufAttrList, NvSciBufAttrListFree>;
using ScopedSyncAttrs = ScopedList<NvSciSyncAttrList, NvSciSyncAttrListFree>;

void append(std::vector<uint8_t> & out, const void * data, size_t size)
{
  const auto * bytes = static_cast<const uint8_t *>(data);
  out.insert(out.end(), bytes, bytes + size);
}

}  // namespace

// -- NvSciImportedBlock --

NvSciImportedBlock::NvSciImportedBlock(
  std::unique_ptr<NvSciMapping> block_mapping, IPCMetadata * block_lease)
: mapping(std::move(block_mapping)), lease(block_lease)
{
}

NvSciImportedBlock::~NvSciImportedBlock()
{
  if (lease != nullptr) {
    munmap(lease, sizeof(IPCMetadata));
  }
}

// -- NvSciPeer --

NvSciPeer::NvSciPeer(
  NvSciRuntime & runtime, pid_t pid, uint64_t session, NvSciIpcEndpoint endpoint,
  std::function<void()> on_closed)
: runtime_(runtime), pid_(pid), session_(session),
  connection_id_(random_connection_id()), endpoint_(endpoint),
  on_closed_(std::move(on_closed))
{
  try {
    NvSciIpcEndpointInfo info{};
    NVSCI_CHECK(NvSciIpcGetEndpointInfo(endpoint_, &info));
    frame_size_ = info.frame_size;
    if (frame_size_ <= sizeof(FrameHeader)) {
      throw CudaError("NvSciIpc frame size too small");
    }
    NVSCI_CHECK(NvSciIpcGetLinuxEventFd(endpoint_, &event_fd_));

    // What this process needs in order to consume the peer's buffers: read
    // access from its CUDA device, and the right to wait on the peer's writes.
    local_buf_waiter_attrs_ = runtime_.create_buf_attrs(0, NvSciBufAccessPerm_Readonly);
    local_sync_waiter_attrs_ = runtime_.create_sync_attrs(cudaNvSciSyncAttrWait);

    NVSCI_CHECK(NvSciIpcResetEndpointSafe(endpoint_));
  } catch (...) {
    if (local_buf_waiter_attrs_) {NvSciBufAttrListFree(local_buf_waiter_attrs_);}
    if (local_sync_waiter_attrs_) {NvSciSyncAttrListFree(local_sync_waiter_attrs_);}
    NvSciIpcCloseEndpointSafe(endpoint_, false);
    throw;
  }
}

NvSciPeer::~NvSciPeer()
{
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    imported_blocks_.clear();
    imported_channels_.clear();
    if (local_buf_waiter_attrs_) {NvSciBufAttrListFree(local_buf_waiter_attrs_);}
    if (local_sync_waiter_attrs_) {NvSciSyncAttrListFree(local_sync_waiter_attrs_);}
    if (peer_buf_waiter_attrs_) {NvSciBufAttrListFree(peer_buf_waiter_attrs_);}
    if (peer_sync_waiter_attrs_) {NvSciSyncAttrListFree(peer_sync_waiter_attrs_);}
  }
  NvSciIpcCloseEndpointSafe(endpoint_, false);
  // Only now may the endpoint pair be handed to anyone else.
  if (on_closed_) {
    on_closed_();
  }
}

bool NvSciPeer::lost() const
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  return lost_;
}

bool NvSciPeer::ready() const
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  return peer_buf_waiter_attrs_ != nullptr && peer_sync_waiter_attrs_ != nullptr;
}

void NvSciPeer::shutdown()
{
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    shut_down_ = true;
  }
  state_cv_.notify_all();
}

void NvSciPeer::acknowledge(uint32_t stream, uint64_t seq) noexcept
{
  try {
    std::lock_guard<std::mutex> lock(io_mutex_);
    uint64_t & pending = pending_acks_[stream];
    pending = std::max(pending, seq);
    // While earlier messages are still queued, this one waits for the next
    // service(), and later imports raise it in place.
    if (outbox_.empty()) {
      send_acks_locked();
    }
  } catch (const std::exception & e) {
    // The publisher then ends the reservation at its deadline instead.
    RCUTILS_LOG_WARN_NAMED(
      "cuda_buffer_backend", "Cannot acknowledge NvSci descriptors to process %d: %s",
      static_cast<int>(pid_), e.what());
  }
}

void NvSciPeer::send_acks_locked()
{
  std::unordered_map<uint32_t, uint64_t> acks;
  acks.swap(pending_acks_);
  for (const auto & ack : acks) {
    enqueue_locked(kAck, ack.first, ack.second, {});
  }
}

void NvSciPeer::service()
{
  std::vector<Message> inbound;
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (lost_) {
      return;
    }
    uint32_t events = 0;
    if (NvSciIpcGetEventSafe(endpoint_, &events) != NvSciError_Success) {
      return;
    }
    // A reset before the connection is established is part of the handshake
    // (the peer reset its own end); after it, the peer has gone away.
    if (connected_ && (events & NV_SCI_IPC_EVENT_CONN_RESET)) {
      lost_ = true;
      return;
    }
    if (!connected_ && (events & NV_SCI_IPC_EVENT_CONN_EST)) {
      connected_ = true;
      queue_hello_locked();
    }
    if (connected_) {
      read_locked(&inbound);
      flush_locked();
      if (outbox_.empty()) {
        send_acks_locked();
      }
    }
  }
  // Imports use the endpoint too, so they hold the I/O lock. The runtime is
  // told about a hello only after the lock is released: the runtime takes its
  // own locks before a peer's, never after.
  bool became_ready = false;
  std::vector<std::pair<uint32_t, uint64_t>> acks;
  for (const Message & message : inbound) {
    if (message.type == kAck) {
      // Handed to the runtime below, outside this peer's locks.
      acks.emplace_back(static_cast<uint32_t>(message.id), message.arg);
      continue;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    try {
      became_ready = handle(message) || became_ready;
    } catch (const std::exception & e) {
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend", "Dropping an NvSci message from process %d: %s",
        static_cast<int>(pid_), e.what());
      reject_locked(message);
    }
  }
  if (became_ready) {
    // Buffers allocated from now on can be shared with this peer.
    runtime_.on_peer_ready(pid_);
  }
  if (!acks.empty()) {
    runtime_.on_acknowledged(pid_, acks);
  }
}

void NvSciPeer::enqueue_locked(
  uint32_t type, uint64_t id, uint64_t arg, std::vector<uint8_t> payload)
{
  MessageHeader header{type, 0, id, arg, payload.size()};
  std::vector<uint8_t> bytes;
  bytes.reserve(sizeof(header) + payload.size());
  append(bytes, &header, sizeof(header));
  append(bytes, payload.data(), payload.size());

  const size_t capacity = frame_size_ - sizeof(FrameHeader);
  for (size_t offset = 0; offset < bytes.size(); offset += capacity) {
    const auto used = static_cast<FrameHeader>(std::min(capacity, bytes.size() - offset));
    std::vector<uint8_t> frame(frame_size_, 0);
    std::memcpy(frame.data(), &used, sizeof(used));
    std::memcpy(frame.data() + sizeof(used), bytes.data() + offset, used);
    outbox_.push_back(std::move(frame));
  }
  flush_locked();
}

void NvSciPeer::flush_locked()
{
  if (!connected_ || lost_) {
    return;
  }
  while (!outbox_.empty()) {
    const std::vector<uint8_t> & frame = outbox_.front();
    uint32_t written = 0;
    const NvSciError err = NvSciIpcWriteSafe(
      endpoint_, frame.data(), static_cast<uint32_t>(frame.size()), &written);
    if (err == NvSciError_InsufficientMemory) {
      return;  // Queue full: the dispatcher retries on the next write event.
    }
    if (err != NvSciError_Success) {
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend", "NvSciIpc write to process %d failed: NvSciError %d",
        static_cast<int>(pid_), static_cast<int>(err));
      lost_ = true;
      return;
    }
    outbox_.pop_front();
  }
}

void NvSciPeer::read_locked(std::vector<Message> * inbound)
{
  std::vector<uint8_t> frame(frame_size_);
  for (;; ) {
    uint32_t received = 0;
    const NvSciError err = NvSciIpcReadSafe(
      endpoint_, frame.data(), static_cast<uint32_t>(frame.size()), &received);
    if (err != NvSciError_Success || received < sizeof(FrameHeader)) {
      break;
    }
    FrameHeader used = 0;
    std::memcpy(&used, frame.data(), sizeof(used));
    used = std::min<FrameHeader>(used, received - sizeof(FrameHeader));
    append(inbox_, frame.data() + sizeof(FrameHeader), used);
  }

  size_t consumed = 0;
  while (inbox_.size() - consumed >= sizeof(MessageHeader)) {
    MessageHeader header;
    std::memcpy(&header, inbox_.data() + consumed, sizeof(header));
    if (header.payload_size > kMaxPayload) {
      RCUTILS_LOG_ERROR_NAMED(
        "cuda_buffer_backend", "Corrupt NvSci message from process %d; resetting",
        static_cast<int>(pid_));
      lost_ = true;
      inbox_.clear();
      return;
    }
    if (inbox_.size() - consumed - sizeof(header) < header.payload_size) {
      break;
    }
    const uint8_t * payload = inbox_.data() + consumed + sizeof(header);
    inbound->push_back(
      {header.type, header.id, header.arg,
        std::vector<uint8_t>(payload, payload + header.payload_size)});
    consumed += sizeof(header) + header.payload_size;
  }
  inbox_.erase(inbox_.begin(), inbox_.begin() + static_cast<std::ptrdiff_t>(consumed));
}

void NvSciPeer::queue_hello_locked()
{
  NvSciBufAttrList buf_attrs = nullptr;
  NvSciSyncAttrList sync_attrs = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    buf_attrs = local_buf_waiter_attrs_;
    sync_attrs = local_sync_waiter_attrs_;
  }

  void * buf_blob = nullptr;
  size_t buf_blob_size = 0;
  void * sync_blob = nullptr;
  size_t sync_blob_size = 0;
  NvSciError err = NvSciBufAttrListIpcExportUnreconciled(
    &buf_attrs, 1, endpoint_, &buf_blob, &buf_blob_size);
  if (err == NvSciError_Success) {
    err = NvSciSyncAttrListIpcExportUnreconciled(
      &sync_attrs, 1, endpoint_, &sync_blob, &sync_blob_size);
  }
  if (err != NvSciError_Success) {
    if (buf_blob) {NvSciBufAttrListFreeDesc(buf_blob);}
    RCUTILS_LOG_ERROR_NAMED(
      "cuda_buffer_backend", "Cannot export NvSci waiter attributes to process %d: NvSciError %d",
      static_cast<int>(pid_), static_cast<int>(err));
    return;
  }

  std::vector<uint8_t> payload;
  append(payload, buf_blob, buf_blob_size);
  append(payload, sync_blob, sync_blob_size);
  NvSciBufAttrListFreeDesc(buf_blob);
  NvSciSyncAttrListFreeDesc(sync_blob);
  // id carries this side's connection id; arg the length of the first blob.
  enqueue_locked(kHello, connection_id_, buf_blob_size, std::move(payload));
}

void NvSciPeer::reject_locked(const Message & message)
{
  if (message.type != kBlock && message.type != kSyncChannel) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (message.type == kBlock) {
      failed_blocks_.insert(message.id);
    } else {
      failed_channels_.insert(static_cast<uint32_t>(message.id));
    }
  }
  state_cv_.notify_all();
  // arg carries the type of the object that was rejected.
  enqueue_locked(kNack, message.id, message.type, {});
}

bool NvSciPeer::handle(const Message & message)
{
  const uint8_t * payload = message.payload.data();
  const size_t payload_size = message.payload.size();

  switch (message.type) {
    case kHello: {
        {
          // Reconciliations use the peer's attribute lists outside this
          // peer's locks, so the lists must never change once set. A peer
          // that restarts gets a new connection, with a new hello; a second
          // one on the same connection is a protocol error. Drop the
          // connection and start over.
          std::lock_guard<std::mutex> lock(state_mutex_);
          if (peer_buf_waiter_attrs_ != nullptr || peer_sync_waiter_attrs_ != nullptr) {
            RCUTILS_LOG_ERROR_NAMED(
              "cuda_buffer_backend",
              "Process %d sent a second NvSci hello on one connection; resetting it",
              static_cast<int>(pid_));
            lost_ = true;  // io_mutex_ is held by the caller.
            return false;
          }
        }
        if (message.arg > payload_size) {
          throw CudaError("malformed NvSci hello");
        }
        NvSciBufAttrList buf_attrs = nullptr;
        NvSciSyncAttrList sync_attrs = nullptr;
        NVSCI_CHECK(
          NvSciBufAttrListIpcImportUnreconciled(
            runtime_.buf_module(), endpoint_, payload, message.arg, &buf_attrs));
        const NvSciError err = NvSciSyncAttrListIpcImportUnreconciled(
          runtime_.sync_module(), endpoint_, payload + message.arg,
          payload_size - message.arg, &sync_attrs);
        if (err != NvSciError_Success) {
          NvSciBufAttrListFree(buf_attrs);
          NVSCI_CHECK(err);
        }
        {
          // Only this (dispatcher) thread sets the lists, and they were
          // checked to be unset above. They are freed with the peer.
          std::lock_guard<std::mutex> lock(state_mutex_);
          peer_buf_waiter_attrs_ = buf_attrs;
          peer_sync_waiter_attrs_ = sync_attrs;
          peer_connection_id_ = message.id;
        }
        // Imports wait for the connection id.
        state_cv_.notify_all();
        return true;
      }

    case kSyncChannel: {
        NvSciSyncObjIpcExportDescriptor desc;
        if (payload_size < sizeof(desc)) {
          throw CudaError("malformed NvSci sync channel message");
        }
        std::memcpy(&desc, payload, sizeof(desc));
        NvSciSyncAttrList waiter = nullptr;
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          waiter = local_sync_waiter_attrs_;
        }
        ScopedSyncAttrs reconciled;
        NVSCI_CHECK(
          NvSciSyncAttrListIpcImportReconciled(
            runtime_.sync_module(), endpoint_, payload + sizeof(desc),
            payload_size - sizeof(desc), &waiter, 1, &reconciled.list));
        NvSciSyncObj obj = nullptr;
        NVSCI_CHECK(
          NvSciSyncObjIpcImport(
            endpoint_, &desc, reconciled.list, NvSciSyncAccessPerm_WaitOnly,
            kImportTimeoutUs, &obj));
        const auto channel_id = static_cast<uint32_t>(message.id);
        auto channel = std::make_shared<NvSciSyncChannel>(channel_id, obj);
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          imported_channels_[channel_id] = std::move(channel);
        }
        state_cv_.notify_all();
        break;
      }

    case kBlock: {
        NvSciBufObjIpcExportDescriptor desc;
        if (payload_size < sizeof(desc)) {
          throw CudaError("malformed NvSci block message");
        }
        std::memcpy(&desc, payload, sizeof(desc));
        NvSciBufAttrList waiter = nullptr;
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          waiter = local_buf_waiter_attrs_;
        }
        ScopedBufAttrs reconciled;
        NVSCI_CHECK(
          NvSciBufAttrListIpcImportReconciled(
            runtime_.buf_module(), endpoint_, payload + sizeof(desc),
            payload_size - sizeof(desc), &waiter, 1, &reconciled.list));
        NvSciBufObj obj = nullptr;
        NVSCI_CHECK(
          NvSciBufObjIpcImport(
            endpoint_, &desc, reconciled.list, NvSciBufAccessPerm_Readonly,
            kImportTimeoutUs, &obj));
        // arg carries the block size in bytes.
        auto mapping = std::make_unique<NvSciMapping>(obj, message.arg);

        const std::string lease_name = NvSciBlock::lease_name(pid_, message.id);
        const int fd = shm_open(lease_name.c_str(), O_RDWR, 0);
        if (fd < 0) {
          throw CudaError("cannot open NvSci lease " + lease_name);
        }
        void * lease = mmap(
          nullptr, sizeof(IPCMetadata), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (lease == MAP_FAILED) {
          throw CudaError("cannot map NvSci lease " + lease_name);
        }
        auto block = std::make_shared<NvSciImportedBlock>(
          std::move(mapping), static_cast<IPCMetadata *>(lease));
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          imported_blocks_[message.id] = std::move(block);
        }
        state_cv_.notify_all();
        break;
      }

    case kNack:
      if (message.arg == kBlock) {
        rejected_blocks_.insert(message.id);
      } else if (message.arg == kSyncChannel) {
        rejected_channels_.insert(static_cast<uint32_t>(message.id));
      }
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend",
        "Process %d cannot import an NvSci object of this process; it gets the CPU path",
        static_cast<int>(pid_));
      break;

    case kRetire: {
        // Buffers still being read keep the mapping alive through their own
        // reference; this only drops the cache's.
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          imported_blocks_.erase(message.id);
          retired_blocks_.insert(message.id);
        }
        // A descriptor still queued for a retired block must not wait for it.
        state_cv_.notify_all();
        break;
      }

    default:
      throw CudaError("unknown NvSci message type " + std::to_string(message.type));
  }
  return false;
}

void NvSciPeer::append_waiter_attrs(
  std::vector<NvSciBufAttrList> * buf_lists,
  std::vector<NvSciSyncAttrList> * sync_lists) const
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (buf_lists && peer_buf_waiter_attrs_) {
    buf_lists->push_back(peer_buf_waiter_attrs_);
  }
  if (sync_lists && peer_sync_waiter_attrs_) {
    sync_lists->push_back(peer_sync_waiter_attrs_);
  }
}

void NvSciPeer::export_channel(const NvSciSyncChannel & channel)
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  if (exported_channels_.count(channel.id) != 0) {
    return;
  }

  NvSciSyncAttrList reconciled = nullptr;  // owned by the sync object
  NVSCI_CHECK(NvSciSyncObjGetAttrList(channel.obj, &reconciled));
  void * blob = nullptr;
  size_t blob_size = 0;
  NVSCI_CHECK(NvSciSyncAttrListIpcExportReconciled(reconciled, endpoint_, &blob, &blob_size));
  NvSciSyncObjIpcExportDescriptor desc;
  const NvSciError err = NvSciSyncObjIpcExport(
    channel.obj, NvSciSyncAccessPerm_WaitOnly, endpoint_, &desc);
  if (err != NvSciError_Success) {
    NvSciSyncAttrListFreeDesc(blob);
    NVSCI_CHECK(err);
  }

  std::vector<uint8_t> payload;
  append(payload, &desc, sizeof(desc));
  append(payload, blob, blob_size);
  NvSciSyncAttrListFreeDesc(blob);

  // Queued, not sent: an export cannot be taken back, so it must not be
  // possible for the message carrying it to be dropped.
  enqueue_locked(kSyncChannel, channel.id, 0, std::move(payload));
  exported_channels_.insert(channel.id);
}

void NvSciPeer::export_block(const NvSciBlock & block)
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  if (exported_blocks_.count(block.id) != 0) {
    return;
  }

  NvSciBufAttrList reconciled = nullptr;  // owned by the buffer object
  NVSCI_CHECK(NvSciBufObjGetAttrList(block.mapping->obj, &reconciled));
  void * blob = nullptr;
  size_t blob_size = 0;
  NVSCI_CHECK(NvSciBufAttrListIpcExportReconciled(reconciled, endpoint_, &blob, &blob_size));
  NvSciBufObjIpcExportDescriptor desc;
  const NvSciError err = NvSciBufObjIpcExport(
    block.mapping->obj, NvSciBufAccessPerm_Readonly, endpoint_, &desc);
  if (err != NvSciError_Success) {
    NvSciBufAttrListFreeDesc(blob);
    NVSCI_CHECK(err);
  }

  std::vector<uint8_t> payload;
  append(payload, &desc, sizeof(desc));
  append(payload, blob, blob_size);
  NvSciBufAttrListFreeDesc(blob);

  enqueue_locked(kBlock, block.id, block.mapping->size, std::move(payload));
  exported_blocks_.insert(block.id);
}

void NvSciPeer::export_fence(const NvSciSyncFence & fence, NvSciSyncFenceIpcExportDescriptor * out)
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  NVSCI_CHECK(NvSciSyncIpcExportFence(&fence, endpoint_, out));
}

bool NvSciPeer::accepts_block(uint64_t block_id) const
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  return rejected_blocks_.count(block_id) == 0;
}

bool NvSciPeer::accepts_channel(uint32_t channel_id) const
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  return rejected_channels_.count(channel_id) == 0;
}

void NvSciPeer::retire_block(uint64_t block_id)
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  rejected_blocks_.erase(block_id);
  if (exported_blocks_.erase(block_id) != 0) {
    enqueue_locked(kRetire, block_id, 0, {});
  }
}

NvSciPeer::WaitResult NvSciPeer::wait_objects(
  uint64_t connection, uint64_t block_id, bool need_channel, uint32_t channel_id,
  std::chrono::milliseconds timeout,
  std::shared_ptr<NvSciImportedBlock> * block,
  std::shared_ptr<NvSciSyncChannel> * channel)
{
  std::unique_lock<std::mutex> lock(state_mutex_);
  WaitResult result = WaitResult::kTimedOut;
  state_cv_.wait_for(
    lock, timeout, [&] {
      if (shut_down_) {
        result = WaitResult::kGone;
        return true;
      }
      // Before the peer's hello it is not known which connection this is.
      if (peer_connection_id_ == 0) {
        return false;
      }
      if (peer_connection_id_ != connection) {
        result = WaitResult::kWrongConnection;
        return true;
      }
      if (failed_blocks_.count(block_id) != 0 || retired_blocks_.count(block_id) != 0 ||
      (need_channel && failed_channels_.count(channel_id) != 0))
      {
        result = WaitResult::kGone;
        return true;
      }
      auto b = imported_blocks_.find(block_id);
      if (b == imported_blocks_.end()) {
        return false;
      }
      std::shared_ptr<NvSciSyncChannel> c;
      if (need_channel) {
        auto it = imported_channels_.find(channel_id);
        if (it == imported_channels_.end()) {
          return false;
        }
        c = it->second;
      }
      *block = b->second;
      *channel = std::move(c);
      result = WaitResult::kReady;
      return true;
    });
  return result;
}

}  // namespace cuda_buffer_backend
