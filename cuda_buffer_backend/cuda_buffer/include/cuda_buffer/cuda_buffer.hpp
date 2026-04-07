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

#ifndef CUDA_BUFFER__CUDA_BUFFER_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_HPP_

#include <cstddef>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <rcutils/logging_macros.h>

#include "cuda_buffer/cuda_buffer_handle.hpp"
#include "cuda_buffer/cuda_error.hpp"

namespace cuda_buffer_backend
{

/// \brief Background thread that synchronizes CUDA events and runs deleters off the critical path.
class BufferRecycler
{
public:
  static BufferRecycler & instance()
  {
    static BufferRecycler recycler;
    return recycler;
  }

  void enqueue(
    std::vector<cudaEvent_t> events,
    std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> ptr)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back({std::move(events), std::move(ptr)});
    cv_.notify_one();
  }

private:
  struct PendingWork
  {
    std::vector<cudaEvent_t> events;
    std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr;
  };

  BufferRecycler()
  {
    thread_ = std::thread(&BufferRecycler::run, this);
  }

  ~BufferRecycler()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void run()
  {
    while (true) {
      PendingWork work;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {return !queue_.empty() || !running_;});
        if (queue_.empty() && !running_) {break;}
        if (queue_.empty()) {continue;}
        work = std::move(queue_.front());
        queue_.erase(queue_.begin());
      }
      for (cudaEvent_t ev : work.events) {
        if (ev) {
          cudaEventSynchronize(ev);
          cudaEventDestroy(ev);
          (void)cudaGetLastError();
        }
      }
      work.device_ptr.reset();
    }
  }

  std::vector<PendingWork> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  bool running_{true};
};

/// \brief GPU memory buffer with event-based synchronization via ReadHandle/WriteHandle.
class CudaBuffer
{
public:
  CudaBuffer() = default;

  CudaBuffer(CudaBuffer && other) noexcept
  : device_ptr_(std::move(other.device_ptr_)),
    size_(other.size_),
    write_event_(other.write_event_),
    owns_write_event_(other.owns_write_event_),
    read_events_(std::move(other.read_events_)),
    handle_state_(std::move(other.handle_state_))
  {
    other.size_ = 0;
    other.write_event_ = nullptr;
    other.owns_write_event_ = false;
  }

  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer & operator=(const CudaBuffer &) = delete;

  CudaBuffer & operator=(CudaBuffer && other) noexcept
  {
    if (this != &other) {
      std::swap(device_ptr_, other.device_ptr_);
      std::swap(size_, other.size_);
      std::swap(write_event_, other.write_event_);
      std::swap(owns_write_event_, other.owns_write_event_);
      std::swap(read_events_, other.read_events_);
      std::swap(handle_state_, other.handle_state_);
    }
    return *this;
  }

  ~CudaBuffer()
  {
    if (owns_write_event_ && write_event_) {
      cudaError_t e = cudaEventDestroy(write_event_);
      if (!cuda_error_is_safe(e)) {
        RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
          "~CudaBuffer: cudaEventDestroy failed: %s", cudaGetErrorName(e));
      }
      (void)cudaGetLastError();
    }
    if (!read_events_.empty()) {
      BufferRecycler::instance().enqueue(
        std::move(read_events_), std::move(device_ptr_));
    }
  }

  CudaBuffer(void * ptr, size_t size, std::function<void(uint8_t *)> custom_deleter)
  : device_ptr_(static_cast<uint8_t *>(ptr), std::move(custom_deleter)), size_(size)
  {
  }

  ReadHandle get_read_handle(cudaStream_t stream) const
  {
    if (handle_state_) {
      std::lock_guard<std::mutex> lk(handle_state_->mtx);
      if (handle_state_->state == HandleState::State::InUse) {
        finalize_write_handle_locked();
      }
    }

    return ReadHandle(
      device_ptr_.get(), write_event_, &read_events_, &events_mutex_, stream);
  }

  WriteHandle get_write_handle(cudaStream_t stream)
  {
    std::lock_guard<std::mutex> lg(events_mutex_);

    if (handle_state_) {
      std::lock_guard<std::mutex> lk(handle_state_->mtx);
      if (handle_state_->state == HandleState::State::InUse) {
        throw CudaError("CudaBuffer: write handle already in use; cannot acquire second handle");
      }
      if (handle_state_->state == HandleState::State::Finalized) {
        throw CudaError("CudaBuffer: write already finalized; cannot acquire write handle");
      }
    }

    if (!read_events_.empty()) {
      throw CudaError("CudaBuffer: read events exist; cannot re-acquire write handle");
    }

    if (!handle_state_) {handle_state_ = std::make_shared<HandleState>();}
    handle_state_->state = HandleState::State::InUse;
    handle_state_->write_stream = stream;
    return WriteHandle(device_ptr_.get(), &write_event_, stream, handle_state_);
  }

  void set_write_event(cudaEvent_t event, bool owns_event = false)
  {
    write_event_ = event;
    owns_write_event_ = owns_event;
  }

  cudaEvent_t get_write_event() const {return write_event_;}

  void finalize_write_handle() const
  {
    if (handle_state_ == nullptr) {return;}
    std::lock_guard<std::mutex> lock(handle_state_->mtx);
    finalize_write_handle_locked();
  }

  size_t size() const {return size_;}
  uint8_t * get_device_ptr() {return device_ptr_.get();}
  const uint8_t * get_device_ptr() const {return device_ptr_.get();}

private:
  void finalize_write_handle_locked() const
  {
    if (handle_state_->state != HandleState::State::InUse) {return;}
    if (cuda_is_stream_usable(handle_state_->write_stream) && write_event_) {
      if (cudaEventRecord(write_event_, handle_state_->write_stream) != cudaSuccess) {
        (void)cudaGetLastError();
      }
    }
    handle_state_->state = HandleState::State::Finalized;
    handle_state_->write_stream = nullptr;
  }

  static void default_cuda_free(uint8_t * p)
  {
    if (p) {
      cudaError_t e = cudaFree(p);
      if (!cuda_error_is_safe(e)) {
        RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
          "cudaFree failed: %s", cudaGetErrorName(e));
      }
      (void)cudaGetLastError();
    }
  }

  std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr_{
    nullptr, default_cuda_free
  };
  size_t size_{0};

  mutable cudaEvent_t write_event_{nullptr};
  bool owns_write_event_{false};
  mutable std::vector<cudaEvent_t> read_events_;
  mutable std::mutex events_mutex_;

  std::shared_ptr<HandleState> handle_state_{nullptr};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_HPP_
