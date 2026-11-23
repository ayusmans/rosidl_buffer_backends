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

#include "deferred_release.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace cuda_buffer_backend
{
namespace detail
{

namespace
{

/// Process-lifetime CUDA cleanup service. Intentionally leaked so its worker
/// thread is not torn down during static destruction, when the CUDA runtime
/// and driver may already be unloading.
class Recycler
{
public:
  static Recycler & get()
  {
    static Recycler * const recycler = new Recycler();
    return *recycler;
  }

  void enqueue(std::vector<cudaEvent_t> events, std::function<void()> release)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back({std::move(events), std::move(release)});
    cv_.notify_one();
  }

private:
  struct PendingWork
  {
    std::vector<cudaEvent_t> events;
    std::function<void()> release;
  };

  Recycler()
  {
    std::thread(&Recycler::run, this).detach();
  }

  void run()
  {
    for (;; ) {
      PendingWork work;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {return !queue_.empty();});
        work = std::move(queue_.front());
        queue_.pop_front();
      }
      for (cudaEvent_t ev : work.events) {
        if (ev) {
          cudaEventSynchronize(ev);
          cudaEventDestroy(ev);
          (void)cudaGetLastError();
        }
      }
      if (work.release) {
        work.release();
      }
    }
  }

  std::deque<PendingWork> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace

void release_after(std::vector<cudaEvent_t> events, std::function<void()> release)
{
  if (events.empty()) {
    if (release) {
      release();
    }
    return;
  }
  Recycler::get().enqueue(std::move(events), std::move(release));
}

}  // namespace detail
}  // namespace cuda_buffer_backend
