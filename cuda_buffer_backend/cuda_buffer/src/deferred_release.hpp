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

// Private to the cuda_buffer library.

#ifndef DEFERRED_RELEASE_HPP_
#define DEFERRED_RELEASE_HPP_

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace cuda_buffer_backend
{
namespace detail
{

/// \brief Run \p release once every event in \p events has completed.
/// \details Destroys the events. With no events, \p release runs before this
/// returns; otherwise it runs on a process-wide background thread, so the
/// caller (typically a destructor on the publish or callback path) does not
/// block on stream work. \p release must not throw.
void release_after(std::vector<cudaEvent_t> events, std::function<void()> release);

}  // namespace detail
}  // namespace cuda_buffer_backend

#endif  // DEFERRED_RELEASE_HPP_
