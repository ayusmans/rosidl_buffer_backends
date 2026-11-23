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

#ifndef PATTERN_HPP_
#define PATTERN_HPP_

#include <cstdint>

namespace cuda_buffer_examples
{

/// \brief The byte every element of message \p sequence is filled with.
/// \details Changes on every message, so a listener that reads a buffer the
/// talker has already reused sees the wrong value.
inline uint8_t pattern_for(uint64_t sequence)
{
  return static_cast<uint8_t>((sequence * 37u + 11u) & 0xffu);
}

}  // namespace cuda_buffer_examples

#endif  // PATTERN_HPP_
