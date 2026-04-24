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

#ifndef TORCH_BUFFER__TORCH_BUFFER_UTILS_HPP_
#define TORCH_BUFFER__TORCH_BUFFER_UTILS_HPP_

#include <torch/torch.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace torch_buffer_backend
{

template<typename T>
constexpr at::ScalarType get_torch_scalar_type()
{
  if constexpr (std::is_same_v<T, uint8_t>) {
    return at::kByte;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return at::kChar;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return at::kShort;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return at::kInt;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return at::kLong;
  } else if constexpr (std::is_same_v<T, float>) {
    return at::kFloat;
  } else if constexpr (std::is_same_v<T, double>) {
    return at::kDouble;
  } else if constexpr (std::is_same_v<T, bool>) {
    return at::kBool;
  } else if constexpr (std::is_same_v<T, c10::Half>) {
    return at::kHalf;
  } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
    return at::kBFloat16;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported element type for Torch conversion.");
    return at::kByte;
  }
}

inline std::string scalar_type_to_string(at::ScalarType type)
{
  return std::string(at::toString(type));
}

inline at::ScalarType string_to_scalar_type(const std::string & name)
{
  if (name == "Byte" || name == "uint8") {return at::kByte;}
  if (name == "Char" || name == "int8") {return at::kChar;}
  if (name == "Short" || name == "int16") {return at::kShort;}
  if (name == "Int" || name == "int32") {return at::kInt;}
  if (name == "Long" || name == "int64") {return at::kLong;}
  if (name == "Float" || name == "float32") {return at::kFloat;}
  if (name == "Double" || name == "float64") {return at::kDouble;}
  if (name == "Bool" || name == "bool") {return at::kBool;}
  if (name == "Half" || name == "float16") {return at::kHalf;}
  if (name == "BFloat16" || name == "bfloat16") {return at::kBFloat16;}
  throw std::runtime_error("Unknown dtype string: " + name);
}

inline size_t scalar_type_size(at::ScalarType type)
{
  switch (type) {
    case at::kByte:
    case at::kChar:
    case at::kBool:
      return 1;
    case at::kShort:
    case at::kHalf:
    case at::kBFloat16:
      return 2;
    case at::kInt:
    case at::kFloat:
      return 4;
    case at::kLong:
    case at::kDouble:
      return 8;
    default:
      throw std::runtime_error("Unknown ScalarType for size calculation");
  }
}

/// \brief Minimum number of bytes a tensor of (shape, strides, element_size)
/// must alias. Strides, if provided, are in elements and must be non-negative
/// (buffers start at offset 0). Throws on negative shape/stride, rank
/// mismatch, or any arithmetic overflow.
inline size_t required_byte_size(
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  size_t element_size)
{
  for (auto s : shape) {
    if (s < 0) {
      throw std::runtime_error(
        "torch_buffer: negative shape dimension (" + std::to_string(s) + ")");
    }
    if (s == 0) {
      return 0;
    }
  }

  auto mul_checked = [](size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
      throw std::runtime_error("torch_buffer: size overflow");
    }
    return a * b;
  };
  auto add_checked = [](size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
      throw std::runtime_error("torch_buffer: size overflow");
    }
    return a + b;
  };

  size_t elems = 1;
  if (strides.empty()) {
    for (auto s : shape) {
      elems = mul_checked(elems, static_cast<size_t>(s));
    }
  } else {
    if (strides.size() != shape.size()) {
      throw std::runtime_error(
        "torch_buffer: shape/strides rank mismatch (" +
        std::to_string(shape.size()) + " vs " +
        std::to_string(strides.size()) + ")");
    }
    size_t max_off = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (strides[i] < 0) {
        throw std::runtime_error(
          "torch_buffer: negative stride not supported (dim " +
          std::to_string(i) + ")");
      }
      size_t span = static_cast<size_t>(shape[i] - 1);
      size_t term = mul_checked(span, static_cast<size_t>(strides[i]));
      max_off = add_checked(max_off, term);
    }
    elems = add_checked(max_off, 1);
  }

  return mul_checked(elems, element_size);
}
}  // namespace torch_buffer_backend

#endif  // TORCH_BUFFER__TORCH_BUFFER_UTILS_HPP_
