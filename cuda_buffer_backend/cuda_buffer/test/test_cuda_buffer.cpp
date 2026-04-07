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
#include <cuda_runtime.h>

#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"

class CudaBufferAccessTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    cudaStreamCreate(&stream1_);
    cudaStreamCreate(&stream2_);
  }

  void TearDown() override
  {
    cudaStreamDestroy(stream1_);
    cudaStreamDestroy(stream2_);
  }

  void allocate_buffer(rosidl::Buffer<uint8_t> & buffer, size_t count)
  {
    auto impl = std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(count);
    buffer = rosidl::Buffer<uint8_t>(std::move(impl));
  }

  void write_pattern(uint8_t * device_ptr, size_t count, uint8_t offset, cudaStream_t stream)
  {
    std::vector<uint8_t> host(count);
    for (size_t i = 0; i < count; ++i) {
      host[i] = static_cast<uint8_t>((offset + i) % 256);
    }
    cudaMemcpyAsync(device_ptr, host.data(), count, cudaMemcpyHostToDevice, stream);
  }

  std::vector<uint8_t> read_to_host(const uint8_t * device_ptr, size_t count, cudaStream_t stream)
  {
    std::vector<uint8_t> host(count);
    cudaMemcpyAsync(host.data(), device_ptr, count, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return host;
  }

  cudaStream_t stream1_{nullptr};
  cudaStream_t stream2_{nullptr};
};

TEST_F(CudaBufferAccessTest, AllocateAndWriteHandle)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 1024);

  cuda_buffer_backend::WriteHandle handle =
    cuda_buffer_backend::from_buffer(buffer, stream1_);

  EXPECT_NE(nullptr, handle.get_ptr());
  EXPECT_EQ("cuda", buffer.get_backend_type());
  EXPECT_EQ(1024u, buffer.size());
}

TEST_F(CudaBufferAccessTest, FromBuffer_ThrowsOnCpuBuffer)
{
  rosidl::Buffer<uint8_t> buffer(64);
  const rosidl::Buffer<uint8_t> & cbuf = buffer;

  EXPECT_THROW(
    cuda_buffer_backend::from_buffer(cbuf, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferAccessTest, FromBuffer_ThrowsOnEmptyBuffer)
{
  rosidl::Buffer<uint8_t> buffer;
  const rosidl::Buffer<uint8_t> & cbuf = buffer;

  EXPECT_THROW(
    cuda_buffer_backend::from_buffer(cbuf, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferAccessTest, ToBuffer_CopiesFromDevicePointer)
{
  constexpr size_t N = 512;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  uint8_t * src_ptr = nullptr;
  cudaMalloc(&src_ptr, N);
  std::vector<uint8_t> host_src(N);
  for (size_t i = 0; i < N; ++i) {
    host_src[i] = static_cast<uint8_t>((10 + i) % 256);
  }
  cudaMemcpyAsync(src_ptr, host_src.data(), N, cudaMemcpyHostToDevice, stream1_);
  cudaStreamSynchronize(stream1_);

  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_buffer(buffer, stream1_);
  cuda_buffer_backend::to_buffer(src_ptr, N, wh, stream1_);
  cudaFree(src_ptr);

  std::vector<uint8_t> result = read_to_host(wh.get_ptr(), N, stream1_);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((10 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

TEST_F(CudaBufferAccessTest, ToBuffer_CopiesFromHostPointer)
{
  constexpr size_t N = 256;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  std::vector<uint8_t> host_src(N);
  for (size_t i = 0; i < N; ++i) {
    host_src[i] = static_cast<uint8_t>((50 + i) % 256);
  }

  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_buffer(buffer, stream1_);
  cuda_buffer_backend::to_buffer(
    host_src.data(), N, wh, stream1_, cudaMemcpyHostToDevice);

  std::vector<uint8_t> result = read_to_host(wh.get_ptr(), N, stream1_);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((50 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

TEST_F(CudaBufferAccessTest, EventSync_WriteOnStream1_ReadOnStream2)
{
  constexpr size_t N = 1024;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 42, stream1_);
  }

  std::vector<uint8_t> result;
  {
    const rosidl::Buffer<uint8_t> & cbuf = buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_buffer(cbuf, stream2_);
    result = read_to_host(rh.get_ptr(), N, stream2_);
  }

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((42 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

TEST_F(CudaBufferAccessTest, DoubleWriteHandle_Throws)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_buffer(buffer, stream1_);

  EXPECT_THROW(
    cuda_buffer_backend::from_buffer(buffer, stream2_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferAccessTest, WriteAfterFinalized_Throws)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_buffer(buffer, stream1_);
  }

  EXPECT_THROW(
    cuda_buffer_backend::from_buffer(buffer, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferAccessTest, ReadAfterReadEvents_BlocksWrite)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_buffer(buffer, stream1_);
  }

  {
    const rosidl::Buffer<uint8_t> & cbuf = buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_buffer(cbuf, stream2_);
  }

  EXPECT_THROW(
    cuda_buffer_backend::from_buffer(buffer, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferAccessTest, GpuPipeline_NoIntermediateCpuSync)
{
  constexpr size_t N = 4096;
  rosidl::Buffer<uint8_t> src_buffer;
  rosidl::Buffer<uint8_t> dst_buffer;
  allocate_buffer(src_buffer, N);
  allocate_buffer(dst_buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_buffer(src_buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 99, stream1_);
  }

  {
    const rosidl::Buffer<uint8_t> & cbuf = src_buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_buffer(cbuf, stream2_);
    cuda_buffer_backend::WriteHandle wh2 =
      cuda_buffer_backend::from_buffer(dst_buffer, stream2_);
    cudaMemcpyAsync(
      wh2.get_ptr(), rh.get_ptr(), N, cudaMemcpyDeviceToDevice, stream2_);
  }

  std::vector<uint8_t> result;
  {
    const rosidl::Buffer<uint8_t> & cbuf = dst_buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_buffer(cbuf, stream1_);
    result = read_to_host(rh.get_ptr(), N, stream1_);
  }

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((99 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
