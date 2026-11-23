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
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"

#ifdef CUDA_BUFFER_ENABLE_NVSCI
#include "cuda_buffer/nvsci_buffer.hpp"
#include "cuda_buffer/nvsci_runtime.hpp"
#endif

using cuda_buffer_backend::CudaBufferImpl;
using cuda_buffer_backend::StorageKind;

namespace
{

constexpr const char * kPrintDefaultFlag = "--print-default-storage";
const char * g_self = "";

bool has_device()
{
  int count = 0;
  const bool ok = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  (void)cudaGetLastError();
  return ok;
}

bool nvsci_available()
{
  return cuda_buffer_backend::is_storage_available(StorageKind::NvSci);
}

/// Run this binary with CUDA_BUFFER_STORAGE=\p value and return the default it reports.
std::string default_with_env(const char * value)
{
  const std::string command =
    "CUDA_BUFFER_STORAGE='" + std::string(value) + "' '" + std::string(g_self) + "' " +
    kPrintDefaultFlag;
  FILE * pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  std::array<char, 64> line{};
  std::string out;
  while (std::fgets(line.data(), line.size(), pipe) != nullptr) {
    out = line.data();
  }
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

void fill(CudaBufferImpl<uint8_t> & impl, uint8_t value, cudaStream_t stream)
{
  auto wh = impl.get_cuda_buffer().get_write_handle(stream);
  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(wh.get_ptr(), value, impl.size(), stream));
}

std::vector<uint8_t> read_back(const cuda_buffer_backend::CudaBuffer & buffer, cudaStream_t stream)
{
  std::vector<uint8_t> host(buffer.size());
  auto rh = buffer.get_read_handle(stream);
  EXPECT_EQ(
    cudaSuccess,
    cudaMemcpyAsync(host.data(), rh.get_ptr(), host.size(), cudaMemcpyDeviceToHost, stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  return host;
}

bool all_equal(const std::vector<uint8_t> & data, uint8_t value)
{
  for (uint8_t b : data) {
    if (b != value) {
      return false;
    }
  }
  return !data.empty();
}

class StorageTest : public ::testing::TestWithParam<StorageKind>
{
protected:
  void SetUp() override
  {
    if (!has_device()) {
      GTEST_SKIP() << "needs a CUDA device";
    }
    if (!cuda_buffer_backend::is_storage_available(GetParam())) {
      GTEST_SKIP() << cuda_buffer_backend::to_string(GetParam()) << " is not available";
    }
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
  }

  void TearDown() override
  {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t stream_{nullptr};
};

}  // namespace

// -- Names and the default --

TEST(StorageSelection, NamesRoundTrip)
{
  for (StorageKind kind : {StorageKind::Vmm, StorageKind::NvSci}) {
    auto parsed = cuda_buffer_backend::parse_storage_kind(cuda_buffer_backend::to_string(kind));
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(kind, *parsed);
  }
  EXPECT_FALSE(cuda_buffer_backend::parse_storage_kind("VMM").has_value());
  EXPECT_FALSE(cuda_buffer_backend::parse_storage_kind("").has_value());
  EXPECT_EQ(
    1u << static_cast<int>(StorageKind::NvSci),
      cuda_buffer_backend::storage_kind_bit(StorageKind::NvSci));
}

TEST(StorageSelection, DefaultIsVmm)
{
  EXPECT_EQ(StorageKind::Vmm, cuda_buffer_backend::kDefaultStorageKind);
  EXPECT_EQ("vmm", default_with_env(""));
}

TEST(StorageSelection, EnvironmentSelectsDefault)
{
  EXPECT_EQ("vmm", default_with_env("vmm"));
  // An unknown name falls back to the default instead of failing.
  EXPECT_EQ("vmm", default_with_env("bogus"));
  // An unavailable backend falls back too.
  EXPECT_EQ(nvsci_available() ? "nvsci" : "vmm", default_with_env("nvsci"));
}

TEST(StorageSelection, SetDefaultRejectsUnavailable)
{
  if (!has_device()) {
    GTEST_SKIP() << "needs a CUDA device";
  }
  if (nvsci_available()) {
    cuda_buffer_backend::set_default_storage_kind(StorageKind::NvSci);
    EXPECT_EQ(StorageKind::NvSci, cuda_buffer_backend::default_storage_kind());
    auto buffer = cuda_buffer_backend::allocate_buffer(256);
    EXPECT_EQ(StorageKind::NvSci, cuda_buffer_backend::storage_kind_of(buffer));
  } else {
    EXPECT_THROW(
      cuda_buffer_backend::set_default_storage_kind(StorageKind::NvSci),
      cuda_buffer_backend::CudaError);
  }
  cuda_buffer_backend::set_default_storage_kind(StorageKind::Vmm);
  EXPECT_EQ(StorageKind::Vmm, cuda_buffer_backend::default_storage_kind());
}

TEST(StorageSelection, CpuBufferHasNoStorageKind)
{
  rosidl::Buffer<uint8_t> cpu(std::vector<uint8_t>(16, 1));
  EXPECT_FALSE(cuda_buffer_backend::storage_kind_of(cpu).has_value());
}

// -- Behaviour that every backend shares --

TEST_P(StorageTest, AllocatesRequestedKind)
{
  auto buffer = cuda_buffer_backend::allocate_buffer(4096, GetParam());
  ASSERT_TRUE(cuda_buffer_backend::storage_kind_of(buffer).has_value());
  EXPECT_EQ(GetParam(), *cuda_buffer_backend::storage_kind_of(buffer));
}

TEST_P(StorageTest, WriteThenReadOnAnotherStream)
{
  CudaBufferImpl<uint8_t> impl(1 << 20, GetParam());
  fill(impl, 0x5a, stream_);
  cudaStream_t other = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&other));
  EXPECT_TRUE(all_equal(read_back(impl.get_cuda_buffer(), other), 0x5a));
  cudaStreamDestroy(other);
}

TEST_P(StorageTest, ResizeClearAndCloneKeepTheKind)
{
  CudaBufferImpl<uint8_t> impl(1024, GetParam());
  fill(impl, 0x11, stream_);
  impl.set_stream(stream_);

  impl.resize(4096);
  EXPECT_EQ(GetParam(), impl.get_cuda_buffer().storage_kind());

  // An empty buffer has no storage, but the impl remembers what it was asked for.
  impl.clear();
  EXPECT_EQ(GetParam(), impl.storage_kind());
  impl.resize(2048);
  EXPECT_EQ(GetParam(), impl.get_cuda_buffer().storage_kind());

  fill(impl, 0x22, stream_);
  auto copy = impl.clone();
  auto * cuda_copy = dynamic_cast<CudaBufferImpl<uint8_t> *>(copy.get());
  ASSERT_NE(nullptr, cuda_copy);
  EXPECT_EQ(GetParam(), cuda_copy->get_cuda_buffer().storage_kind());
  EXPECT_TRUE(all_equal(read_back(cuda_copy->get_cuda_buffer(), stream_), 0x22));
}

TEST_P(StorageTest, PromotesCpuBufferToRequestedKind)
{
  rosidl::Buffer<uint8_t> cpu(std::vector<uint8_t>(512, 7));
  auto rh = cuda_buffer_backend::from_input_buffer(cpu, stream_, GetParam());
  ASSERT_NE(nullptr, rh.get_promoted_buffer());
  EXPECT_EQ(GetParam(), cuda_buffer_backend::storage_kind_of(*rh.get_promoted_buffer()));
}

INSTANTIATE_TEST_SUITE_P(
  Backends, StorageTest, ::testing::Values(StorageKind::Vmm, StorageKind::NvSci),
  [](const ::testing::TestParamInfo<StorageKind> & info) {
    return std::string(cuda_buffer_backend::to_string(info.param));
  });

// -- Both backends in one process --

TEST(MixedStorage, CopyBetweenBackends)
{
  if (!has_device() || !nvsci_available()) {
    GTEST_SKIP() << "needs a CUDA device with NvSci";
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  {
    CudaBufferImpl<uint8_t> vmm(1 << 16, StorageKind::Vmm);
    CudaBufferImpl<uint8_t> nvsci(1 << 16, StorageKind::NvSci);
    fill(vmm, 0x3c, stream);
    {
      auto rh = vmm.get_cuda_buffer().get_read_handle(stream);
      auto wh = nvsci.get_cuda_buffer().get_write_handle(stream);
      ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(wh.get_ptr(), rh.get_ptr(), vmm.size(), cudaMemcpyDeviceToDevice, stream));
    }
    EXPECT_EQ(StorageKind::Vmm, vmm.get_cuda_buffer().storage_kind());
    EXPECT_EQ(StorageKind::NvSci, nvsci.get_cuda_buffer().storage_kind());
    EXPECT_TRUE(all_equal(read_back(nvsci.get_cuda_buffer(), stream), 0x3c));
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

#ifdef CUDA_BUFFER_ENABLE_NVSCI

// -- NvSci sharing within a process --

namespace
{

class NvSciShareTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!has_device() || !nvsci_available()) {
      GTEST_SKIP() << "needs a CUDA device with NvSci";
    }
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
  }

  void TearDown() override
  {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t stream_{nullptr};
};

/// A subscriber endpoint in this process.
cuda_buffer_backend::NvSciSubscriber local_subscriber(const std::string & key = "sub")
{
  cuda_buffer_backend::NvSciSubscriber subscriber;
  subscriber.pid = getpid();
  subscriber.key = key;
  return subscriber;
}

}  // namespace

TEST_F(NvSciShareTest, ImportInSameProcessSeesTheWrite)
{
  auto & runtime = cuda_buffer_backend::NvSciRuntime::get();
  CudaBufferImpl<uint8_t> impl(1 << 20, StorageKind::NvSci);
  fill(impl, 0x77, stream_);

  const auto & producer = static_cast<const cuda_buffer_backend::NvSciBuffer &>(
    impl.get_cuda_buffer());
  cuda_buffer_backend::NvSciExport desc;
  ASSERT_TRUE(runtime.export_buffer(producer, local_subscriber(), &desc));
  EXPECT_EQ(getpid(), desc.pid);
  EXPECT_NE(0u, desc.uid);
  EXPECT_EQ(0u, desc.connection);
  EXPECT_NE(0u, desc.stream);
  EXPECT_NE(0u, desc.seq);

  auto imported = runtime.import_buffer(desc, impl.size());
  ASSERT_NE(nullptr, imported);
  EXPECT_FALSE(imported->is_writable());
  EXPECT_THROW(imported->get_write_handle(stream_), cuda_buffer_backend::CudaError);
  EXPECT_EQ(producer.get_device_ptr(), imported->get_device_ptr());
  EXPECT_TRUE(all_equal(read_back(*imported, stream_), 0x77));
}

TEST_F(NvSciShareTest, ImportedBlockIsNotReusedAndStaleImportIsRefused)
{
  auto & runtime = cuda_buffer_backend::NvSciRuntime::get();
  cuda_buffer_backend::NvSciExport desc;
  const uint8_t * first_ptr = nullptr;
  std::unique_ptr<cuda_buffer_backend::NvSciBuffer> imported;
  {
    CudaBufferImpl<uint8_t> impl(8192, StorageKind::NvSci);
    fill(impl, 0x01, stream_);
    const auto & producer = static_cast<const cuda_buffer_backend::NvSciBuffer &>(
      impl.get_cuda_buffer());
    first_ptr = producer.get_device_ptr();
    ASSERT_TRUE(runtime.export_buffer(producer, local_subscriber(), &desc));
    imported = runtime.import_buffer(desc, impl.size());
  }
  cudaStreamSynchronize(stream_);
  // Buffers are released on another thread once their work is done.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // The importer still holds the block, so a new allocation gets another one.
  {
    CudaBufferImpl<uint8_t> other(8192, StorageKind::NvSci);
    EXPECT_NE(first_ptr, other.get_cuda_buffer().get_device_ptr());
  }

  imported.reset();
  cudaStreamSynchronize(stream_);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Once released, the block is reused: the subscriber handled its only
  // descriptor. The old descriptor no longer matches it.
  bool reused = false;
  std::vector<std::unique_ptr<CudaBufferImpl<uint8_t>>> held;
  for (int i = 0; i < 8 && !reused; ++i) {
    held.push_back(std::make_unique<CudaBufferImpl<uint8_t>>(8192, StorageKind::NvSci));
    reused = held.back()->get_cuda_buffer().get_device_ptr() == first_ptr;
  }
  EXPECT_TRUE(reused);
  EXPECT_THROW(runtime.import_buffer(desc, 8192), cuda_buffer_backend::CudaError);
}

TEST_F(NvSciShareTest, BlockStaysReservedUntilItsSubscriberHandlesTheDescriptor)
{
  auto & runtime = cuda_buffer_backend::NvSciRuntime::get();
  cuda_buffer_backend::NvSciExport desc;
  const uint8_t * first_ptr = nullptr;
  {
    CudaBufferImpl<uint8_t> impl(12288, StorageKind::NvSci);
    fill(impl, 0x05, stream_);
    const auto & producer = static_cast<const cuda_buffer_backend::NvSciBuffer &>(
      impl.get_cuda_buffer());
    first_ptr = producer.get_device_ptr();
    // Sent, but the subscriber has not taken it yet.
    ASSERT_TRUE(runtime.export_buffer(producer, local_subscriber("slow"), &desc));
  }
  cudaStreamSynchronize(stream_);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Nobody reads the block, but the subscriber may still: it is not reused,
  // however long the import takes.
  {
    CudaBufferImpl<uint8_t> other(12288, StorageKind::NvSci);
    EXPECT_NE(first_ptr, other.get_cuda_buffer().get_device_ptr());
  }

  // The late import still sees the data, and handling it ends the reservation.
  {
    auto imported = runtime.import_buffer(desc, 12288);
    EXPECT_TRUE(all_equal(read_back(*imported, stream_), 0x05));
  }
  cudaStreamSynchronize(stream_);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  bool reused = false;
  std::vector<std::unique_ptr<CudaBufferImpl<uint8_t>>> held;
  for (int i = 0; i < 8 && !reused; ++i) {
    held.push_back(std::make_unique<CudaBufferImpl<uint8_t>>(12288, StorageKind::NvSci));
    reused = held.back()->get_cuda_buffer().get_device_ptr() == first_ptr;
  }
  EXPECT_TRUE(reused);
}

#endif  // CUDA_BUFFER_ENABLE_NVSCI

int main(int argc, char ** argv)
{
  g_self = argv[0];
  if (argc > 1 && std::strcmp(argv[1], kPrintDefaultFlag) == 0) {
    std::printf(
      "%s\n", cuda_buffer_backend::to_string(cuda_buffer_backend::default_storage_kind()));
    return 0;
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
