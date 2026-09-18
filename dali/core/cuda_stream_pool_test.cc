// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <atomic>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/mm/memory.h"

namespace dali {

class CUDAStreamPoolTest : public ::testing::Test {
 public:
  void TestPutGet() {
    int devices = 0;
    (void)cudaGetDeviceCount(&devices);
    if (devices == 0) {
      (void)cudaGetLastError();  // No CUDA devices - we don't care about the error
      GTEST_SKIP();
    }

    CUDAStreamPool pool;

    vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
      threads.emplace_back([&]() {
        std::mt19937_64 rng;
        std::uniform_int_distribution<int> dev_dist(0, devices-1);
        std::uniform_int_distribution<int> release_method(0, 4);
        for (int i = 0; i < 10000; i++) {
          int device_id = dev_dist(rng);
          CUDAStreamLease s = pool.Get(device_id);  // not current device (in general)
          CUDAStreamLease stream = std::move(s);  // check move constructor
          ASSERT_EQ(DeviceFromStream(stream), device_id)
            << "CUDAStreamPool mixed streams from different devices";
          ASSERT_EQ(cudaSuccess, cudaSetDevice(device_id));
          ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
          switch (release_method(rng)) {
            case 1:
              stream.reset();
              break;
            case 2:
              pool.Put(stream.release());
              break;
            case 3:
              pool.Put(stream.release(), device_id);
              break;
            case 4:
              stream.release().reset();
              break;
            default:
              // just go out of scope
              break;
          }
        }
      });
    }
    for (auto &t : threads)
      t.join();
    ASSERT_EQ(0, pool.lease_count_.load());
  }
};

namespace test {

TEST_F(CUDAStreamPoolTest, PutGet) {
  TestPutGet();
}

TEST_F(CUDAStreamPoolTest, GetSameNotBusy) {
  CUDAStreamPool pool;
  auto s = pool.Get();
  ASSERT_EQ(cudaStreamQuery(s), cudaSuccess);
  cudaStream_t s1 = s.get();
  s.reset();
  s = pool.Get();
  EXPECT_EQ(s.get(), s1);
}

TEST_F(CUDAStreamPoolTest, GetNotBusy) {
  for (int attempt = 0; attempt < 100; attempt++) {
    CUDAStreamPool pool;
    auto s = pool.Get();
    cudaStream_t s1 = s.get();
    ASSERT_EQ(cudaStreamQuery(s), cudaSuccess);
    const int N = 16 << 20;
    auto mem = mm::alloc_raw_async_unique<char, mm::memory_kind::device>(N, s, s, 256);
    for (int i = 0; i < 128; i++)
      CUDA_CALL(cudaMemsetAsync(mem.get(), i, N, s));
    s.reset();
    auto ret = cudaStreamQuery(s1);
    s = pool.Get();
    if (s == s1) {
      // If we got s1 again, it must have finished its work
      ASSERT_EQ(cudaStreamQuery(s1), cudaSuccess);
      mem.reset();
      pool.Purge();
      std::cerr << "Reattempting." << std::endl;
      continue;
    }
    EXPECT_EQ(cudaStreamQuery(s), cudaSuccess);
    s.reset();
    CUDA_CALL(cudaStreamSynchronize(s1));  // This stream should still live in the pool
    mem.reset();
    pool.Purge();
    return;
  }
  GTEST_SKIP() << "Weak test - CPU wasn't able to outrun GPU tasks.";
}

TEST_F(CUDAStreamPoolTest, GetCompletedBusyStream) {
  struct Blocker {
    std::atomic_bool entered{false};
    std::atomic_bool release{false};
  } blocker;

  CUDAStreamPool pool;
  auto lease = pool.Get(0);
  cudaStream_t raw = lease.get();
  CUDA_CALL(cudaLaunchHostFunc(raw, [](void *arg) {
    auto &blocker = *static_cast<Blocker *>(arg);
    blocker.entered = true;
    while (!blocker.release)
      std::this_thread::yield();
  }, &blocker));

  while (!blocker.entered)
    std::this_thread::yield();
  lease.reset();  // the blocked stream goes to the busy list

  blocker.release = true;
  CUDA_CALL(cudaStreamSynchronize(raw));

  auto reused = pool.Get(0);  // the completed stream is recovered from the busy list
  EXPECT_EQ(reused.get(), raw);
  reused.reset();
  pool.Purge();
}

TEST_F(CUDAStreamPoolTest, ErrorPutNullStream) {
  CUDAStreamPool pool;
  EXPECT_THROW(pool.Put(CUDAStream()), std::invalid_argument);
}

TEST_F(CUDAStreamPoolTest, PutWithoutDeviceId) {
  CUDAStreamPool pool;  // never used for Get, so Put is what initializes it
  CUDAStream stream = CUDAStream::Create(true, 0);
  cudaStream_t raw = stream.get();
  EXPECT_NO_THROW(pool.Put(std::move(stream)));  // device taken from the stream itself

  auto lease = pool.Get(0);
  EXPECT_EQ(lease.get(), raw) << "The stream should have been returned from the pool.";
  lease.reset();
  pool.Purge();
}

}  // namespace test
}  // namespace dali
