// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/util/thread_pool.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>

namespace dali {

namespace test {

TEST(ThreadPool, AddWork) {
  OldThreadPool tp(16, 0, false, "OldThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.AddWork(increase);
  }
  ASSERT_EQ(count, 0);
  tp.RunAll();
  ASSERT_EQ(count, 64);
}


TEST(ThreadPool, AddWorkWithPriority) {
  // only one thread to ensure deterministic behavior
  OldThreadPool tp(1, 0, false, "OldThreadPool test");
  std::atomic<int> count{0};
  auto set_to_1 = [&count](int thread_id) {
    count = 1;
  };
  auto increase_by_1 = [&count](int thread_id) {
    count++;
  };
  auto mult_by_2 = [&count](int thread_id) {
    int val = count.load();
    while (!count.compare_exchange_weak(val, val * 2)) {}
  };
  tp.AddWork(increase_by_1, 2);
  tp.AddWork(mult_by_2, 7);
  tp.AddWork(mult_by_2, 9);
  tp.AddWork(mult_by_2, 8);
  tp.AddWork(increase_by_1, 100);
  tp.AddWork(set_to_1, 1000);

  tp.RunAll();
  ASSERT_EQ(((1+1) << 3) + 1, count);
}


TEST(ThreadPool, CheckName) {
  const char given_thread_pool_name[] = "OldThreadPool test";
  const char full_thread_pool_name[] = "[DALI][TP0]OldThreadPool test";
  // max len supported by pthread_getname_np is 16
  char read_thread_pool_name[16] = {0, };
  // only one thread to ensure deterministic behavior
  OldThreadPool tp(1, 0, false, given_thread_pool_name);
  auto set_name = [&read_thread_pool_name](int thread_id) {
    pthread_getname_np(pthread_self(), read_thread_pool_name, sizeof(read_thread_pool_name));
  };
  tp.AddWork(set_name, 1);

  tp.RunAll();
  // skip terminating \0 character
  ASSERT_EQ(0, memcmp(read_thread_pool_name, full_thread_pool_name,
                      std::min(sizeof(full_thread_pool_name), sizeof(read_thread_pool_name)) - 1));
}

TEST(ThreadPool, InvalidThreadCount) {
  // Test that constructor throws when thread count is zero
  ASSERT_THROW(ThreadPool(0, 0, false, "test"), std::exception);
  // Test that constructor throws when thread count is negative
  ASSERT_THROW(ThreadPool(-1, 0, false, "test"), std::exception);
}

TEST(ThreadPool, NumThreads) {
  ThreadPool tp(8, 0, false, "ThreadPool test");
  ASSERT_EQ(tp.NumThreads(), 8);
}

TEST(ThreadPool, GetThreadIds) {
  ThreadPool tp(4, 0, false, "ThreadPool test");
  auto thread_ids = tp.GetThreadIds();
  ASSERT_EQ(thread_ids.size(), 4);
  // Check that all thread IDs are unique
  for (size_t i = 0; i < thread_ids.size(); i++) {
    for (size_t j = i + 1; j < thread_ids.size(); j++) {
      ASSERT_NE(thread_ids[i], thread_ids[j]);
    }
  }
}

TEST(ThreadPool, RunAllAfterStarted) {
  ThreadPool tp(4, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };

  // Start work immediately
  for (int i = 0; i < 10; i++) {
    tp.AddWork(increase, 0, true);
  }
  // Call RunAll after work has already started (tests the started_ branch)
  tp.RunAll(false);

  // Add more work after started
  for (int i = 0; i < 10; i++) {
    tp.AddWork(increase);
  }

  tp.WaitForWork();
  ASSERT_EQ(count, 20);
}

TEST(ThreadPool, ExceptionHandling) {
  ThreadPool tp(2, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto throw_exception = [](int thread_id) {
    throw std::runtime_error("Test exception");
  };
  auto increase = [&count](int thread_id) { count++; };

  tp.AddWork(throw_exception);
  tp.AddWork(increase);
  tp.AddWork(increase);

  tp.RunAll(false);

  // WaitForWork with checkForErrors = true should rethrow the exception
  ASSERT_THROW(tp.WaitForWork(true), std::runtime_error);

  // Count should still be 2 from the successful tasks
  ASSERT_EQ(count, 2);
}

TEST(ThreadPool, WaitForWorkNoErrors) {
  ThreadPool tp(4, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };

  for (int i = 0; i < 20; i++) {
    tp.AddWork(increase);
  }

  tp.RunAll(false);
  // WaitForWork with checkForErrors = false should not throw
  tp.WaitForWork(false);
  ASSERT_EQ(count, 20);
}

TEST(ThreadPool, MultipleExceptions) {
  ThreadPool tp(4, 0, false, "ThreadPool test");
  auto throw_exception = [](int thread_id) {
    throw std::runtime_error("Test exception");
  };

  // Add multiple tasks that throw exceptions
  for (int i = 0; i < 8; i++) {
    tp.AddWork(throw_exception);
  }

  tp.RunAll(false);

  // Should throw the first exception that occurred
  ASSERT_THROW(tp.WaitForWork(true), std::runtime_error);
}

TEST(ThreadPool, AffinityMask) {
  // Save the original environment variable value
  const char* original_affinity = std::getenv("DALI_AFFINITY_MASK");
  std::string original_value;
  bool had_original = false;
  if (original_affinity) {
    original_value = original_affinity;
    had_original = true;
  }

  // Set DALI_AFFINITY_MASK environment variable
  setenv("DALI_AFFINITY_MASK", "0,1,2,3", 1);

  // Create thread pool with affinity enabled
  ThreadPool tp(4, 0, true, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };

  for (int i = 0; i < 16; i++) {
    tp.AddWork(increase);
  }

  tp.RunAll();
  ASSERT_EQ(count, 16);

  // Restore the original environment variable
  if (had_original) {
    setenv("DALI_AFFINITY_MASK", original_value.c_str(), 1);
  } else {
    unsetenv("DALI_AFFINITY_MASK");
  }
}

TEST(ThreadPool, AffinityMaskInsufficientEntries) {
  // Save the original environment variable value
  const char* original_affinity = std::getenv("DALI_AFFINITY_MASK");
  std::string original_value;
  bool had_original = false;
  if (original_affinity) {
    original_value = original_affinity;
    had_original = true;
  }

  // Set DALI_AFFINITY_MASK with fewer entries than threads
  setenv("DALI_AFFINITY_MASK", "0,1", 1);

  // Create thread pool with more threads than affinity mask entries
  ThreadPool tp(4, 0, true, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };

  for (int i = 0; i < 16; i++) {
    tp.AddWork(increase);
  }

  tp.RunAll();
  ASSERT_EQ(count, 16);

  // Restore the original environment variable
  if (had_original) {
    setenv("DALI_AFFINITY_MASK", original_value.c_str(), 1);
  } else {
    unsetenv("DALI_AFFINITY_MASK");
  }
}

}  // namespace test

}  // namespace dali
