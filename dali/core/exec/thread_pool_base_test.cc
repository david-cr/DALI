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

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "dali/core/exec/thread_pool_base.h"
#include "dali/core/format.h"
#include "dali/test/timing.h"

namespace dali {

struct SerialExecutor {
  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void()>>>
  AddTask(Runnable &&runnable) {
    runnable();
  }
};

TEST(NewThreadPool, AddTask) {
  ThreadPoolBase tp(4);
  std::atomic_int flag{0};
  for (int i = 0; i < 16; i++)
    tp.AddTask([&, i]() {
      int f = (flag |= (1 << i));
      if (f == 0xffff)
        flag.notify_all();
    });

  int f = flag.load();
  while (f != 0xffff) {
    flag.wait(f);
    f = flag.load();
  }
  // No conditions - this test succeeds if it doesn't hang
}

TEST(NewThreadPool, BulkAddTask) {
  ThreadPoolBase tp(4);
  std::atomic_int flag{0};
  {
    ThreadPoolBase::TaskBulkAdd bulk = tp.BeginBulkAdd();
    for (int i = 0; i < 16; i++)
      bulk.Add([&, i]() {
        int f = (flag |= (1 << i));
        if (f == 0xffff)
          flag.notify_all();
      });
    EXPECT_EQ(bulk.Size(), 16);
    // submitted automatically on destruction
  }

  int f = flag.load();
  while (f != 0xffff) {
    flag.wait(f);
    f = flag.load();
  }
  // No conditions - this test succeeds if it doesn't hang
}

TEST(NewThreadPool, RunJobInThreadPool) {
  Job job;
  ThreadPoolBase tp(4);
  int a = 0, b = 0, c = 0;
  job.AddTask([&]() {
    a = 1;
  });
  job.AddTask([&]() {
    b = 2;
  });
  job.AddTask([&]() {
    c = 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}

TEST(NewThreadPool, RunIncrementalJobInThreadPool) {
  ThreadPoolBase tp(4);
  IncrementalJob job;
  std::atomic_int a = 0, b = 0, c = 0;
  job.AddTask([&]() {
    a += 1;
  });
  job.AddTask([&]() {
    b += 2;
  });
  job.Run(tp, false);

  for (int i = 0; (a.load() != 1 || b.load() != 2) && i < 100000; i++)
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  ASSERT_TRUE(a.load() == 1 && b.load() == 2) << "The job didn't start.";

  job.AddTask([&]() {
    c += 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a.load(), 1);
  EXPECT_EQ(b.load(), 2);
  EXPECT_EQ(c.load(), 3);
}

TEST(NewThreadPool, RunLargeIncrementalJobInThreadPool) {
  ThreadPoolBase tp(4);
  const int max_attempts = 10;
  for (int attempt = 0; attempt < max_attempts; attempt++) {
    IncrementalJob job;
    std::atomic_int acc = 0;
    const int total_tasks = 40000;
    const int batch_size = 100;
    for (int i = 0; i < total_tasks; i += batch_size) {
      for (int j = i; j < i + batch_size; j++) {
        job.AddTask([&, j] {
          acc += j;
        });
      }
      job.Run(tp, false);
      if (i == 0) {
        for (int spin = 0; acc.load() == 0 && spin < 100000; spin++)
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        ASSERT_NE(acc.load(), 0) << "The job isn't running in the background.";
      }
    }
    int target_value = total_tasks * (total_tasks - 1) / 2;
    if (acc.load() == target_value) {
      if (attempt == max_attempts - 1) {
        FAIL() << "The job always finishes before a call to wait.";
      } else {
        std::cerr << "The job shouldn't have completed yet - retrying.\n";
      }
      job.Wait();
      continue;
    }
    job.Run(tp, true);
    EXPECT_EQ(acc.load(), target_value);
    break;
  }
}

template <typename JobType>
class NewThreadPoolJobTest : public ::testing::Test {};

template <typename JobType>
class NewThreadPoolCooperativeJobTest : public ::testing::Test {};

template <typename JobType>
class NewThreadPoolNonCooperativeJobTest : public ::testing::Test {};

using JobTypes = ::testing::Types<Job, IncrementalJob, CooperativeJob, CooperativeIncrementalJob>;
TYPED_TEST_SUITE(NewThreadPoolJobTest, JobTypes);

using CooperativeJobTypes = ::testing::Types<CooperativeJob, CooperativeIncrementalJob>;
TYPED_TEST_SUITE(NewThreadPoolCooperativeJobTest, CooperativeJobTypes);

using NonCooperativeJobTypes = ::testing::Types<Job, IncrementalJob>;
TYPED_TEST_SUITE(NewThreadPoolNonCooperativeJobTest, NonCooperativeJobTypes);


TYPED_TEST(NewThreadPoolJobTest, RunJobInSeries) {
  TypeParam job;
  SerialExecutor tp;
  int a = 0, b = 0, c = 0;
  job.AddTask([&]() {
    a = 1;
  });
  job.AddTask([&]() {
    b = 2;
  });
  job.AddTask([&]() {
    c = 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}

TYPED_TEST(NewThreadPoolJobTest, Discard) {
  EXPECT_NO_THROW({
    TypeParam job;
    job.AddTask([]() {});
    job.Discard();
  });
}

TYPED_TEST(NewThreadPoolJobTest, ErrorJobNotStarted) {
  try {
    TypeParam job;
    job.AddTask([]() {});
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "The job is not empty"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TYPED_TEST(NewThreadPoolJobTest, ErrorWaitBeforeRun) {
  TypeParam job;
  try {
    job.AddTask([]() {});
    job.Wait();
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "hasn't been run"));
    job.Discard();
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TYPED_TEST(NewThreadPoolJobTest, RethrowMultipleErrors) {
  TypeParam job;
  ThreadPoolBase tp(4);
  job.AddTask([&]() {
    throw std::runtime_error("Runtime");
  });
  job.AddTask([&]() {
    // do nothing
  });
  job.AddTask([&]() {
    throw std::logic_error("Logic");
  });
  EXPECT_THROW(job.Run(tp, true), MultipleErrors);
}

TYPED_TEST(NewThreadPoolCooperativeJobTest, Reentrant) {
  TypeParam job;
  ThreadPoolBase tp(1);  // must not hang with just one thread
  std::atomic_int outer{0}, inner{0};
  for (int i = 0; i < 10; i++) {
    job.AddTask([&, i]() {
        outer |= (i << 10);
    });
  }

  job.AddTask([&]() {
    TypeParam innerJob;

    for (int i = 0; i < 10; i++)
      innerJob.AddTask([&, i]() {
        inner |= (1 << i);
      });

    innerJob.Run(tp, false);
    innerJob.Wait();
    outer |= (1 << 11);
  });

  for (int i = 11; i < 20; i++) {
    job.AddTask([&, i]() {
        outer |= (1 << i);
    });
  }
  job.Run(tp, true);
}

TYPED_TEST(NewThreadPoolNonCooperativeJobTest, Reentrant) {
  TypeParam job;
  ThreadPoolBase tp(1);  // must not hang with just one thread
  job.AddTask([&]() {
    TypeParam innerJob;
    innerJob.AddTask([]() {});

    try {
      innerJob.Run(tp, true);
    } catch (...) {
      innerJob.Discard();
      throw;
    }
  });

  job.Run(tp, false);
  EXPECT_THROW(job.Wait(), std::logic_error);
}

TYPED_TEST(NewThreadPoolJobTest, JobPerf) {
  using JobType = TypeParam;
  ThreadPoolBase tp(4);
  auto do_test = [&](int jobs, int tasks) {
    std::vector<int> v(tasks);
    auto start = test::perf_timer::now();
    std::optional<JobType> j;
    for (int i = 0; i < jobs; i++) {
      j.emplace();
      for (int t = 1; t < tasks; t++) {
        j->AddTask([&, t]() {
          v[t]++;
        });
      }
      j->Run(tp, false);
      v[0]++;
      j->Wait();
      j.reset();
    }
    auto end = test::perf_timer::now();

    for (int t = 0; t < tasks; t++)
      EXPECT_EQ(v[t], jobs) << "Tasks didn't do their job";
    print(
        std::cout, "Ran ", jobs, " jobs of ", tasks, " tasks each in ",
        test::format_time(end - start), "\n");

    return end - start;
  };

  int total_tasks = 100000;
  int jobs0 = 10000, tasks0 = total_tasks / jobs0;
  auto time0 = do_test(jobs0, tasks0);
  int jobs1 = 100, tasks1 = total_tasks / jobs1;
  auto time1 = do_test(jobs1, tasks1);

  // time0 = task_time * total_tasks + job_overhead * jobs0
  // time1 = task_time * total_tasks + job_overhead * jobs1
  // hence
  // time0 - time1 = job_overhead * (jobs0 - jobs1)
  // job_overhead = (time0 - time1) / (jobs0 - jobs1)

  double job_overhead = test::seconds(time0 - time1) / (jobs0 - jobs1);
  print(std::cout, "Job overhead ", test::format_time(job_overhead), "\n");
}

/** Exposes ThreadPoolBase::Shutdown, which is protected, so that the tests can drive
 * the thread pool's post-shutdown behavior. */
class TestableThreadPool : public ThreadPoolBase {
 public:
  using ThreadPoolBase::ThreadPoolBase;
  using ThreadPoolBase::Shutdown;
};

TYPED_TEST(NewThreadPoolJobTest, ErrorWaitTwice) {
  ThreadPoolBase tp(4);
  TypeParam job;
  job.AddTask([]() {});
  job.Run(tp, true);
  try {
    job.Wait();
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "already been waited for"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TYPED_TEST(NewThreadPoolJobTest, WaitEmptyJobWithoutRun) {
  TypeParam job;
  // A job with no tasks may skip Run altogether, so waiting for it is legal.
  EXPECT_NO_THROW(job.Wait());
  EXPECT_FALSE(job.Started());
  EXPECT_TRUE(job.WaitStarted());
  EXPECT_TRUE(job.WaitCompleted());
}

TYPED_TEST(NewThreadPoolJobTest, ErrorDiscardAfterRun) {
  ThreadPoolBase tp(4);
  TypeParam job;
  job.AddTask([]() {});
  job.Run(tp, true);
  try {
    job.Discard();
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "already been started"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TYPED_TEST(NewThreadPoolNonCooperativeJobTest, ErrorWaitFromInsideThreadPool) {
  ThreadPoolBase tp(4);
  TypeParam job;
  job.AddTask([]() {});
  job.Run(tp, false);

  // A non-cooperative job cannot be waited for from within the pool it runs in.
  Job probe;
  probe.AddTask([&]() {
    job.Wait();
  });
  try {
    probe.Run(tp, true);
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "from inside the thread pool"));
    job.Wait();
    return;
  }
  job.Wait();
  GTEST_FAIL() << "Expected a logic error.";
}

TEST(NewThreadPool, ErrorRunJobTwice) {
  ThreadPoolBase tp(4);
  Job job;
  job.AddTask([]() {});
  job.Run(tp, true);
  try {
    job.Run(tp, true);
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "already been started"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TEST(NewThreadPool, ErrorIncrementalJobInAnotherExecutor) {
  ThreadPoolBase tp1(2), tp2(2);
  IncrementalJob job;
  job.AddTask([]() {});
  job.Run(tp1, true);
  try {
    job.Run(tp2, true);
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "different executor"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TEST(NewThreadPool, ErrorInitTwice) {
  ThreadPoolBase tp(2);
  try {
    tp.Init(2);
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "already started"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TEST(NewThreadPool, ErrorInitAfterShutdown) {
  TestableThreadPool tp(2);
  tp.Shutdown(true);
  EXPECT_EQ(tp.NumThreads(), 0);
  try {
    tp.Init(2);
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "being shut down"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TEST(NewThreadPool, ErrorAddTaskAfterShutdown) {
  TestableThreadPool tp(2);
  tp.Shutdown(true);
  try {
    tp.AddTask([]() {});
  } catch (std::logic_error &e) {
    EXPECT_NE(nullptr, strstr(e.what(), "no longer accepts new tasks"));
    return;
  }
  GTEST_FAIL() << "Expected a logic error.";
}

TEST(NewThreadPool, ShutdownIsIdempotent) {
  TestableThreadPool never_started;
  EXPECT_NO_THROW(never_started.Shutdown(true));

  TestableThreadPool tp(2);
  tp.Shutdown(true);
  EXPECT_NO_THROW(tp.Shutdown(false));
  EXPECT_NO_THROW(tp.Shutdown(true));
  EXPECT_EQ(tp.NumThreads(), 0);
}

TEST(NewThreadPool, CooperativeWaitWithEmptyTaskQueue) {
  ThreadPoolBase tp(4);
  std::atomic_bool release{false}, waiting{false};

  CooperativeJob slow;
  slow.AddTask([&]() {
    while (!release.load())
      std::this_thread::sleep_for(std::chrono::microseconds(100));
  });
  slow.Run(tp, false);

  // Releases the slow task only after the cooperative wait has had a chance to spin
  // for a while with nothing left in the pool's task queue.
  std::thread releaser([&]() {
    while (!waiting.load())
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    release = true;
  });

  CooperativeJob waiter;
  waiter.AddTask([&]() {
    waiting = true;
    slow.Wait();
  });
  waiter.Run(tp, true);
  releaser.join();
  EXPECT_TRUE(slow.WaitCompleted());
}

}  // namespace dali
