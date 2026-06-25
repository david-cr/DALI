// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/test/dali_test_decoder.h"
#include "dali/pipeline/executor/executor_impl.h"
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/async_separated_pipelined_executor.h"
#include "dali/pipeline/executor/executor_factory.h"
#include "dali/pipeline/operator/builtin/external_source.h"
#include "dali/test/dali_test_utils.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {

template <typename ExecutorToTest>
class ExecutorTest : public GenericDecoderTest<RGB> {
 protected:
  template <typename... T>
  std::unique_ptr<ExecutorToTest> GetExecutor(T&&... args) {
    return std::unique_ptr<ExecutorToTest>(new ExecutorToTest(std::forward<T>(args)...));
  }

  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs + t_decodeJPEGs;
  }

  void SetUp() override {
    DALISingleOpTest::SetUp();
    set_batch_size(jpegs_.nImages());
  }

  inline void set_batch_size(int size) { batch_size_ = size; }

  inline OpSpec& PrepareSpec(OpSpec &spec) const {
    spec.AddArg("max_batch_size", batch_size_)
      .AddArg("num_threads", num_threads_);
    return spec;
  }

  bool HasConditionals(ExecutorBase &exe) const {
    return exe.HasConditionals();
  }

  bool IsSeparated() {
    return std::is_same_v<ExecutorToTest, SeparatedPipelinedExecutor>
        || std::is_same_v<ExecutorToTest, AsyncSeparatedPipelinedExecutor>;
  }

  template<typename Factory>
  void RunCheckpointingTest(Factory executor_and_graph_factory,
                            int epoch_size, int epochs_cnt = 3) {
    auto collect_result = [&](const TensorList<CPUBackend> &data) {
      std::vector<uint8_t> result;
      for (int i = 0; i < data.num_samples(); i++)
        result.push_back(data.tensor<uint8_t>(i)[0]);
      return result;
    };

    Workspace ws;
    auto run_epoch = [&](std::unique_ptr<ExecutorToTest> &exec) {
      std::vector<std::vector<uint8_t>> results;
      for (int i = 0; i < epoch_size; i++) {
        exec->Run();
        exec->Outputs(&ws);

        if (ws.OutputIsType<CPUBackend>(0)) {
          results.push_back(collect_result(ws.Output<CPUBackend>(0)));
        } else {
          TensorList<CPUBackend> cpu;
          cpu.Copy(ws.Output<GPUBackend>(0));
          results.push_back(collect_result(cpu));
        }
      }

      return results;
    };

    auto [exec1, graph1] = executor_and_graph_factory();
    auto [exec2, graph2] = executor_and_graph_factory();

    for (int i = 0; i < epochs_cnt; i++)
      run_epoch(exec1);

    auto cpt = exec1->GetCurrentCheckpoint();
    exec2->RestoreStateFromCheckpoint(cpt);

    for (int i = 0; i < epochs_cnt; i++)
      EXPECT_EQ(run_epoch(exec1), run_epoch(exec2));
  }

  int batch_size_, num_threads_ = 1;
};

using ExecutorTypes =
    ::testing::Types<SimpleExecutor, PipelinedExecutor, SeparatedPipelinedExecutor,
                     AsyncPipelinedExecutor, AsyncSeparatedPipelinedExecutor>;

TYPED_TEST_SUITE(ExecutorTest, ExecutorTypes);

template <typename ExecutorToTest>
using ExecutorSyncTest = ExecutorTest<ExecutorToTest>;

using ExecutorSyncTypes =
    ::testing::Types<SimpleExecutor, PipelinedExecutor, SeparatedPipelinedExecutor>;

TYPED_TEST_SUITE(ExecutorSyncTest, ExecutorSyncTypes);

TYPED_TEST(ExecutorTest, TestRunBasicGraph) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", StorageDevice::CPU)
          .AddOutput("images", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", StorageDevice::CPU)
          .AddOutput("final_images", StorageDevice::CPU)), "");

  vector<string> outputs = {"final_images_cpu"};
  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, this->batch_size_);
  src_op->SetDataSource(tl);

  exe->Run();

  Workspace ws;
  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_TRUE(ws.OutputIsType<CPUBackend>(0));
}

TYPED_TEST(ExecutorTest, TestRunBasicGraphWithCB) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", StorageDevice::CPU)
          .AddOutput("images", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", StorageDevice::CPU)
          .AddOutput("final_images", StorageDevice::CPU)), "");

  vector<string> outputs = {"final_images_cpu"};

  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, this->batch_size_);
  src_op->SetDataSource(tl);

  exe->Run();

  Workspace ws;
  exe->Outputs(&ws);
  ASSERT_EQ(ws.NumInput(), 0);
  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_TRUE(ws.OutputIsType<CPUBackend>(0));
}

// This test does not work with Async Executors
TYPED_TEST(ExecutorSyncTest, TestPrefetchedExecution) {
  int batch_size = this->batch_size_ / 2;
  this->set_batch_size(batch_size);
  this->SetEps(1.6);

  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("device_id", 0)
          .AddOutput("data", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "cpu")
          .AddInput("data", StorageDevice::CPU)
          .AddOutput("images", StorageDevice::CPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "mixed")
          .AddInput("images", StorageDevice::CPU)
          .AddOutput("images", StorageDevice::GPU)), "");

  graph.AddOp(this->PrepareSpec(
          OpSpec("Copy")
          .AddArg("device", "gpu")
          .AddInput("images", StorageDevice::GPU)
          .AddOutput("final_images", StorageDevice::GPU)), "");

  vector<string> outputs = {"final_images_gpu"};
  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op =
      dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);

  TensorList<CPUBackend> tl;
  test::MakeRandomBatch(tl, this->batch_size_ * 2);

  // Split the batch into two
  TensorList<CPUBackend> tl2;
  TensorList<CPUBackend> tl1;
  TensorListShape<> shape1(batch_size, tl.shape().sample_dim()),
      shape2(batch_size, tl.shape().sample_dim());
  for (int i = 0; i < batch_size; ++i) {
    shape1.set_tensor_shape(i, tl.tensor_shape(i));
    shape2.set_tensor_shape(i, tl.tensor_shape(i+batch_size));
  }
  tl1.Resize(shape1, DALI_UINT8);
  tl2.Resize(shape2, DALI_UINT8);
  for (int i = 0; i < batch_size; ++i) {
    std::memcpy(
        tl1.template mutable_tensor<uint8_t>(i),
        tl.template tensor<uint8_t>(i),
        volume(tl.tensor_shape(i)));
    std::memcpy(
        tl2.template mutable_tensor<uint8_t>(i),
        tl.template tensor<uint8_t>(i+batch_size),
        volume(tl.tensor_shape(i+batch_size)));
  }


  Workspace ws;


  auto run = [&src_op, &exe] (TensorList<CPUBackend> &input) {
    src_op->SetDataSource(input);
    exe->Run();
  };

  auto check = [&exe, &ws, &tl, batch_size] (int batch_idx) {
    exe->Outputs(&ws);
    ASSERT_EQ(ws.NumOutput(), 1);
    ASSERT_EQ(ws.NumInput(), 0);
    ASSERT_TRUE(ws.OutputIsType<GPUBackend>(0));
    test::CheckResults(ws, batch_size, batch_idx, tl);
  };

  // Run twice without getting the results if we are not SimpleExecutor which will overwrite data
  // due to prefetch queue = 1.
  if (std::is_same_v<SimpleExecutor, TypeParam>) {
    run(tl1);
    check(0);
    run(tl2);
    check(1);
  } else {
    run(tl1);
    run(tl2);
    check(0);
    check(1);
  }
}


TYPED_TEST(ExecutorTest, TestPinning) {
  auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe->Init();

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                    .AddArg("device", "cpu")
                                    .AddArg("device_id", 0)
                                    .AddOutput("data_0", StorageDevice::CPU)),
              "ExternalSource_0");

  // First set of Copy + Copy and Pass Through
  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddArg("device", "cpu")
                                    .AddInput("data_0", StorageDevice::CPU)
                                    .AddOutput("copy_0", StorageDevice::CPU)),
              "Copy_0");

  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddArg("device", "cpu")
                                    .AddInput("data_0", StorageDevice::CPU)
                                    .AddOutput("copy_1", StorageDevice::CPU)),
              "Copy_1");

  graph.AddOp(this->PrepareSpec(OpSpec("PassthroughOp")
                                    .AddArg("device", "cpu")
                                    .AddInput("copy_0", StorageDevice::CPU)
                                    .AddOutput("pass_through_0", StorageDevice::CPU)),
              "PassThrough_0");

  // Trigger pinning of first set when it moves CPU -> GPU
  graph.AddOp(this->PrepareSpec(OpSpec("MakeContiguous")
                                    .AddArg("device", "mixed")
                                    .AddInput("pass_through_0", StorageDevice::CPU)
                                    .AddOutput("out_0", StorageDevice::GPU)),
              "MakeContiguous_0");

  // but not the Copy_1 to compare against
  graph.AddOp(this->PrepareSpec(OpSpec("MakeContiguous")
                                    .AddArg("device", "mixed")
                                    .AddInput("copy_1", StorageDevice::CPU)
                                    .AddOutput("out_1", StorageDevice::CPU)),
              "MakeContiguous_1");


  // Second set of Copy and Pass Through
  graph.AddOp(this->PrepareSpec(OpSpec("Copy")
                                    .AddArg("device", "cpu")
                                    .AddInput("data_0", StorageDevice::CPU)
                                    .AddOutput("copy_2", StorageDevice::CPU)),
              "Copy_2");

  graph.AddOp(this->PrepareSpec(OpSpec("PassthroughOp")
                                    .AddArg("device", "cpu")
                                    .AddInput("copy_2", StorageDevice::CPU)
                                    .AddOutput("pass_through_1", StorageDevice::CPU)),
              "PassThrough_1");

  // Check pinning argument inputs to operators in GPU stage
  graph.AddOp(this->PrepareSpec(OpSpec("CopyArgumentOp")
                                    .AddArg("device", "gpu")
                                    .AddArgumentInput("to_copy", "pass_through_1")
                                    .AddOutput("out_2", StorageDevice::GPU)),
              "DummyOpGpu");

  vector<string> outputs = {"copy_0_cpu",         "copy_1_cpu", "pass_through_0_cpu", "copy_2_cpu",
                            "pass_through_1_cpu", "out_0_gpu",  "out_1_cpu",          "out_2_gpu"};

  exe->Build(&graph, outputs);

  // Set the data for the external source
  auto *src_op = dynamic_cast<ExternalSource<CPUBackend> *>(graph.Node(OpType::CPU, 0).op.get());
  ASSERT_NE(src_op, nullptr);
  TensorList<CPUBackend> tl;
  tl.Resize(uniform_list_shape(this->batch_size_, TensorShape<>{}), DALI_FLOAT);
  src_op->SetDataSource(tl);

  exe->Run();

  Workspace ws;
  exe->Outputs(&ws);

  // Utilize the fact that the outputs are shared from the executor, so we can check if they are
  // pinned in a way we expect
  // Currently we expect to pin anything that is CPU argument input into GPU operator, and
  // is a CPU -> GPU copy (not via a decoder), so CPU input to Mixed operator that returns GPU data.
  // The whole pass-through group should be pinned as well.

  EXPECT_TRUE(ws.Output<CPUBackend>(0).is_pinned());   // copy_0_cpu
  EXPECT_FALSE(ws.Output<CPUBackend>(1).is_pinned());  // copy_1_cpu
  EXPECT_TRUE(ws.Output<CPUBackend>(2).is_pinned());   // pass_through_0_cpu
  EXPECT_TRUE(ws.Output<CPUBackend>(3).is_pinned());   // copy_2_cpu
  EXPECT_TRUE(ws.Output<CPUBackend>(4).is_pinned());   // pass_through_1_cpu
}


TYPED_TEST(ExecutorTest, TestCondtionalDetection) {
  auto exe_no_cond = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  auto exe_with_cond = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
  exe_no_cond->Init();
  exe_with_cond->Init();

  // Build a basic graph without conditionals.
  OpGraph graph_no_cond;
  graph_no_cond.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                            .AddArg("device", "cpu")
                                            .AddArg("device_id", 0)
                                            .AddOutput("data", StorageDevice::CPU)),
                      "ExternalSource");

  // Build a basic graph without conditionals.
  OpGraph graph_with_cond;
  graph_with_cond.AddOp(this->PrepareSpec(OpSpec("ExternalSource")
                                            .AddArg("device", "cpu")
                                            .AddArg("device_id", 0)
                                            .AddOutput("input", StorageDevice::CPU)),
                      "ExternalSource");

  graph_with_cond.AddOp(this->PrepareSpec(OpSpec("_conditional__Split")
                                              .AddArg("device", "cpu")
                                              .AddInput("input", StorageDevice::CPU)
                                              .AddArgumentInput("predicate", "input")
                                              .AddOutput("true_output", StorageDevice::CPU)
                                              .AddOutput("false_output", StorageDevice::CPU)
                                              .AddArg("_if_stmt", true)),
                        "split");

  graph_with_cond.AddOp(this->PrepareSpec(OpSpec("_conditional__Merge")
                                              .AddArg("device", "cpu")
                                              .AddInput("true_output", StorageDevice::CPU)
                                              .AddInput("false_output", StorageDevice::CPU)
                                              .AddArgumentInput("predicate", "input")
                                              .AddOutput("output", StorageDevice::CPU)),
                        "merge");

  exe_no_cond->Build(&graph_no_cond, {"data_cpu"});
  exe_with_cond->Build(&graph_with_cond, {"output_cpu"});

  EXPECT_FALSE(this->HasConditionals(*exe_no_cond));
  EXPECT_TRUE(this->HasConditionals(*exe_with_cond));
}


TYPED_TEST(ExecutorTest, SimpleCheckpointingCPU) {
  constexpr int epoch_size = 4;
  auto prepare_executor_and_graph = [&] {
    auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe->EnableCheckpointing(true);
    exe->Init();

    auto graph = std::make_unique<OpGraph>();
    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulSource")
          .AddArg("checkpointing", true)
          .AddArg("epoch_size", epoch_size)
          .AddOutput("state", StorageDevice::CPU)),
      "dummy");

    exe->Build(graph.get(), {"state_cpu"});
    return std::pair{std::move(exe), std::move(graph)};
  };

  if (this->IsSeparated())
    EXPECT_THROW(
      this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size),
      DALIException);
  else
    this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size);
}

TYPED_TEST(ExecutorTest, PipelineCheckpointingCPU) {
  constexpr int epoch_size = 4;
  auto prepare_executor_and_graph = [&] {
    auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe->EnableCheckpointing(true);
    exe->Init();

    auto graph = std::make_unique<OpGraph>();
    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulSource")
          .AddArg("checkpointing", true)
          .AddArg("epoch_size", epoch_size)
          .AddOutput("data", StorageDevice::CPU)),
      "dummy_src");

    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulOp")
          .AddArg("device", "cpu")
          .AddInput("data", StorageDevice::CPU)
          .AddOutput("processed", StorageDevice::CPU)),
      "dummy_op");

    exe->Build(graph.get(), {"processed_cpu"});
    return std::pair{std::move(exe), std::move(graph)};
  };

  if (this->IsSeparated())
    EXPECT_THROW(
      this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size),
      DALIException);
  else
    this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size);
}

TYPED_TEST(ExecutorTest, PipelineCheckpointingMixed) {
  constexpr int epoch_size = 4;
  auto prepare_executor_and_graph = [&] {
    auto exe = this->GetExecutor(this->batch_size_, this->num_threads_, 0, 1);
    exe->EnableCheckpointing(true);
    exe->Init();

    auto graph = std::make_unique<OpGraph>();
    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulSource")
          .AddArg("checkpointing", true)
          .AddArg("epoch_size", epoch_size)
          .AddOutput("data1", StorageDevice::CPU)),
      "dummy_src");

    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulOp")
          .AddArg("device", "mixed")
          .AddInput("data1", StorageDevice::CPU)
          .AddOutput("data2", StorageDevice::GPU)),
      "dummy_op1");

    graph->AddOp(
      this->PrepareSpec(
        OpSpec("TestStatefulOp")
          .AddArg("device", "gpu")
          .AddInput("data2", StorageDevice::GPU)
          .AddOutput("processed", StorageDevice::GPU)),
      "dummy_op2");

    exe->Build(graph.get(), {"processed_gpu"});
    return std::pair{std::move(exe), std::move(graph)};
  };

  if (this->IsSeparated())
    EXPECT_THROW(
      this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size),
      DALIException);
  else
    this->RunCheckpointingTest(prepare_executor_and_graph, epoch_size);
}

// Tests for executor_factory.cc coverage
class ExecutorFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear environment variables before each test
    unsetenv("DALI_EXEC2_MAX_THREADS");
    unsetenv("DALI_EXEC2_NUM_THREADS");
    unsetenv("DALI_USE_EXEC2");
  }

  void TearDown() override {
    // Clean up environment variables after each test
    unsetenv("DALI_EXEC2_MAX_THREADS");
    unsetenv("DALI_EXEC2_NUM_THREADS");
    unsetenv("DALI_USE_EXEC2");
  }
};

TEST_F(ExecutorFactoryTest, TestStreamPolicyPerOperator) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::StreamPolicyPerOperator;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestStreamPolicySingle) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::StreamPolicySingle;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestStreamPolicyPerBackend) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::StreamPolicyPerBackend;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestConcurrencyNone) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::ConcurrencyNone;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestConcurrencyFull) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::ConcurrencyFull;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestConcurrencyBackend) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::ConcurrencyBackend;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestExec2MaxThreadsEnvVar) {
  setenv("DALI_EXEC2_MAX_THREADS", "8", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestExec2MaxThreadsEnvVarInvalid) {
  setenv("DALI_EXEC2_MAX_THREADS", "0", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestExec2NumThreadsEnvVar) {
  setenv("DALI_EXEC2_NUM_THREADS", "6", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestExec2NumThreadsEnvVarZero) {
  setenv("DALI_EXEC2_NUM_THREADS", "0", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestForceExec2EnvVar) {
  setenv("DALI_USE_EXEC2", "1", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::AsyncPipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestInvalidExecutorType) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  // Use an invalid combination of executor type flags
  ExecutorType invalid_type = static_cast<ExecutorType>(0xFF);
  EXPECT_THROW(GetExecutor(invalid_type, flags, 4, 2, 0, 1024, qs), DALIException);
}

TEST_F(ExecutorFactoryTest, TestAllExecutorTypes) {
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;

  auto simple = GetExecutor(ExecutorType::Simple, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(simple, nullptr);

  auto pipelined = GetExecutor(ExecutorType::Pipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(pipelined, nullptr);

  auto separated = GetExecutor(ExecutorType::SeparatedPipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(separated, nullptr);

  auto async = GetExecutor(ExecutorType::AsyncPipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(async, nullptr);

  auto async_sep = GetExecutor(ExecutorType::AsyncSeparatedPipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(async_sep, nullptr);

  auto dynamic = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(dynamic, nullptr);
}

TEST_F(ExecutorFactoryTest, TestCombinedFlags) {
  QueueSizes qs{2, 2};

  // Test combination of stream policy and concurrency
  ExecutorFlags flags1 = ExecutorFlags::StreamPolicyPerOperator |
                         ExecutorFlags::ConcurrencyFull;
  auto executor1 = GetExecutor(ExecutorType::Dynamic, flags1, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor1, nullptr);

  ExecutorFlags flags2 = ExecutorFlags::StreamPolicySingle |
                         ExecutorFlags::ConcurrencyNone;
  auto executor2 = GetExecutor(ExecutorType::Dynamic, flags2, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor2, nullptr);
}

TEST_F(ExecutorFactoryTest, TestExec2MaxThreadsEnvVarNegative) {
  setenv("DALI_EXEC2_MAX_THREADS", "-1", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestExec2NumThreadsEnvVarNegative) {
  setenv("DALI_EXEC2_NUM_THREADS", "-1", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::Dynamic, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestForceExec2EnvVarZero) {
  setenv("DALI_USE_EXEC2", "0", 1);
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::AsyncPipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestAsyncPipelinedWithoutForceExec2) {
  // Ensure DALI_USE_EXEC2 is not set to test the else branch
  unsetenv("DALI_USE_EXEC2");
  QueueSizes qs{2, 2};
  ExecutorFlags flags = ExecutorFlags::None;
  auto executor = GetExecutor(ExecutorType::AsyncPipelined, flags, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor, nullptr);
}

TEST_F(ExecutorFactoryTest, TestDynamicExecutorStreamPolicies) {
  QueueSizes qs{2, 2};

  // Test StreamPolicyPerOperator with Dynamic executor
  ExecutorFlags flags1 = ExecutorFlags::StreamPolicyPerOperator;
  auto executor1 = GetExecutor(ExecutorType::Dynamic, flags1, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor1, nullptr);

  // Test StreamPolicySingle with Dynamic executor
  ExecutorFlags flags2 = ExecutorFlags::StreamPolicySingle;
  auto executor2 = GetExecutor(ExecutorType::Dynamic, flags2, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor2, nullptr);

  // Test StreamPolicyPerBackend with Dynamic executor
  ExecutorFlags flags3 = ExecutorFlags::StreamPolicyPerBackend;
  auto executor3 = GetExecutor(ExecutorType::Dynamic, flags3, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor3, nullptr);
}

TEST_F(ExecutorFactoryTest, TestDynamicExecutorConcurrencyModes) {
  QueueSizes qs{2, 2};

  // Test ConcurrencyNone with Dynamic executor
  ExecutorFlags flags1 = ExecutorFlags::ConcurrencyNone;
  auto executor1 = GetExecutor(ExecutorType::Dynamic, flags1, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor1, nullptr);

  // Test ConcurrencyFull with Dynamic executor
  ExecutorFlags flags2 = ExecutorFlags::ConcurrencyFull;
  auto executor2 = GetExecutor(ExecutorType::Dynamic, flags2, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor2, nullptr);

  // Test ConcurrencyBackend with Dynamic executor
  ExecutorFlags flags3 = ExecutorFlags::ConcurrencyBackend;
  auto executor3 = GetExecutor(ExecutorType::Dynamic, flags3, 4, 2, 0, 1024, qs);
  ASSERT_NE(executor3, nullptr);
}

// Tests for async_pipelined_executor.cc coverage.
//
// Exposes the protected scheduling entry point and the internal error/stop
// state so the asynchronous CPU stage can be driven directly without a full
// pipeline run. On the error/stop path RunCPU's worker lambda returns early
// (notifying the mixed stage) before PipelinedExecutor::RunCPU() is reached,
// so no graph needs to be built.
class TestableAsyncExecutor : public AsyncPipelinedExecutor {
 public:
  using AsyncPipelinedExecutor::AsyncPipelinedExecutor;
  using AsyncPipelinedExecutor::RunCPU;

  void ForceExecError() { exec_error_ = true; }
  void TriggerStop() { this->SignalStop(); }
  void WaitForCpuWork() { cpu_thread_.WaitForWork(false); }
};

// When the executor is already in an error state, the CPU worker lambda must
// decrement the work counter, notify the mixed stage, and return early instead
// of running the CPU stage. This exercises the `exec_error_ || IsStopSignaled()`
// early-return branch in AsyncPipelinedExecutor::RunCPU.
TEST(AsyncPipelinedExecutorErrorTest, RunCPUReturnsEarlyOnExecError) {
  TestableAsyncExecutor exe(/*batch_size=*/1, /*num_thread=*/1, /*device_id=*/0,
                            /*bytes_per_sample_hint=*/1);
  exe.Init();

  exe.ForceExecError();
  exe.RunCPU();
  // Block until the scheduled CPU lambda has actually run, so the early-return
  // branch is recorded by the coverage instrumentation.
  EXPECT_NO_THROW(exe.WaitForCpuWork());
}

// Same early-return branch, but triggered through the stop signal rather than
// the error flag, mirroring how Shutdown()/Outputs() abort in-flight work.
TEST(AsyncPipelinedExecutorErrorTest, RunCPUReturnsEarlyOnStopSignal) {
  TestableAsyncExecutor exe(/*batch_size=*/1, /*num_thread=*/1, /*device_id=*/0,
                            /*bytes_per_sample_hint=*/1);
  exe.Init();

  exe.TriggerStop();
  exe.RunCPU();
  EXPECT_NO_THROW(exe.WaitForCpuWork());
}

}  // namespace dali
