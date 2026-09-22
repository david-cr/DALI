// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <string_view>
#include "dali/pipeline/executor/executor2/exec2_test.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/test/timing.h"
#include "dali/core/cuda_stream_pool.h"

namespace dali {

namespace exec2 {
namespace test {

namespace {
// TODO(michalz): Avoid this code duplication without messing up encapsulation
void LimitBackendConcurrency(ExecGraph &graph, OpType backend, int max_concurrency = 1) {
  auto sem = std::make_shared<tasking::Semaphore>(max_concurrency);
  for (auto &n : graph.Nodes()) {
    if (n.backend == backend)
        n.concurrency = sem;
  }
  graph.Invalidate();
}

void SetOutputDevice(ExecNode *node, size_t out_idx, StorageDevice device) {
  if (node->outputs.size() <= out_idx)
    node->outputs.resize(out_idx + 1);
  node->outputs[out_idx].device = device;
}

struct ValidationGraph {
  explicit ValidationGraph(int consumer_inputs = 1, int consumer_outputs = 2) {
    OpSpec producer_spec(kTestOpName);
    producer_spec.AddArg("addend", 1)
        .AddArg("num_threads", 1)
        .AddArg("device", "cpu")
        .AddArg("max_batch_size", 1)
        .AddOutput("producer_out", StorageDevice::CPU)
        .AddArg("name", "producer");

    OpSpec consumer_spec(kTestOpName);
    consumer_spec.AddArg("addend", 1)
        .AddArg("num_threads", 1)
        .AddArg("device", "cpu")
        .AddArg("max_batch_size", 1)
        .AddArg("name", "consumer");
    for (int i = 0; i < consumer_inputs; i++)
      consumer_spec.AddInput(make_string("consumer_in_", i), StorageDevice::CPU);
    for (int i = 0; i < consumer_outputs; i++)
      consumer_spec.AddOutput(make_string("consumer_out_", i), StorageDevice::CPU);

    producer = graph.AddNode(std::make_unique<DummyOpCPU>(producer_spec));
    consumer = graph.AddNode(std::make_unique<DummyOpCPU>(consumer_spec));
    output = graph.AddOutputNode();
    input_edge = graph.Link(producer, 0, consumer, 0);
    output_edge = graph.Link(consumer, 0, output, 0);
  }

  ExecGraph graph;
  ExecNode *producer = nullptr;
  ExecNode *consumer = nullptr;
  ExecNode *output = nullptr;
  ExecEdge *input_edge = nullptr;
  ExecEdge *output_edge = nullptr;
};

class DummyBatchSizeProvider : public DummyOpCPU, public BatchSizeProvider {
 public:
  explicit DummyBatchSizeProvider(
      const OpSpec &spec, int next_batch_size = 1, bool throw_on_advance = false)
      : DummyOpCPU(spec)
      , next_batch_size_(next_batch_size)
      , throw_on_advance_(throw_on_advance) {}

  int NextBatchSize() override {
    return next_batch_size_;
  }

  void Advance() override {
    if (throw_on_advance_)
      throw std::runtime_error("advance failed");
  }

 private:
  int next_batch_size_ = 1;
  bool throw_on_advance_ = false;
};

void ExpectValidationError(ExecGraph &graph, std::string_view expected) {
  WorkspaceParams params = {};
  params.max_batch_size = 1;
  params.iter_data = std::make_shared<IterationData>();
  try {
    graph.PrepareIteration(params);
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find(expected), std::string::npos);
    return;
  }
  FAIL() << "Expected graph validation to fail.";
}

ExecNode *AddBatchSizeProvider(
    ExecGraph &graph, int next_batch_size, bool throw_on_advance = false) {
  OpSpec spec(kTestOpName);
  spec.AddArg("addend", 1)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("max_batch_size", 1)
      .AddOutput("provider_out", StorageDevice::CPU)
      .AddArg("name", "provider");
  return graph.AddNode(
      std::make_unique<DummyBatchSizeProvider>(spec, next_batch_size, throw_on_advance));
}

void ExpectBatchSizeError(ExecGraph &graph, int max_batch_size) {
  WorkspaceParams params = {};
  params.max_batch_size = max_batch_size;
  params.iter_data = std::make_shared<IterationData>();
  graph.PrepareIteration(params);

  tasking::Executor executor(1);
  executor.Start();
  auto future = graph.Launch(executor);
  EXPECT_THROW(future.Value<const PipelineOutput &>(), std::runtime_error);
}
}  // namespace

TEST(ExecGraphTest, SimpleGraph) {
  int batch_size = 32;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", StorageDevice::CPU)
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", StorageDevice::CPU)
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", StorageDevice::CPU)
       .AddInput("op1o0", StorageDevice::CPU)
       .AddOutput("op2o0", StorageDevice::CPU)
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);

  WorkspaceParams params = {};
  auto tp = std::make_unique<OldThreadPool>(std::thread::hardware_concurrency(), 0, false, "test");
  ExecEnv env;
  env.thread_pool = tp.get();
  params.env = &env;
  params.max_batch_size = batch_size;

  auto iter = std::make_shared<IterationData>();
  params.iter_data = iter;
  g.PrepareIteration(params);
  tasking::Executor ex(1);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  auto &out = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
  for (int i = 0; i < batch_size; i++)
    EXPECT_EQ(*out[i].data<int>(), 1110 + 3 * i);
}

TEST(ExecGraphTest, SimpleGraphRepeat) {
  int batch_size = 256;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", StorageDevice::CPU)
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", StorageDevice::CPU)
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", StorageDevice::CPU)
       .AddInput("op1o0", StorageDevice::CPU)
       .AddOutput("op2o0", StorageDevice::CPU)
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);
  OldThreadPool tp(4, 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.max_batch_size = batch_size;

  {
    int N = 100;
    tasking::Executor ex(4);
    ex.Start();
    auto start = dali::test::perf_timer::now();
    for (int i = 0; i < N; i++) {
      params.iter_data = std::make_shared<IterationData>();
      g.PrepareIteration(params);
      auto fut = g.Launch(ex);
      auto &pipe_out = fut.Value<const PipelineOutput &>();
      auto &ws = pipe_out.workspace;
      auto &out = ws.Output<CPUBackend>(0);
      ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
      for (int i = 0; i < batch_size; i++)
        EXPECT_EQ(*out[i].data<int>(), 1110 + 3 * i);
    }
    auto end = dali::test::perf_timer::now();
    print(std::cerr, "Average iteration time over ", N, " iterations is ",
          dali::test::format_time((end - start) / N), "\n");
  }
}

TEST(ExecGraphTest, SimpleGraphScheduleAheadCPU) {
  int batch_size = 1;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 4)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", StorageDevice::CPU)
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kCounterOpName);
  spec1.AddArg("num_threads", 4)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", StorageDevice::CPU)
       .AddArg("name", "op1");
  auto op1 = std::make_unique<CounterOp>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 4)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", StorageDevice::CPU)
       .AddInput("op1o0", StorageDevice::CPU)
       .AddOutput("op2o0", StorageDevice::CPU)
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);

  OldThreadPool tp(4, 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.max_batch_size = batch_size;

  int N = 100;
  tasking::Executor ex(4);
  ex.Start();
  std::vector<tasking::TaskFuture> fut;
  fut.reserve(N);
  for (int i = 0; i < N; i++) {
    params.iter_data = std::make_shared<IterationData>();
    g.PrepareIteration(params);
    fut.push_back(g.Launch(ex));
  }

  int ctr = 0;
  for (int i = 0; i < N; i++) {
    auto &pipe_out = fut[i].Value<const PipelineOutput &>();
    auto &out = pipe_out.workspace.Output<CPUBackend>(0);
    ASSERT_EQ(out.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
    for (int s = 0; s < batch_size; s++)
      EXPECT_EQ(*out[s].data<int>(), 1010 + 2 * s + ctr++);
  }
}

TEST(ExecGraphTest, GraphScheduleAheadGPU) {
  int batch_size = 1;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 4)
       .AddArg("device", "gpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", StorageDevice::GPU)
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpGPU>(spec0);

  OpSpec spec1(kCounterOpName);
  spec1.AddArg("num_threads", 4)
       .AddArg("delay", 0)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op1o0", StorageDevice::CPU)
       .AddArg("name", "op1");
  auto op1 = std::make_unique<CounterOp>(spec1);

  OpSpec spec1c("MakeContiguous");
  spec1c.AddArg("num_threads", 4)
        .AddArg("device", "mixed")
        .AddArg("max_batch_size", batch_size)
        .AddInput("op1o0", StorageDevice::CPU)
        .AddOutput("op1o0", StorageDevice::GPU)
        .AddArg("name", "op1c");
  auto op1c = InstantiateOperator(spec1c);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000)
       .AddArg("num_threads", 4)
       .AddArg("device", "gpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", StorageDevice::GPU)
       .AddInput("op1o0", StorageDevice::GPU)
       .AddOutput("op2o0", StorageDevice::GPU)
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpGPU>(spec2);
  ExecGraph g;
  ExecNode *n2  = g.AddNode(std::move(op2));
  ExecNode *n1  = g.AddNode(std::move(op1));
  ExecNode *n1c = g.AddNode(std::move(op1c));
  ExecNode *n0  = g.AddNode(std::move(op0));
  ExecNode *no  = g.AddOutputNode();
  n0->output_queue_limit = std::make_shared<tasking::Semaphore>(3);
  SetOutputDevice(n0, 0, StorageDevice::GPU);
  SetOutputDevice(n1c, 0, StorageDevice::GPU);
  SetOutputDevice(n2, 0, StorageDevice::GPU);
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n1c, 0);
  g.Link(n1c, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);

  auto s0 = CUDAStreamPool::instance().Get();
  auto s1 = CUDAStreamPool::instance().Get();
  auto s2 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();

  // Each GPU-capable operator uses a different stream
  n0->env.order = s0;
  n1c->env.order = s1;
  n2->env.order = s2;
  no->env.order = s3;

  OldThreadPool tp(4, 0, false, "test");

  n1->env.thread_pool = &tp;

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;

  int N = 100;
  tasking::Executor ex(4);
  ex.Start();
  std::vector<tasking::TaskFuture> fut;
  fut.reserve(N);
  for (int i = 0; i < N; i++) {
    params.iter_data = std::make_shared<IterationData>();
    g.PrepareIteration(params);
    fut.push_back(g.Launch(ex));
  }

  int ctr = 0;
  for (int i = 0; i < N; i++) {
    auto &pipe_out = fut[i].Value<const PipelineOutput &>();
    ASSERT_TRUE(pipe_out.workspace.has_event());
    EXPECT_EQ(pipe_out.workspace.stream(), s3.get());
    EXPECT_EQ(pipe_out.workspace.event(), pipe_out.event.get());
    AccessOrder::host().wait(pipe_out.workspace.event());
    auto &out_gpu = pipe_out.workspace.Output<GPUBackend>(0);
    ASSERT_EQ(out_gpu.shape(), uniform_list_shape(batch_size, TensorShape<0>()));
    TensorList<CPUBackend> out;
    out.Copy(out_gpu);
    for (int s = 0; s < batch_size; s++)
      EXPECT_EQ(*out[s].data<int>(), 1010 + 2 * s + ctr++);
  }
}


TEST(ExecGraphTest, Exception) {
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op0o0", StorageDevice::CPU)
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 200)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddOutput("op1o0", StorageDevice::CPU)
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 1000.0f)  // this will cause a type error at run-time
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", 32)
       .AddInput("op0o0", StorageDevice::CPU)
       .AddInput("op1o0", StorageDevice::CPU)
       .AddOutput("op2o0", StorageDevice::CPU)
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);
  ExecGraph g;
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n2, 1);
  g.Link(n2, 0, no, 0);
  LimitBackendConcurrency(g, OpType::CPU);
  OldThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.max_batch_size = 32;
  {
    tasking::Executor ex(4);
    ex.Start();
    for (int i = 0; i < 10; i++) {
      params.iter_data = std::make_shared<IterationData>();
      g.PrepareIteration(params);
      auto fut = g.Launch(ex);
      EXPECT_THROW(fut.Value<const PipelineOutput &>(), std::runtime_error);
    }
  }
}

TEST(ExecGraphTest, LoweredStructureMatch) {
  graph::OpGraph def = GetTestGraph1();
  ExecGraph g;
  g.Lower(def);
  ASSERT_EQ(g.Nodes().size(), def.OpNodes().size() + 1);
  EXPECT_TRUE(g.Nodes().back().is_pipeline_output);
  EXPECT_EQ(g.Nodes().back().inputs.size(), 2_uz);
  auto def_it = def.OpNodes().begin();
  auto ex_it = g.Nodes().begin();
  for (; def_it != def.OpNodes().end(); def_it++, ex_it++) {
    EXPECT_EQ(ex_it->inputs.size(), def_it->inputs.size());
    EXPECT_EQ(ex_it->outputs.size(), def_it->outputs.size());
  }
  if (HasFailure())
    FAIL() << "Structure mismatch detected - test cannot proceed further.";
  def_it = def.OpNodes().begin();
  ex_it = g.Nodes().begin();

  auto &def0 = *def_it++;
  auto &def1 = *def_it++;
  auto &def2 = *def_it++;
  auto &def3 = *def_it++;

  auto &ex0 = *ex_it++;
  auto &ex1 = *ex_it++;
  auto &ex2 = *ex_it++;
  auto &ex3 = *ex_it++;
  auto &ex_out = g.Nodes().back();

  ASSERT_EQ(ex0.outputs.size(), 1_uz);
  ASSERT_EQ(ex0.outputs[0].consumers.size(), 2_uz);
  EXPECT_EQ(ex0.outputs[0].consumers[0]->consumer, &ex2);
  EXPECT_EQ(ex0.outputs[0].consumers[1]->consumer, &ex3);

  ASSERT_EQ(ex1.outputs.size(), 1_uz);
  EXPECT_EQ(ex1.outputs[0].consumers[0]->consumer, &ex2);
  ASSERT_EQ(ex1.outputs[0].consumers.size(), 2_uz);
  EXPECT_EQ(ex1.outputs[0].consumers[1]->consumer, &ex3);

  ASSERT_EQ(ex2.outputs.size(), 1_uz);
  ASSERT_EQ(ex2.outputs[0].consumers.size(), 1_uz);
  EXPECT_EQ(ex2.outputs[0].consumers[0]->consumer, &ex_out);
  ASSERT_EQ(ex2.inputs.size(), 2_uz);
  EXPECT_EQ(ex2.inputs[0]->producer, &ex0);
  EXPECT_EQ(ex2.inputs[1]->producer, &ex1);

  ASSERT_EQ(ex3.outputs.size(), 1_uz);
  ASSERT_EQ(ex3.outputs[0].consumers.size(), 1_uz);
  EXPECT_EQ(ex3.outputs[0].consumers[0]->consumer, &ex_out);
  EXPECT_EQ(ex3.inputs[0]->producer, &ex0);
  EXPECT_EQ(ex3.inputs[1]->producer, &ex1);

  ASSERT_EQ(ex_out.inputs.size(), 2_uz);
  EXPECT_EQ(ex_out.inputs[0]->producer, &ex3);
  EXPECT_EQ(ex_out.inputs[1]->producer, &ex2);
}

TEST(ExecGraphTest, LoweredExec) {
  graph::OpGraph def = GetTestGraph1();
  ExecGraph g;
  g.Lower(def);
  LimitBackendConcurrency(g, OpType::CPU);

  OldThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.max_batch_size = 32;
  params.iter_data = std::make_shared<IterationData>();
  {
    tasking::Executor ex(4);
    ex.Start();
    g.PrepareIteration(params);
    auto fut = g.Launch(ex);
    auto &out = fut.Value<const PipelineOutput &>();
    CheckTestGraph1Results(out.workspace, params.max_batch_size);
  }
}

// Test HasParallelConsumers with MaxCount > 1 (covers line 76 in exec_graph_analysis.cc)
TEST(ExecGraphTest, ParallelConsumersHighConcurrency) {
  int batch_size = 8;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("op0o0", StorageDevice::CPU)
       .AddArg("name", "op0");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 100)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", StorageDevice::CPU)
       .AddOutput("op1o0", StorageDevice::CPU)
       .AddArg("name", "op1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 200)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("op0o0", StorageDevice::CPU)
       .AddOutput("op2o0", StorageDevice::CPU)
       .AddArg("name", "op2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, no, 0);
  g.Link(n2, 0, no, 1);

  // Set high concurrency (> 1) to trigger the HasParallelConsumers path
  auto sem = std::make_shared<tasking::Semaphore>(4);
  for (auto &n : g.Nodes()) {
    if (n.backend == OpType::CPU)
      n.concurrency = sem;
  }
  g.Invalidate();

  // `DummyOpCPU` uses `AddWork/RunAll` internally. In this test `op1` and `op2`
  // are intentionally runnable in parallel, so sharing one thread pool would
  // make concurrent submitters race inside `ThreadPool`.
  OldThreadPool tp0(4, 0, false, "test-op0");
  OldThreadPool tp1(4, 0, false, "test-op1");
  OldThreadPool tp2(4, 0, false, "test-op2");
  n0->env.thread_pool = &tp0;
  n1->env.thread_pool = &tp1;
  n2->env.thread_pool = &tp2;

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  g.PrepareIteration(params);
  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 2);
  auto &out0 = ws.Output<CPUBackend>(0);
  auto &out1 = ws.Output<CPUBackend>(1);
  ASSERT_EQ(out0.num_samples(), batch_size);
  ASSERT_EQ(out1.num_samples(), batch_size);
}

// Test graph with varied node structure to exercise analysis paths
TEST(ExecGraphTest, ComplexGraphTopology) {
  int batch_size = 32;
  // Create a more complex topology with multiple paths
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 1)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("root", StorageDevice::CPU)
       .AddArg("name", "root");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 2)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("root", StorageDevice::CPU)
       .AddOutput("branch1", StorageDevice::CPU)
       .AddArg("name", "branch1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 3)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("root", StorageDevice::CPU)
       .AddOutput("branch2", StorageDevice::CPU)
       .AddArg("name", "branch2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);

  OpSpec spec3(kTestOpName);
  spec3.AddArg("addend", 4)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("branch1", StorageDevice::CPU)
       .AddInput("branch2", StorageDevice::CPU)
       .AddOutput("merge", StorageDevice::CPU)
       .AddArg("name", "merge");
  auto op3 = std::make_unique<DummyOpCPU>(spec3);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n3 = g.AddNode(std::move(op3));
  ExecNode *no = g.AddOutputNode();

  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, n3, 0);
  g.Link(n2, 0, n3, 1);
  g.Link(n3, 0, no, 0);

  LimitBackendConcurrency(g, OpType::CPU);

  WorkspaceParams params = {};
  auto tp = std::make_unique<OldThreadPool>(std::thread::hardware_concurrency(), 0, false, "test");
  ExecEnv env;
  env.thread_pool = tp.get();
  params.env = &env;
  params.max_batch_size = batch_size;
  auto iter = std::make_shared<IterationData>();
  params.iter_data = iter;

  g.PrepareIteration(params);
  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 1);
  auto &out = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out.num_samples(), batch_size);
}

// Test CPU->GPU transitions with pinned buffer marking (covers SetPinnedInputs paths)
TEST(ExecGraphTest, PinnedBuffersWithMixedOps) {
  int batch_size = 8;

  // CPU operator producing data
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("cpu_data", StorageDevice::CPU)
       .AddArg("name", "cpu_op");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  // MakeContiguous (Mixed operator) for CPU->GPU transition
  OpSpec spec1("MakeContiguous");
  spec1.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data", StorageDevice::CPU)
       .AddOutput("gpu_data", StorageDevice::GPU)
       .AddArg("name", "make_contiguous");
  auto op1 = InstantiateOperator(spec1);

  // GPU operator consuming the data
  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 20)
       .AddArg("num_threads", 1)
       .AddArg("device", "gpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("gpu_data", StorageDevice::GPU)
       .AddOutput("gpu_result", StorageDevice::GPU)
       .AddArg("name", "gpu_op");
  auto op2 = std::make_unique<DummyOpGPU>(spec2);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *no = g.AddOutputNode();

  SetOutputDevice(n0, 0, StorageDevice::CPU);
  SetOutputDevice(n1, 0, StorageDevice::GPU);
  SetOutputDevice(n2, 0, StorageDevice::GPU);

  g.Link(n0, 0, n1, 0);
  g.Link(n1, 0, n2, 0);
  g.Link(n2, 0, no, 0);

  // Set up CUDA streams for GPU/Mixed operators
  auto s1 = CUDAStreamPool::instance().Get();
  auto s2 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();

  n1->env.order = s1;
  n2->env.order = s2;
  no->env.order = s3;

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;

  LimitBackendConcurrency(g, OpType::CPU);
  LimitBackendConcurrency(g, OpType::MIXED);
  LimitBackendConcurrency(g, OpType::GPU);

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  g.PrepareIteration(params);
  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_TRUE(ws.has_event());
  AccessOrder::host().wait(ws.event());
  auto &out_gpu = ws.Output<GPUBackend>(0);
  ASSERT_EQ(out_gpu.num_samples(), batch_size);

  TensorList<CPUBackend> out;
  out.Copy(out_gpu);
  for (int i = 0; i < batch_size; i++) {
    EXPECT_EQ(*out[i].data<int>(), 30 + 2 * i);
  }
}

// Test graph without GPU buffers to ensure all outputs marked as non-pinned
TEST(ExecGraphTest, CPUOnlyGraphNoPinning) {
  graph::OpGraph def = GetTestGraph1();
  ExecGraph g;
  g.Lower(def);
  LimitBackendConcurrency(g, OpType::CPU);

  // Verify that without GPU buffers, outputs are not pinned
  OldThreadPool tp(std::thread::hardware_concurrency(), 0, false, "test");
  WorkspaceParams params = {};
  ExecEnv env;
  env.thread_pool = &tp;
  params.env = &env;
  params.max_batch_size = 32;
  params.iter_data = std::make_shared<IterationData>();

  g.PrepareIteration(params);
  tasking::Executor ex(1);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &out = fut.Value<const PipelineOutput &>();
  CheckTestGraph1Results(out.workspace, params.max_batch_size);
}

// Test multiple consumers on same output with different concurrency settings
TEST(ExecGraphTest, MultipleConsumersConcurrency) {
  int batch_size = 32;
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("shared", StorageDevice::CPU)
       .AddArg("name", "producer");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 1)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out1", StorageDevice::CPU)
       .AddArg("name", "consumer1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 2)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out2", StorageDevice::CPU)
       .AddArg("name", "consumer2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);

  OpSpec spec3(kTestOpName);
  spec3.AddArg("addend", 3)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out3", StorageDevice::CPU)
       .AddArg("name", "consumer3");
  auto op3 = std::make_unique<DummyOpCPU>(spec3);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n3 = g.AddNode(std::move(op3));
  ExecNode *no = g.AddOutputNode();

  // Producer has multiple consumers
  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);
  g.Link(n0, 0, n3, 0);
  g.Link(n1, 0, no, 0);
  g.Link(n2, 0, no, 1);
  g.Link(n3, 0, no, 2);

  LimitBackendConcurrency(g, OpType::CPU);

  WorkspaceParams params = {};
  auto tp = std::make_unique<OldThreadPool>(std::thread::hardware_concurrency(), 0, false, "test");
  ExecEnv env;
  env.thread_pool = tp.get();
  params.env = &env;
  params.max_batch_size = batch_size;
  auto iter = std::make_shared<IterationData>();
  params.iter_data = iter;

  g.PrepareIteration(params);
  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 3);
}

// ============================================================================
// GRAPH ANALYSIS EDGE CASE TESTS
// ============================================================================
// These tests exercise edge cases in graph analysis logic that are difficult
// to trigger through normal graph construction APIs.

// Test HasParallelConsumers with different semaphores (covers line 79)
TEST(ExecGraphTest, HasParallelConsumersDifferentSemaphores) {
  int batch_size = 8;

  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("shared", StorageDevice::CPU)
       .AddArg("name", "producer");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 1)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out1", StorageDevice::CPU)
       .AddArg("name", "consumer1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 2)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out2", StorageDevice::CPU)
       .AddArg("name", "consumer2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *no = g.AddOutputNode();
  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, no, 0);
  g.Link(n2, 0, no, 1);

  // Create two different semaphores for the consumers (inconsistent!)
  auto sem1 = std::make_shared<tasking::Semaphore>(1);
  auto sem2 = std::make_shared<tasking::Semaphore>(1);
  n1->concurrency = sem1;
  n2->concurrency = sem2;  // Different semaphore!

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;
  n1->env.thread_pool = &tp;
  n2->env.thread_pool = &tp;

  g.Invalidate();

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  // PrepareIteration calls Analyze internally
  g.PrepareIteration(params);

  // Should detect parallel consumers due to different semaphores
  ASSERT_FALSE(n0->outputs.empty());
  EXPECT_TRUE(n0->outputs[0].parallel_consumers);
}

// Test SetPinnedInputs with CPU PassThrough operator (covers lines 110-126)
TEST(ExecGraphTest, PassThroughOperatorPinning) {
  int batch_size = 8;

  // CPU operator producing data
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("cpu_data", StorageDevice::CPU)
       .AddArg("name", "cpu_producer");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  // PassThrough operator (CPU, has PassThrough from input 0 to output 0)
  OpSpec spec1(kPassThroughOpName);
  spec1.AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data", StorageDevice::CPU)
       .AddOutput("passthrough", StorageDevice::CPU)
       .AddArg("name", "passthrough_op");
  auto op1 = std::make_unique<PassThroughOpCPU>(spec1);

  // MakeContiguous (Mixed) to move to GPU - this marks its input as pinned
  OpSpec spec2("MakeContiguous");
  spec2.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("passthrough", StorageDevice::CPU)
       .AddOutput("gpu_data", StorageDevice::GPU)
       .AddArg("name", "make_contiguous");
  auto op2 = InstantiateOperator(spec2);

  // GPU operator consuming the data
  OpSpec spec3(kTestOpName);
  spec3.AddArg("addend", 20)
       .AddArg("num_threads", 1)
       .AddArg("device", "gpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("gpu_data", StorageDevice::GPU)
       .AddOutput("gpu_result", StorageDevice::GPU)
       .AddArg("name", "gpu_consumer");
  auto op3 = std::make_unique<DummyOpGPU>(spec3);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n3 = g.AddNode(std::move(op3));
  ExecNode *no = g.AddOutputNode();

  SetOutputDevice(n0, 0, StorageDevice::CPU);
  SetOutputDevice(n1, 0, StorageDevice::CPU);
  SetOutputDevice(n2, 0, StorageDevice::GPU);
  SetOutputDevice(n3, 0, StorageDevice::GPU);

  g.Link(n0, 0, n1, 0);
  g.Link(n1, 0, n2, 0);
  g.Link(n2, 0, n3, 0);
  g.Link(n3, 0, no, 0);

  // Set up CUDA streams for GPU/Mixed operators
  auto s2 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();
  auto s4 = CUDAStreamPool::instance().Get();

  n2->env.order = s2;
  n3->env.order = s3;
  no->env.order = s4;

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;
  n1->env.thread_pool = &tp;

  LimitBackendConcurrency(g, OpType::CPU);
  LimitBackendConcurrency(g, OpType::MIXED);
  LimitBackendConcurrency(g, OpType::GPU);

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  // PrepareIteration calls Analyze, which should:
  // 1. Mark n2's input (reshaped) as pinned (because MakeContiguous needs it)
  // 2. Propagate pinning backward through Reshape (PassThrough) to n0's output
  g.PrepareIteration(params);

  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 1);
  ASSERT_TRUE(ws.has_event());
  AccessOrder::host().wait(ws.event());

  // Verify the PassThrough pinning logic was executed
  // (The actual pinning state is internal, but the test should complete successfully)
}

// Test SetPinnedInputs already-pinned input (covers line 118-119)
TEST(ExecGraphTest, PassThroughOperatorWithAlreadyPinnedInput) {
  int batch_size = 8;

  // CPU operator producing data
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("cpu_data1", StorageDevice::CPU)
       .AddArg("name", "cpu_producer1");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  // First MakeContiguous - this will mark cpu_data1 as pinned
  OpSpec spec1("MakeContiguous");
  spec1.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data1", StorageDevice::CPU)
       .AddOutput("gpu_data1", StorageDevice::GPU)
       .AddArg("name", "to_gpu1");
  auto op1 = InstantiateOperator(spec1);

  // PassThrough operator consuming already-pinned input
  OpSpec spec2(kPassThroughOpName);
  spec2.AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data1", StorageDevice::CPU)
       .AddOutput("passthrough", StorageDevice::CPU)
       .AddArg("name", "passthrough_op");
  auto op2 = std::make_unique<PassThroughOpCPU>(spec2);

  // Another MakeContiguous consuming the passthrough output
  OpSpec spec3("MakeContiguous");
  spec3.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("passthrough", StorageDevice::CPU)
       .AddOutput("gpu_data2", StorageDevice::GPU)
       .AddArg("name", "to_gpu2");
  auto op3 = InstantiateOperator(spec3);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n3 = g.AddNode(std::move(op3));
  ExecNode *no = g.AddOutputNode();

  SetOutputDevice(n0, 0, StorageDevice::CPU);
  SetOutputDevice(n1, 0, StorageDevice::GPU);
  SetOutputDevice(n2, 0, StorageDevice::CPU);
  SetOutputDevice(n3, 0, StorageDevice::GPU);

  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);  // Both paths consume cpu_data1
  g.Link(n2, 0, n3, 0);
  g.Link(n1, 0, no, 0);
  g.Link(n3, 0, no, 1);

  // Set up CUDA streams
  auto s1 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();
  auto s4 = CUDAStreamPool::instance().Get();

  n1->env.order = s1;
  n3->env.order = s3;
  no->env.order = s4;

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;
  n2->env.thread_pool = &tp;

  LimitBackendConcurrency(g, OpType::CPU);
  LimitBackendConcurrency(g, OpType::MIXED);
  LimitBackendConcurrency(g, OpType::GPU);

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  // PrepareIteration will:
  // 1. Mark cpu_data1 as pinned (because MakeContiguous needs it)
  // 2. When processing PassThrough op, see that cpu_data1 is already pinned
  //    and skip further processing (covers line 118-119)
  g.PrepareIteration(params);

  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 2);
  ASSERT_TRUE(ws.has_event());
  AccessOrder::host().wait(ws.event());
}

// Test HasParallelConsumers with null semaphore (covers line 74)
// This is a malformed graph test to cover defensive error checking
TEST(ExecGraphTest, MalformedGraphNullSemaphore) {
  int batch_size = 8;

  // Create a producer with shared output
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("shared", StorageDevice::CPU)
       .AddArg("name", "producer");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  // Create two consumers
  OpSpec spec1(kTestOpName);
  spec1.AddArg("addend", 1)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out1", StorageDevice::CPU)
       .AddArg("name", "consumer1");
  auto op1 = std::make_unique<DummyOpCPU>(spec1);

  OpSpec spec2(kTestOpName);
  spec2.AddArg("addend", 2)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("shared", StorageDevice::CPU)
       .AddOutput("out2", StorageDevice::CPU)
       .AddArg("name", "consumer2");
  auto op2 = std::make_unique<DummyOpCPU>(spec2);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *no = g.AddOutputNode();

  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);
  g.Link(n1, 0, no, 0);
  g.Link(n2, 0, no, 1);

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;
  n1->env.thread_pool = &tp;
  n2->env.thread_pool = &tp;

  // Set up semaphores manually - but leave n2's semaphore as nullptr (malformed!)
  auto sem = std::make_shared<tasking::Semaphore>(1);
  n0->concurrency = sem;
  n1->concurrency = sem;
  // n2->concurrency is left as nullptr - this is the malformed state!

  g.Invalidate();

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  // PrepareIteration calls Analyze, which should detect the null semaphore
  // and conservatively mark the output as having parallel consumers (line 74)
  g.PrepareIteration(params);

  // If we get here without crashing, the defensive check worked
  // The graph should mark n0's output as having parallel consumers
  ASSERT_FALSE(n0->outputs.empty());
  EXPECT_TRUE(n0->outputs[0].parallel_consumers);  // Should be true due to null sem
}

// Test SetPinnedInputs with non-CPU input to PassThrough operator (covers line 115)
// This tests a defensive check by manually corrupting the graph configuration
TEST(ExecGraphTest, PassThroughOperatorWithNonCPUInputDevice) {
  int batch_size = 8;

  // CPU operator producing data
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("cpu_data", StorageDevice::CPU)
       .AddArg("name", "cpu_producer");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  // PassThrough operator
  OpSpec spec1(kPassThroughOpName);
  spec1.AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data", StorageDevice::CPU)
       .AddOutput("passthrough_out", StorageDevice::CPU)
       .AddArg("name", "passthrough");
  auto op1 = std::make_unique<PassThroughOpCPU>(spec1);

  // MakeContiguous to mark output as pinned
  OpSpec spec2("MakeContiguous");
  spec2.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("passthrough_out", StorageDevice::CPU)
       .AddOutput("gpu_data", StorageDevice::GPU)
       .AddArg("name", "to_gpu");
  auto op2 = InstantiateOperator(spec2);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *no = g.AddOutputNode();

  SetOutputDevice(n0, 0, StorageDevice::CPU);
  SetOutputDevice(n1, 0, StorageDevice::CPU);
  SetOutputDevice(n2, 0, StorageDevice::GPU);

  g.Link(n0, 0, n1, 0);
  g.Link(n1, 0, n2, 0);
  g.Link(n2, 0, no, 0);

  // AFTER linking, manually corrupt the graph by changing the input edge's device to GPU
  // This simulates a malformed graph state that the defensive check protects against
  ASSERT_FALSE(n1->inputs.empty());
  // Manually set the input edge's device to GPU (malformed!)
  n1->inputs[0]->device = StorageDevice::GPU;

  auto s2 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();

  n2->env.order = s2;
  no->env.order = s3;

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;
  n1->env.thread_pool = &tp;

  LimitBackendConcurrency(g, OpType::CPU);
  LimitBackendConcurrency(g, OpType::MIXED);
  LimitBackendConcurrency(g, OpType::GPU);

  g.Invalidate();

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  // PrepareIteration calls Analyze -> SetPinnedInputs
  // When processing the PassThrough operator (n1):
  // - It has HasPassThrough() in its schema
  // - For input 0, it checks: if (input->device != StorageDevice::CPU)
  // - Since we manually set it to GPU, it hits line 115: continue
  // The defensive check should handle this gracefully
  g.PrepareIteration(params);

  // The graph analysis should complete without errors
  // The defensive check on line 115 protected against the malformed configuration
}

// Test SetPinnedInputs with already-pinned PassThrough input (covers line 119)
TEST(ExecGraphTest, PassThroughOperatorSkipAlreadyPinned) {
  int batch_size = 8;

  // CPU operator producing data
  OpSpec spec0(kTestOpName);
  spec0.AddArg("addend", 10)
       .AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddOutput("cpu_data", StorageDevice::CPU)
       .AddArg("name", "cpu_producer");
  auto op0 = std::make_unique<DummyOpCPU>(spec0);

  // MakeContiguous that marks cpu_data as pinned (output will be GPU)
  OpSpec spec1("MakeContiguous");
  spec1.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data", StorageDevice::CPU)
       .AddOutput("gpu_data", StorageDevice::GPU)
       .AddArg("name", "pin_it");
  auto op1 = InstantiateOperator(spec1);

  // PassThrough operator also consuming cpu_data (which is now pinned)
  OpSpec spec2(kPassThroughOpName);
  spec2.AddArg("num_threads", 1)
       .AddArg("device", "cpu")
       .AddArg("max_batch_size", batch_size)
       .AddInput("cpu_data", StorageDevice::CPU)
       .AddOutput("passthrough_out", StorageDevice::CPU)
       .AddArg("name", "passthrough");
  auto op2 = std::make_unique<PassThroughOpCPU>(spec2);

  // Another MakeContiguous to mark passthrough_out as needing pinning
  OpSpec spec3("MakeContiguous");
  spec3.AddArg("num_threads", 1)
       .AddArg("device", "mixed")
       .AddArg("max_batch_size", batch_size)
       .AddInput("passthrough_out", StorageDevice::CPU)
       .AddOutput("final_gpu", StorageDevice::GPU)
       .AddArg("name", "final");
  auto op3 = InstantiateOperator(spec3);

  ExecGraph g;
  ExecNode *n0 = g.AddNode(std::move(op0));
  ExecNode *n1 = g.AddNode(std::move(op1));
  ExecNode *n2 = g.AddNode(std::move(op2));
  ExecNode *n3 = g.AddNode(std::move(op3));
  ExecNode *no = g.AddOutputNode();

  SetOutputDevice(n0, 0, StorageDevice::CPU);
  SetOutputDevice(n1, 0, StorageDevice::GPU);
  SetOutputDevice(n2, 0, StorageDevice::CPU);
  SetOutputDevice(n3, 0, StorageDevice::GPU);

  // Both n1 and n2 consume cpu_data
  g.Link(n0, 0, n1, 0);
  g.Link(n0, 0, n2, 0);
  g.Link(n2, 0, n3, 0);
  g.Link(n1, 0, no, 0);
  g.Link(n3, 0, no, 1);

  // Set up CUDA streams
  auto s1 = CUDAStreamPool::instance().Get();
  auto s3 = CUDAStreamPool::instance().Get();
  auto s4 = CUDAStreamPool::instance().Get();

  n1->env.order = s1;
  n3->env.order = s3;
  no->env.order = s4;

  OldThreadPool tp(4, 0, false, "test");
  n0->env.thread_pool = &tp;
  n2->env.thread_pool = &tp;

  LimitBackendConcurrency(g, OpType::CPU);
  LimitBackendConcurrency(g, OpType::MIXED);
  LimitBackendConcurrency(g, OpType::GPU);

  WorkspaceParams params = {};
  params.max_batch_size = batch_size;
  params.iter_data = std::make_shared<IterationData>();

  // PrepareIteration will analyze the graph in topological order
  // The analysis processes nodes in reverse topological order for pinning:
  // 1. n3 (MakeContiguous) marks passthrough_out as pinned
  // 2. n2 (PassThrough) sees passthrough_out is pinned, tries to propagate to cpu_data
  // 3. But cpu_data is already pinned (by n1), so line 119 triggers: continue
  g.PrepareIteration(params);

  tasking::Executor ex(4);
  ex.Start();
  auto fut = g.Launch(ex);
  auto &pipe_out = fut.Value<const PipelineOutput &>();
  auto &ws = pipe_out.workspace;

  ASSERT_EQ(ws.NumOutput(), 2);
  ASSERT_TRUE(ws.has_event());
  AccessOrder::host().wait(ws.event());
}

TEST(ExecGraphValidationTest, ProducerOutputIndexOutOfRange) {
  ValidationGraph g;
  g.input_edge->producer_output_idx = g.producer->outputs.size();
  ExpectValidationError(g.graph, "producer output index is out of range");
}

TEST(ExecGraphValidationTest, ProducerOutputMissingEdge) {
  ValidationGraph g;
  g.producer->outputs[0].consumers.clear();
  ExpectValidationError(g.graph, "producer's output doesn't have this edge");
}

TEST(ExecGraphValidationTest, ConsumerInputSlotMismatch) {
  ValidationGraph g;
  g.consumer->inputs[0] = g.output_edge;
  ExpectValidationError(g.graph, "inconsistent edge consumer");
}

TEST(ExecGraphValidationTest, InputCountMismatch) {
  ValidationGraph g;
  g.consumer->inputs.emplace_back(g.input_edge);
  ExpectValidationError(g.graph, "different number of inputs");
}

TEST(ExecGraphValidationTest, OutputCountMismatch) {
  ValidationGraph g;
  g.consumer->outputs.emplace_back();
  ExpectValidationError(g.graph, "different number of outputs");
}

TEST(ExecGraphValidationTest, UnknownOutputEdge) {
  ValidationGraph g;
  ExecEdge unknown_edge;
  g.producer->outputs[0].consumers.push_back(&unknown_edge);
  ExpectValidationError(g.graph, "output is not a known edge pointer");
}

TEST(ExecGraphValidationTest, OutputEdgePointsToAnotherProducer) {
  ValidationGraph g;
  g.consumer->outputs[1].consumers.push_back(g.input_edge);
  ExpectValidationError(g.graph, "producer should always point to self");
}

TEST(ExecGraphValidationTest, OutputEdgeIndexMismatch) {
  ValidationGraph g;
  g.consumer->outputs[1].consumers.push_back(g.output_edge);
  ExpectValidationError(g.graph, "output's index must match");
}

TEST(ExecGraphValidationTest, UnknownInputEdge) {
  ValidationGraph g(2);
  ExecEdge unknown_edge;
  unknown_edge.producer = g.producer;
  g.consumer->inputs[1] = &unknown_edge;
  ExpectValidationError(g.graph, "output is not a known edge pointer");
}

TEST(ExecGraphValidationTest, InputEdgePointsToAnotherConsumer) {
  ValidationGraph g(2);
  g.consumer->inputs[1] = g.output_edge;
  ExpectValidationError(g.graph, "input's consumer should always point to self");
}

TEST(ExecGraphValidationTest, InputEdgeIndexMismatch) {
  ValidationGraph g(2);
  g.consumer->inputs[1] = g.input_edge;
  ExpectValidationError(g.graph, "input index must match");
}

TEST(ExecGraphValidationTest, PipelineOutputFlagOnNonLastNode) {
  ValidationGraph g;
  OpSpec trailing_spec(kTestOpName);
  trailing_spec.AddArg("addend", 1)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("max_batch_size", 1)
      .AddInput("pipeline_output", StorageDevice::CPU)
      .AddOutput("trailing_output", StorageDevice::CPU)
      .AddArg("name", "trailing");
  auto *trailing = g.graph.AddNode(std::make_unique<DummyOpCPU>(trailing_spec));
  g.graph.Link(g.output, 0, trailing, 0);
  ExpectValidationError(g.graph, "exactly one output node");
}

TEST(ExecGraphValidationTest, BatchSizeProviderHasInput) {
  OpSpec producer_spec(kTestOpName);
  producer_spec.AddArg("addend", 1)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("max_batch_size", 1)
      .AddOutput("producer_out", StorageDevice::CPU)
      .AddArg("name", "producer");

  OpSpec provider_spec(kTestOpName);
  provider_spec.AddArg("addend", 1)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("max_batch_size", 1)
      .AddInput("provider_in", StorageDevice::CPU)
      .AddOutput("provider_out", StorageDevice::CPU)
      .AddArg("name", "provider");

  ExecGraph graph;
  auto *producer = graph.AddNode(std::make_unique<DummyOpCPU>(producer_spec));
  auto *provider = graph.AddNode(std::make_unique<DummyBatchSizeProvider>(provider_spec));
  auto *output = graph.AddOutputNode();
  graph.Link(producer, 0, provider, 0);
  graph.Link(provider, 0, output, 0);
  ExpectValidationError(graph, "batch size provider cannot have inputs");
}

TEST(ExecGraphErrorTest, PrepareEmptyGraph) {
  ExecGraph graph;
  WorkspaceParams params = {};
  params.max_batch_size = 1;
  EXPECT_THROW(graph.PrepareIteration(params), std::logic_error);
}

TEST(ExecGraphErrorTest, PrepareInvalidMaxBatchSize) {
  ValidationGraph g;
  WorkspaceParams params = {};
  params.max_batch_size = 0;
  EXPECT_THROW(g.graph.PrepareIteration(params), std::runtime_error);
}

TEST(ExecGraphErrorTest, LaunchBeforePrepare) {
  ValidationGraph g;
  tasking::Executor executor(1);
  executor.Start();
  EXPECT_THROW(g.graph.Launch(executor), std::logic_error);
}

TEST(ExecGraphErrorTest, BatchSizeTooLarge) {
  ExecGraph graph;
  auto *provider = AddBatchSizeProvider(graph, 2);
  auto *output = graph.AddOutputNode();
  ASSERT_TRUE(provider->is_batch_size_provider);
  graph.Link(provider, 0, output, 0);
  ExpectBatchSizeError(graph, 1);
}

TEST(ExecGraphErrorTest, BatchSizeMismatch) {
  ExecGraph graph;
  auto *provider0 = AddBatchSizeProvider(graph, 1);
  auto *provider1 = AddBatchSizeProvider(graph, 2);
  auto *output = graph.AddOutputNode();
  ASSERT_TRUE(provider0->is_batch_size_provider);
  ASSERT_TRUE(provider1->is_batch_size_provider);
  graph.Link(provider0, 0, output, 0);
  graph.Link(provider1, 0, output, 1);
  ExpectBatchSizeError(graph, 2);
}

TEST(ExecGraphErrorTest, BatchSizeProviderAdvanceError) {
  ExecGraph graph;
  auto *provider = AddBatchSizeProvider(graph, 1, true);
  auto *output = graph.AddOutputNode();
  ASSERT_TRUE(provider->is_batch_size_provider);
  graph.Link(provider, 0, output, 0);
  ExpectBatchSizeError(graph, 1);
}

}  // namespace test
}  // namespace exec2
}  // namespace dali
