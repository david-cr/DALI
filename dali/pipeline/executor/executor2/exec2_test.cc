// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/executor/executor2/exec2_test.h"
#include "dali/pipeline/executor/executor2/exec2.h"
#include <optional>
#include <stdexcept>
#include "dali/pipeline/operator/checkpointing/checkpoint.h"

namespace std {
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::optional<T> &opt) {
  if (opt)
    return os << *opt;
  else
    return os << "<nullopt>";
}
}  // namespace std

namespace dali {
namespace exec2 {

#define PRINT_ENUM_VALUE(_label) case decltype(value)::_label:\
  os << #_label; break;

inline std::ostream &operator<<(std::ostream &os, StreamPolicy value) {
  switch (value) {
    PRINT_ENUM_VALUE(Single);
    PRINT_ENUM_VALUE(PerBackend);
    PRINT_ENUM_VALUE(PerOperator);
    default:
      os << static_cast<std::underlying_type_t<decltype(value)>>(value);
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, OperatorConcurrency value) {
  switch (value) {
    PRINT_ENUM_VALUE(None);
    PRINT_ENUM_VALUE(Full);
    PRINT_ENUM_VALUE(Backend);
    default:
      os << static_cast<std::underlying_type_t<decltype(value)>>(value);
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, QueueDepthPolicy value) {
  switch (value) {
    PRINT_ENUM_VALUE(FullyBuffered);
    PRINT_ENUM_VALUE(BackendChange);
    PRINT_ENUM_VALUE(OutputOnly);
    default:
      os << static_cast<std::underlying_type_t<decltype(value)>>(value);
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Executor2::Config &cfg) {
  #define PRINT_CONFIG_FIELD(field) #field " : ", cfg.field, "\n"
  print(os,
    PRINT_CONFIG_FIELD(device),
    PRINT_CONFIG_FIELD(thread_pool_threads),
    PRINT_CONFIG_FIELD(operator_threads),
    PRINT_CONFIG_FIELD(concurrency),
    PRINT_CONFIG_FIELD(queue_policy),
    PRINT_CONFIG_FIELD(stream_policy),
    PRINT_CONFIG_FIELD(cpu_queue_depth),
    PRINT_CONFIG_FIELD(gpu_queue_depth),
    PRINT_CONFIG_FIELD(set_affinity));
  return os;
}

void PrintTo(const Executor2::Config &cfg, std::ostream *os) {
  print(*os, cfg);
}

namespace test {

class Exec2Test : public::testing::TestWithParam<Executor2::Config> {
 public:
  Exec2Test() {
    config_ = GetParam();
  }

  Executor2::Config config_;
};


TEST_P(Exec2Test, Graph1_CPUOnly) {
  Executor2 exec(config_);
  graph::OpGraph graph = GetTestGraph1();
  exec.Build(graph);
  for (int i = 0; i < 10; i++) {
    exec.Run();
  }
  Workspace ws;
  for (int i = 0; i < 10; i++) {
    ws.Clear();
    exec.Outputs(&ws);
    CheckTestGraph1Results(ws, config_.max_batch_size);
  }
}

TEST_P(Exec2Test, Graph2_CPU2GPU) {
  Executor2 exec(config_);
  graph::OpGraph graph = GetTestGraph2();
  exec.Build(graph);
  for (int i = 0; i < 10; i++) {
    exec.Run();
  }
  Workspace ws;
  for (int i = 0; i < 10; i++) {
    ws.Clear();
    exec.Outputs(&ws);
    CheckTestGraph2Results(ws, config_.max_batch_size);
  }
}

TEST_P(Exec2Test, Graph3_SinkOnly) {
  Executor2 exec(config_);
  graph::OpGraph graph = GetTestGraph3();
  exec.Build(graph);
  for (int i = 0; i < 10; i++) {
    exec.Run();
  }
  Workspace ws;
  int64_t acc = 0;
  int bs = config_.max_batch_size;
  for (int i = 0; i < 10; i++) {
    ws.Clear();
    exec.Outputs(&ws);
    auto *sink = dynamic_cast<SinkOp*>(exec.GetOperator("op1"));
    ASSERT_NE(sink, nullptr);
    int64_t batch_sum = bs * (bs - 1) / 2;
    acc += batch_sum;
    EXPECT_EQ(sink->acc, acc);
  }
}


Executor2::Config MakeCfg(QueueDepthPolicy q, OperatorConcurrency c, StreamPolicy s) {
  Executor2::Config cfg;
  cfg.queue_policy = q;
  cfg.concurrency = c;
  cfg.stream_policy = s;
  cfg.thread_pool_threads = 4;
  cfg.operator_threads = 4;
  cfg.device = 0;
  return cfg;
}

std::vector<Executor2::Config> configs = {
  MakeCfg(QueueDepthPolicy::OutputOnly, OperatorConcurrency::None, StreamPolicy::Single),
  MakeCfg(QueueDepthPolicy::FullyBuffered, OperatorConcurrency::Full, StreamPolicy::Single),
  MakeCfg(QueueDepthPolicy::BackendChange, OperatorConcurrency::Backend, StreamPolicy::PerBackend),
  MakeCfg(QueueDepthPolicy::FullyBuffered, OperatorConcurrency::Full, StreamPolicy::PerOperator),
};

INSTANTIATE_TEST_SUITE_P(Exec2Test, Exec2Test, testing::ValuesIn(configs));


// ===== Error-path / API coverage tests for exec2.cc =====
//
// These drive the public Executor2 API into its guarded error and edge-case
// branches, which the happy-path parametrized tests above never reach.

namespace {

Executor2::Config MakeErrCfg() {
  return MakeCfg(QueueDepthPolicy::OutputOnly, OperatorConcurrency::None, StreamPolicy::Single);
}

}  // namespace

// Building an executor twice must fail - the state can only advance from New.
TEST(Exec2ErrorTest, BuildTwiceThrows) {
  Executor2 exec(MakeErrCfg());
  auto graph = GetTestGraph1();
  exec.Build(graph);
  auto graph2 = GetTestGraph1();
  EXPECT_THROW(exec.Build(graph2), std::logic_error);
}

// Running before the executor is built/started must fail.
TEST(Exec2ErrorTest, RunBeforeBuildThrows) {
  Executor2 exec(MakeErrCfg());
  EXPECT_THROW(exec.Run(), std::runtime_error);
}

// Requesting outputs when none are pending must fail.
TEST(Exec2ErrorTest, OutputsWithoutRunThrows) {
  Executor2 exec(MakeErrCfg());
  auto graph = GetTestGraph1();
  exec.Build(graph);
  Workspace ws;
  EXPECT_THROW(exec.Outputs(&ws), std::out_of_range);
}

// A graph with GPU/mixed nodes requires a device id in the config.
TEST(Exec2ErrorTest, GpuGraphWithoutDeviceThrows) {
  auto cfg = MakeErrCfg();
  cfg.device = std::nullopt;
  Executor2 exec(cfg);
  auto graph = GetTestGraph2();  // contains mixed + gpu nodes
  EXPECT_THROW(exec.Build(graph), std::invalid_argument);
}

// GetExecutorMeta is intentionally unsupported and returns an empty map.
TEST(Exec2ErrorTest, GetExecutorMetaEmpty) {
  Executor2 exec(MakeErrCfg());
  EXPECT_TRUE(exec.GetExecutorMeta().empty());
}

// GetCurrentCheckpoint before any iteration data exists must fail.
TEST(Exec2ErrorTest, GetCheckpointBeforeBuildThrows) {
  Executor2 exec(MakeErrCfg());
  EXPECT_THROW(exec.GetCurrentCheckpoint(), std::runtime_error);
}

// GetCurrentCheckpoint after building without checkpointing enabled must fail.
TEST(Exec2ErrorTest, GetCheckpointWithoutCheckpointingThrows) {
  Executor2 exec(MakeErrCfg());
  auto graph = GetTestGraph1();
  exec.Build(graph);
  EXPECT_THROW(exec.GetCurrentCheckpoint(), std::runtime_error);
}

// Restoring from a checkpoint that has more operator states than the graph must fail.
TEST(Exec2ErrorTest, RestoreCheckpointSuperfluousThrows) {
  Executor2 exec(MakeErrCfg());
  Checkpoint cpt;
  cpt.AddOperator("ghost_op");  // one op, but the (unbuilt) graph has none
  EXPECT_THROW(exec.RestoreStateFromCheckpoint(cpt), std::runtime_error);
}

// Restoring an empty checkpoint on a fresh executor must succeed and initialize
// the iteration data (the !last_iter_data_ branch of RestoreFromCheckpoint).
TEST(Exec2ErrorTest, RestoreEmptyCheckpointInitsIterData) {
  Executor2 exec(MakeErrCfg());
  Checkpoint cpt;
  cpt.SetIterationId(7);
  EXPECT_NO_THROW(exec.RestoreStateFromCheckpoint(cpt));
}


}  // namespace test
}  // namespace exec2
}  // namespace dali
