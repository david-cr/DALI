// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <random>
#include <memory>
#include <cstring>
#include <cuda_runtime.h>
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/common.h"

namespace dali {
namespace testing {

class NonsilenceOpTest : public ::testing::Test {
 protected:
  void SetUp() final {
  }

  int window_size_ = 3;
  std::vector<float> input_{0, 0, 0, 0, 1000, -1000, 1000, 0, 0, 0};
  std::vector<float> mms_ref_{0,          0,       0,          0,          333333.344,
                              666666.688, 1000000, 666666.688, 333333.344, 0};
  std::pair<int, int> nonsilence_region_ref_{2, 5};
  int buffer_length_ = 10;
  TensorShape<1> shape_ = {buffer_length_};
};


TEST_F(NonsilenceOpTest, UnderlyingKernelsTest) {
  auto in = make_tensor_cpu(reinterpret_cast<const float *>(this->input_.data()), this->shape_);
  kernels::signal::MovingMeanSquareArgs mms_args{this->window_size_, -1};
  Tensor<CPUBackend> intermediate_buffer;
  detail::RunKernel(in, intermediate_buffer, mms_args);

  for (size_t i = 0; i < this->mms_ref_.size(); i++) {
    EXPECT_FLOAT_EQ(this->mms_ref_[i], intermediate_buffer.data<float>()[i]);
  }
}


TEST_F(NonsilenceOpTest, DetectNonsilenceRegionTest) {
  auto in = make_tensor_cpu(reinterpret_cast<const float *>(this->input_.data()), this->shape_);
  Tensor<CPUBackend> intermediate_buffer;
  auto nonsilence_region = detail::DetectNonsilenceRegion<float>(intermediate_buffer,
                                                                 {in, 0, 1.f, false,
                                                                  this->window_size_, -1});
  // It's impossible to figure out where within the window the nonsilent region begins and ends
  EXPECT_PRED2(EqualEps(this->window_size_),
               nonsilence_region.first, nonsilence_region_ref_.first);
  EXPECT_PRED2(EqualEps(this->window_size_),
               nonsilence_region.second, nonsilence_region_ref_.second);
}


TEST_F(NonsilenceOpTest, LeadTrailThreshTest) {
  std::vector<float> t0 = {0, 0, 0, 0, 0, 1.5, -100, 1.5};
  using detail::LeadTrailThresh;

  EXPECT_EQ(LeadTrailThresh(make_cspan(t0), .5f), std::make_pair(5_i64, 3_i64));

  std::vector<float> t1 = {1.5, -100, 1.5, 0, 0, 0, 0};
  EXPECT_EQ(LeadTrailThresh(make_cspan(t1), .5f), std::make_pair(0_i64, 3_i64));

  std::vector<float> t2 = {0, 0, 0, 0, 0, 1.5, -100, -100, 1.5, 0, 0, 0, 0};
  EXPECT_EQ(LeadTrailThresh(make_cspan(t2), 1.5f), std::make_pair(5_i64, 4_i64));

  std::vector<int> t3 = {23, 62, 46, 12, 53};
  EXPECT_EQ(LeadTrailThresh(make_cspan(t3), 100), std::make_pair(0_i64, 0_i64));

  std::vector<int64_t> t4 = {623, 45, 62, 46, 23};
  EXPECT_EQ(LeadTrailThresh(make_cspan(t4), 10L), std::make_pair(0_i64, 5_i64));

  std::vector<int> t5 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(LeadTrailThresh(make_cspan(t5), 1), std::make_pair(0_i64, 0_i64));

  std::vector<int> t6 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(LeadTrailThresh(make_cspan(t6), 0), std::make_pair(0_i64, 12_i64));
}

// ============================================================================
// NonsilenceOperator Tests
// ============================================================================

class NonsilenceOperatorCPUTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  OpSpec MakeOpSpec() {
    return OpSpec("NonsilentRegion")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("cutoff_db", -60.0f)
        .AddArg("window_length", 2048)
        // reference_power not set - will use maximum power of signal
        .AddArg("reset_interval", 8192)
        .AddInput("input", StorageDevice::CPU)
        .AddOutput("begin", StorageDevice::CPU)
        .AddOutput("length", StorageDevice::CPU);
  }

  // Sets up the workspace with input and empty outputs
  void SetupWorkspace(Workspace& ws,
                      std::shared_ptr<TensorList<CPUBackend>> input,
                      ThreadPool& tp) {
    ws.AddInput(input);
    auto output_begin = std::make_shared<TensorList<CPUBackend>>();
    auto output_length = std::make_shared<TensorList<CPUBackend>>();
    ws.AddOutput(output_begin);
    ws.AddOutput(output_length);
    ws.SetBatchSizes(input->num_samples());
    ws.SetThreadPool(&tp);
  }

  // Helper to create test data: silence at start/end, signal in middle
  template<typename T>
  std::vector<T> CreateTestData(int length) {
    std::vector<T> data(length, static_cast<T>(0));
    // Add some signal in the middle
    int signal_start = length / 4;
    int signal_end = 3 * length / 4;
    for (int i = signal_start; i < signal_end; i++) {
      data[i] = static_cast<T>(1000);
    }
    return data;
  }
};

// Test case: RunImpl with int8_t input
TEST_F(NonsilenceOperatorCPUTest, RunImplInt8) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_INT8);

  std::vector<int8_t> data = CreateTestData<int8_t>(100);
  std::memcpy(input->mutable_tensor<int8_t>(0), data.data(),
              data.size() * sizeof(int8_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with uint16_t input
TEST_F(NonsilenceOperatorCPUTest, RunImplUInt16) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_UINT16);

  std::vector<uint16_t> data = CreateTestData<uint16_t>(100);
  std::memcpy(input->mutable_tensor<uint16_t>(0), data.data(),
              data.size() * sizeof(uint16_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with uint32_t input
TEST_F(NonsilenceOperatorCPUTest, RunImplUInt32) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_UINT32);

  std::vector<uint32_t> data = CreateTestData<uint32_t>(100);
  std::memcpy(input->mutable_tensor<uint32_t>(0), data.data(),
              data.size() * sizeof(uint32_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with int32_t input
TEST_F(NonsilenceOperatorCPUTest, RunImplInt32) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_INT32);

  std::vector<int32_t> data = CreateTestData<int32_t>(100);
  std::memcpy(input->mutable_tensor<int32_t>(0), data.data(),
              data.size() * sizeof(int32_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with uint64_t input
TEST_F(NonsilenceOperatorCPUTest, RunImplUInt64) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_UINT64);

  std::vector<uint64_t> data = CreateTestData<uint64_t>(100);
  std::memcpy(input->mutable_tensor<uint64_t>(0), data.data(),
              data.size() * sizeof(uint64_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with int64_t input
TEST_F(NonsilenceOperatorCPUTest, RunImplInt64) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_INT64);

  std::vector<int64_t> data = CreateTestData<int64_t>(100);
  std::memcpy(input->mutable_tensor<int64_t>(0), data.data(),
              data.size() * sizeof(int64_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with unsupported type - should throw exception
TEST_F(NonsilenceOperatorCPUTest, UnsupportedTypeThrowsException) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_FLOAT64);  // Unsupported type

  std::vector<double> data(100, 0.5);
  std::memcpy(input->mutable_tensor<double>(0), data.data(),
              data.size() * sizeof(double));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_THROW({
    op->Run(ws);
  }, DALIException);
}

// Test case: RunImpl with reference_power explicitly set
// This covers the true branch of if (!reference_max_) in RunImplTyped
TEST_F(NonsilenceOperatorCPUTest, RunImplWithReferencePower) {
  OpSpec spec = MakeOpSpec();
  spec.AddArg("reference_power", 1.0f);  // Explicitly set reference_power
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data = CreateTestData<float>(100);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with batch size > 1 (multiple samples)
TEST_F(NonsilenceOperatorCPUTest, RunImplBatchSizeGreaterThanOne) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}, {200}, {150}};  // 3 samples in batch
  input->Resize(shape, DALI_FLOAT);

  // Fill each sample with test data
  for (int i = 0; i < 3; i++) {
    std::vector<float> data = CreateTestData<float>(shape[i][0]);
    std::memcpy(input->mutable_tensor<float>(i), data.data(),
                data.size() * sizeof(float));
  }

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with reference_power and batch size > 1
TEST_F(NonsilenceOperatorCPUTest, RunImplWithReferencePowerBatch) {
  OpSpec spec = MakeOpSpec();
  spec.AddArg("reference_power", 1.0f);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}, {200}};  // 2 samples
  input->Resize(shape, DALI_INT16);

  for (int i = 0; i < 2; i++) {
    std::vector<int16_t> data = CreateTestData<int16_t>(shape[i][0]);
    std::memcpy(input->mutable_tensor<int16_t>(i), data.data(),
                data.size() * sizeof(int16_t));
  }

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with window_length smaller than input length
TEST_F(NonsilenceOperatorCPUTest, RunImplSmallWindowLength) {
  OpSpec spec = OpSpec("NonsilentRegion")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("cutoff_db", -60.0f)
      .AddArg("window_length", 10)  // Smaller window
      .AddArg("reset_interval", 8192)
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("begin", StorageDevice::CPU)
      .AddOutput("length", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_UINT8);

  std::vector<uint8_t> data = CreateTestData<uint8_t>(100);
  std::memcpy(input->mutable_tensor<uint8_t>(0), data.data(),
              data.size() * sizeof(uint8_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with window_length larger than input size
// This tests std::min when window_length_ > input.num_elements()
TEST_F(NonsilenceOperatorCPUTest, RunImplLargeWindowLength) {
  OpSpec spec = OpSpec("NonsilentRegion")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("cutoff_db", -60.0f)
      .AddArg("window_length", 5000)  // Larger than input size
      .AddArg("reset_interval", 8192)
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("begin", StorageDevice::CPU)
      .AddOutput("length", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};  // Input smaller than window_length
  input->Resize(shape, DALI_INT16);

  std::vector<int16_t> data = CreateTestData<int16_t>(100);
  std::memcpy(input->mutable_tensor<int16_t>(0), data.data(),
              data.size() * sizeof(int16_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with multiple threads
// This ensures intermediate_buffers_ loop is fully covered
TEST_F(NonsilenceOperatorCPUTest, RunImplMultipleThreads) {
  OpSpec spec = OpSpec("NonsilentRegion")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 4)  // Multiple threads
      .AddArg("max_batch_size", 32)
      .AddArg("cutoff_db", -60.0f)
      .AddArg("window_length", 2048)
      .AddArg("reset_interval", 8192)
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("begin", StorageDevice::CPU)
      .AddOutput("length", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}, {200}, {150}};  // Multiple samples
  input->Resize(shape, DALI_FLOAT);

  for (int i = 0; i < 3; i++) {
    std::vector<float> data = CreateTestData<float>(shape[i][0]);
    std::memcpy(input->mutable_tensor<float>(i), data.data(),
                data.size() * sizeof(float));
  }

  Workspace ws;
  ThreadPool tp(4, 0, false, "TestPool");  // Match num_threads
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with reset_interval = -1 (for non-floating point types)
TEST_F(NonsilenceOperatorCPUTest, RunImplResetIntervalMinusOne) {
  OpSpec spec = OpSpec("NonsilentRegion")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("cutoff_db", -60.0f)
      .AddArg("window_length", 2048)
      .AddArg("reset_interval", -1)  // -1 means no reset for non-floating point
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("begin", StorageDevice::CPU)
      .AddOutput("length", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_UINT16);  // Non-floating point type

  std::vector<uint16_t> data = CreateTestData<uint16_t>(100);
  std::memcpy(input->mutable_tensor<uint16_t>(0), data.data(),
              data.size() * sizeof(uint16_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// Test case: RunImpl with very small input
TEST_F(NonsilenceOperatorCPUTest, RunImplVerySmallInput) {
  OpSpec spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{5}};  // Very small input
  input->Resize(shape, DALI_UINT8);

  std::vector<uint8_t> data = CreateTestData<uint8_t>(5);
  std::memcpy(input->mutable_tensor<uint8_t>(0), data.data(),
              data.size() * sizeof(uint8_t));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op->Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<CPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW({
    op->Run(ws);
  });
}

// ============================================================================
// NonsilenceOperator GPU Tests
// ============================================================================

class NonsilenceOperatorGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  OpSpec MakeGPUOpSpec() {
    return OpSpec("NonsilentRegion")
        .AddArg("device", "gpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("cutoff_db", -60.0f)
        .AddArg("window_length", 2048)
        .AddArg("reset_interval", 8192)
        .AddInput("input", StorageDevice::GPU)
        .AddOutput("begin", StorageDevice::GPU)
        .AddOutput("length", StorageDevice::GPU);
  }

  void SetupGPUWorkspace(Workspace& ws,
                         std::shared_ptr<TensorList<GPUBackend>> input) {
    ws.AddInput(input);
    auto output_begin = std::make_shared<TensorList<GPUBackend>>();
    auto output_length = std::make_shared<TensorList<GPUBackend>>();
    ws.AddOutput(output_begin);
    ws.AddOutput(output_length);
    ws.SetBatchSizes(input->num_samples());
    ws.set_stream(0);
  }

  // Helper to create GPU tensor from host data
  template<typename T>
  std::shared_ptr<TensorList<GPUBackend>> CreateGPUInput(
      const TensorListShape<>& shape, DALIDataType dtype) {
    auto input = std::make_shared<TensorList<GPUBackend>>();
    input->Resize(shape, dtype);
    for (int i = 0; i < shape.num_samples(); i++) {
      int length = shape[i][0];
      std::vector<T> data(length, static_cast<T>(0));
      // Add some signal in the middle
      int signal_start = length / 4;
      int signal_end = 3 * length / 4;
      for (int j = signal_start; j < signal_end; j++) {
        data[j] = static_cast<T>(1000);
      }
      CUDA_CALL(cudaMemcpy(input->mutable_tensor<T>(i), data.data(),
                           data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return input;
  }
};

// GPU Test: RunImpl with uint8_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplUInt8) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<uint8_t>(shape, DALI_UINT8);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with int8_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplInt8) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<int8_t>(shape, DALI_INT8);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with uint16_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplUInt16) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<uint16_t>(shape, DALI_UINT16);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with int16_t input (already covered but included for completeness)
TEST_F(NonsilenceOperatorGPUTest, RunImplInt16) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<int16_t>(shape, DALI_INT16);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with uint32_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplUInt32) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<uint32_t>(shape, DALI_UINT32);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with int32_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplInt32) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<int32_t>(shape, DALI_INT32);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with uint64_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplUInt64) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<uint64_t>(shape, DALI_UINT64);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with int64_t input
TEST_F(NonsilenceOperatorGPUTest, RunImplInt64) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<int64_t>(shape, DALI_INT64);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with float input (already covered but included for completeness)
TEST_F(NonsilenceOperatorGPUTest, RunImplFloat) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<float>(shape, DALI_FLOAT);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with unsupported type - should throw exception
TEST_F(NonsilenceOperatorGPUTest, UnsupportedTypeThrowsException) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_FLOAT64);  // Unsupported type

  std::vector<double> data(100, 0.5);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<double>(0), data.data(),
                       data.size() * sizeof(double), cudaMemcpyHostToDevice));

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_THROW(op->Run(ws), DALIException);
}

// GPU Test: RunImpl with reference_power explicitly set
// Covers the !reference_max_ branch in CalcNonsilentRegion
TEST_F(NonsilenceOperatorGPUTest, RunImplWithReferencePower) {
  OpSpec spec = MakeGPUOpSpec();
  spec.AddArg("reference_power", 1.0f);
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}};
  auto input = CreateGPUInput<float>(shape, DALI_FLOAT);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// GPU Test: RunImpl with batch size > 1
TEST_F(NonsilenceOperatorGPUTest, RunImplBatchSizeGreaterThanOne) {
  OpSpec spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{100}, {200}, {150}};  // 3 samples
  auto input = CreateGPUInput<float>(shape, DALI_FLOAT);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ASSERT_EQ(output_desc.size(), 2);

  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);
  ws.Output<GPUBackend>(1).Resize(output_desc[1].shape, output_desc[1].type);

  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(0));
}

}  // namespace testing
}  // namespace dali
