// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <memory>
#include <cstring>
#include "dali/operators/audio/mfcc/mfcc.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/common.h"

namespace dali {
namespace detail {
namespace test {

void check_lifter_coeffs(span<const float> coeffs, double lifter, int64_t length) {
  ASSERT_EQ(length, coeffs.size());
  auto coeffs_data = coeffs.data();
  ASSERT_NE(nullptr, coeffs_data);
  for (int64_t i = 0; i < length; i++) {
    float expected = 1.0 + 0.5 * lifter * std::sin((i + 1) * M_PI / lifter);
    EXPECT_NEAR(expected, coeffs_data[i], 1e-4);
  }
}

TEST(LifterCoeffsCPU, correctness) {
  LifterCoeffs<CPUBackend> coeffs;

  auto lifter = 0.0f;
  coeffs.Calculate(10, lifter);
  ASSERT_TRUE(coeffs.empty());

  lifter = 1.234f;
  coeffs.Calculate(10, lifter);
  check_lifter_coeffs(make_cspan(coeffs), lifter, 10);

  coeffs.Calculate(20, lifter);
  check_lifter_coeffs(make_cspan(coeffs), lifter, 20);

  lifter = 2.234f;
  coeffs.Calculate(10, lifter);
  check_lifter_coeffs(make_cspan(coeffs), lifter, 10);

  coeffs.Calculate(5, lifter);
  check_lifter_coeffs(make_cspan(coeffs), lifter, 10);
}

TEST(LifterCoeffsGPU, correctness) {
  LifterCoeffs<GPUBackend> coeffs;
  std::vector<float> coeffs_cpu;

  auto lifter = 0.0f;
  coeffs.Calculate(10, lifter);
  ASSERT_TRUE(coeffs.empty());

  lifter = 1.234f;
  coeffs.Calculate(10, lifter);
  coeffs_cpu.resize(coeffs.size());
  CUDA_CALL(cudaMemcpy(coeffs_cpu.data(), coeffs.data(),
                       coeffs.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_lifter_coeffs(make_cspan(coeffs_cpu), lifter, coeffs.size());

  coeffs.Calculate(20, lifter);
  coeffs_cpu.resize(coeffs.size());
  CUDA_CALL(cudaMemcpy(coeffs_cpu.data(), coeffs.data(),
                       coeffs.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_lifter_coeffs(make_cspan(coeffs_cpu), lifter, coeffs.size());

  coeffs.Calculate(10, lifter);
  coeffs_cpu.resize(coeffs.size());
  CUDA_CALL(cudaMemcpy(coeffs_cpu.data(), coeffs.data(),
                       coeffs.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_lifter_coeffs(make_cspan(coeffs_cpu), lifter, coeffs.size());

  coeffs.Calculate(5, lifter);
  coeffs_cpu.resize(coeffs.size());
  CUDA_CALL(cudaMemcpy(coeffs_cpu.data(), coeffs.data(),
                       coeffs.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_lifter_coeffs(make_cspan(coeffs_cpu), lifter, coeffs.size());
}

}  // namespace test
}  // namespace detail

// ============================================================================
// MFCC Operator Tests
// ============================================================================

class MFCCCPUTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  OpSpec MakeOpSpec() {
    return OpSpec("MFCC")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("n_mfcc", 20)
        .AddArg("dct_type", 2)
        .AddArg("normalize", false)
        .AddArg("axis", 0)
        .AddArg("lifter", 0.0f)
        .AddInput("input", StorageDevice::CPU)
        .AddOutput("output", StorageDevice::CPU);
  }

  // Sets up the workspace with input and an empty output.
  // MFCC::SetupImpl accesses ws.Output<CPUBackend>(0), so an empty output
  // must exist before calling Setup, matching the eager operator pattern.
  void SetupWorkspace(Workspace& ws,
                      std::shared_ptr<TensorList<CPUBackend>> input,
                      ThreadPool& tp) {
    ws.AddInput(input);
    auto output = std::make_shared<TensorList<CPUBackend>>();
    ws.AddOutput(output);
    ws.SetBatchSizes(input->num_samples());
    ws.SetThreadPool(&tp);
  }
};

// Test case 1: Successful SetupImpl with 2D tensor
TEST_F(MFCCCPUTest, SetupImpl2DTensor) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{128, 100}};  // 2D tensor: [mel_bins, time]
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 2: Successful SetupImpl with 3D tensor
TEST_F(MFCCCPUTest, SetupImpl3DTensor) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{10, 128, 100}};  // 3D tensor: [batch, mel_bins, time]
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(10 * 128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 3: Successful SetupImpl with 4D tensor
TEST_F(MFCCCPUTest, SetupImpl4DTensor) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{5, 10, 128, 100}};  // 4D tensor
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(5 * 10 * 128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 4: Invalid axis - should throw DALI_ENFORCE exception
TEST_F(MFCCCPUTest, InvalidAxisThrowsException) {
  OpSpec spec = OpSpec("MFCC")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("n_mfcc", 20)
      .AddArg("dct_type", 2)
      .AddArg("normalize", false)
      .AddArg("axis", 5)  // Invalid axis for 2D tensor
      .AddArg("lifter", 0.0f)
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU);
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{128, 100}};
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 5: Unsupported data type in SetupImpl - should throw DALI_FAIL
TEST_F(MFCCCPUTest, UnsupportedDataTypeSetupImpl) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{128, 100}};
  input->Resize(shape, DALI_FLOAT64);  // Unsupported type

  std::vector<double> data(128 * 100, 0.5);
  std::memcpy(input->mutable_tensor<double>(0), data.data(),
              data.size() * sizeof(double));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 6: Unsupported number of dimensions - should throw DALI_FAIL
TEST_F(MFCCCPUTest, UnsupportedDimensionsSetupImpl) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{5, 10, 20, 30, 40}};  // 5D tensor (unsupported)
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(5 * 10 * 20 * 30 * 40, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 7: Successful RunImpl with 2D tensor
TEST_F(MFCCCPUTest, SuccessfulRunImpl2D) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{128, 100}};
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  // Setup determines output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  // Resize the output based on setup results
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

// Test case 8: Successful RunImpl with liftering enabled (2D tensor)
TEST_F(MFCCCPUTest, SuccessfulRunImplWithLifter) {
  OpSpec spec = OpSpec("MFCC")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("n_mfcc", 20)
      .AddArg("dct_type", 2)
      .AddArg("normalize", false)
      .AddArg("axis", 0)
      .AddArg("lifter", 22.0f)  // Enable liftering
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU);
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{128, 100}};
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  // Setup determines output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  // Resize the output based on setup results
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

// Test case 9: Successful RunImpl with liftering enabled (3D tensor)
TEST_F(MFCCCPUTest, SuccessfulRunImplWithLifter3D) {
  OpSpec spec = OpSpec("MFCC")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("n_mfcc", 20)
      .AddArg("dct_type", 2)
      .AddArg("normalize", false)
      .AddArg("axis", 0)
      .AddArg("lifter", 22.0f)  // Enable liftering
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU);
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{10, 128, 100}};  // 3D tensor: [batch, mel_bins, time]
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(10 * 128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  // Setup determines output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  // Resize the output based on setup results
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

// Test case 10: Successful RunImpl with liftering enabled (4D tensor)
TEST_F(MFCCCPUTest, SuccessfulRunImplWithLifter4D) {
  OpSpec spec = OpSpec("MFCC")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("n_mfcc", 20)
      .AddArg("dct_type", 2)
      .AddArg("normalize", false)
      .AddArg("axis", 0)
      .AddArg("lifter", 22.0f)  // Enable liftering
      .AddInput("input", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU);
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{5, 10, 128, 100}};  // 4D tensor
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(5 * 10 * 128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  // Setup determines output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  // Resize the output based on setup results
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

// Test case 11: Successful RunImpl with 3D tensor
TEST_F(MFCCCPUTest, SuccessfulRunImpl3D) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{10, 128, 100}};
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(10 * 128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  // Setup determines output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  // Resize the output based on setup results
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

// Test case 12: Successful RunImpl with 4D tensor
TEST_F(MFCCCPUTest, SuccessfulRunImpl4D) {
  OpSpec spec = MakeOpSpec();
  MFCC<CPUBackend> op(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{5, 10, 128, 100}};
  input->Resize(shape, DALI_FLOAT);

  std::vector<float> data(5 * 10 * 128 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  // Setup determines output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  ASSERT_TRUE(setup_result);

  // Resize the output based on setup results
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

}  // namespace dali
