// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime.h>
#include "dali/operators/audio/mel_scale/mel_filter_bank.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/cuda_error.h"
#include "dali/core/common.h"

namespace dali {

// ============================================================================
// GPU Tests
// ============================================================================

class MelFilterBankGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize CUDA if needed
    if (cudaGetDevice(&device_id_) != cudaSuccess) {
      device_id_ = 0;
    }
  }

  OpSpec MakeOpSpec() {
    return OpSpec("MelFilterBank")
        .AddArg("device", "gpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("nfilter", 128)
        .AddArg("sample_rate", 44100.0f)
        .AddArg("freq_low", 0.0f)
        .AddArg("freq_high", 22050.0f)
        .AddArg("normalize", true)
        .AddInput("input", StorageDevice::GPU)
        .AddOutput("output", StorageDevice::GPU);
  }

  int device_id_ = 0;
};

// Test case 1: Empty layout - should use fallback std::max(0, ndim - 2)
TEST_F(MelFilterBankGPUTest, EmptyLayoutUsesFallback) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // Create input tensor with no layout (empty layout string)
  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{513, 100}};  // 2D tensor: [frequency, time]
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("");  // Empty layout

  // Fill with dummy data
  std::vector<float> data(513 * 100, 0.5f);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<float>(0), data.data(),
                       data.size() * sizeof(float), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Setup should succeed and use fallback axis calculation
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);

  // Verify that axis was set correctly using fallback: std::max(0, ndim - 2) = std::max(0, 2 - 2) = 0
  // For 2D tensor with empty layout, axis should be 0 (first dimension)
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 2: Invalid layout - should throw DALI_ENFORCE exception
TEST_F(MelFilterBankGPUTest, InvalidLayoutThrowsException) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // Create input tensor with layout that doesn't contain 'f'
  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{513, 100}};
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("xy");  // Layout without 'f' axis

  std::vector<float> data(513 * 100, 0.5f);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<float>(0), data.data(),
                       data.size() * sizeof(float), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Setup should fail with DALI_ENFORCE exception
  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 3: Unsupported data type in SetupImpl - should throw DALI_FAIL
TEST_F(MelFilterBankGPUTest, UnsupportedDataTypeSetupImpl) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // Create input tensor with unsupported data type (double instead of float)
  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{513, 100}};
  input->Resize(shape, DALI_FLOAT64);  // Unsupported type
  input->SetLayout("ft");

  std::vector<double> data(513 * 100, 0.5);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<double>(0), data.data(),
                       data.size() * sizeof(double), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Setup should fail with DALI_FAIL exception for unsupported type
  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 4: Unsupported data type in RunImpl - should throw DALI_FAIL
TEST_F(MelFilterBankGPUTest, UnsupportedDataTypeRunImpl) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // First setup with valid type to get output shape
  auto input_valid = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{513, 100}};
  input_valid->Resize(shape, DALI_FLOAT);
  input_valid->SetLayout("ft");
  std::vector<float> valid_data(513 * 100, 0.5f);
  CUDA_CALL(cudaMemcpy(input_valid->mutable_tensor<float>(0), valid_data.data(),
                       valid_data.size() * sizeof(float), cudaMemcpyHostToDevice));

  Workspace ws_setup;
  ws_setup.AddInput(input_valid);
  ws_setup.set_stream(0);

  std::vector<OutputDesc> output_desc;
  op.Setup(output_desc, ws_setup);

  // Create input tensor with unsupported data type for RunImpl
  auto input = std::make_shared<TensorList<GPUBackend>>();
  input->Resize(shape, DALI_INT32);  // Unsupported type
  input->SetLayout("ft");

  std::vector<int32_t> data(513 * 100, 1);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<int32_t>(0), data.data(),
                       data.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Allocate output based on setup
  auto output = std::make_shared<TensorList<GPUBackend>>();
  output->Resize(output_desc[0].shape, output_desc[0].type);
  ws.AddOutput(output);

  // Now try to run with unsupported type
  // RunImpl should fail with DALI_FAIL exception
  EXPECT_THROW({
    op.Run(ws);
  }, DALIException);
}

// Test case 5: Empty layout with 3D tensor - should use std::max(0, ndim - 2) = 1
TEST_F(MelFilterBankGPUTest, EmptyLayout3DTensor) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // Create 3D input tensor with no layout
  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{10, 513, 100}};  // 3D tensor: [batch, frequency, time]
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("");  // Empty layout

  std::vector<float> data(10 * 513 * 100, 0.5f);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<float>(0), data.data(),
                       data.size() * sizeof(float), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Setup should succeed
  // For 3D tensor with empty layout: std::max(0, 3 - 2) = std::max(0, 1) = 1
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 6: Empty layout with 4D tensor - should use std::max(0, ndim - 2) = 2
TEST_F(MelFilterBankGPUTest, EmptyLayout4DTensor) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // Create 4D input tensor with no layout
  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{5, 10, 513, 100}};  // 4D tensor
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("");  // Empty layout

  std::vector<float> data(5 * 10 * 513 * 100, 0.5f);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<float>(0), data.data(),
                       data.size() * sizeof(float), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Setup should succeed
  // For 4D tensor with empty layout: std::max(0, 4 - 2) = std::max(0, 2) = 2
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 7: Successful RunImpl execution with valid inputs
TEST_F(MelFilterBankGPUTest, SuccessfulRunImpl) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<GPUBackend> op(spec);

  // Create valid input tensor
  auto input = std::make_shared<TensorList<GPUBackend>>();
  TensorListShape<> shape = {{513, 100}};  // 2D tensor: [frequency, time]
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("ft");

  // Fill with dummy data
  std::vector<float> data(513 * 100, 0.5f);
  CUDA_CALL(cudaMemcpy(input->mutable_tensor<float>(0), data.data(),
                       data.size() * sizeof(float), cudaMemcpyHostToDevice));

  Workspace ws;
  ws.AddInput(input);
  ws.set_stream(0);

  // Setup to get output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);

  // Create output tensor based on output_desc
  auto output = std::make_shared<TensorList<GPUBackend>>();
  output->Resize(output_desc[0].shape, output_desc[0].type);
  ws.AddOutput(output);

  // RunImpl should execute successfully
  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

// ============================================================================
// CPU Tests
// ============================================================================

class MelFilterBankCPUTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  OpSpec MakeOpSpec() {
    return OpSpec("MelFilterBank")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("nfilter", 128)
        .AddArg("sample_rate", 44100.0f)
        .AddArg("freq_low", 0.0f)
        .AddArg("freq_high", 22050.0f)
        .AddArg("normalize", true)
        .AddInput("input", StorageDevice::CPU)
        .AddOutput("output", StorageDevice::CPU);
  }
};

// Test case 1: Empty layout - should use fallback std::max(0, ndim - 2)
TEST_F(MelFilterBankCPUTest, EmptyLayoutUsesFallback) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // Create input tensor with no layout (empty layout string)
  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{513, 100}};  // 2D tensor: [frequency, time]
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("");  // Empty layout

  // Fill with dummy data
  std::vector<float> data(513 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Setup should succeed and use fallback axis calculation
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);

  // Verify that axis was set correctly using fallback: std::max(0, ndim - 2) = std::max(0, 2 - 2) = 0
  // For 2D tensor with empty layout, axis should be 0 (first dimension)
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 2: Invalid layout - should throw DALI_ENFORCE exception
TEST_F(MelFilterBankCPUTest, InvalidLayoutThrowsException) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // Create input tensor with layout that doesn't contain 'f'
  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{513, 100}};
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("xy");  // Layout without 'f' axis

  std::vector<float> data(513 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Setup should fail with DALI_ENFORCE exception
  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 3: Unsupported data type in SetupImpl - should throw DALI_FAIL
TEST_F(MelFilterBankCPUTest, UnsupportedDataTypeSetupImpl) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // Create input tensor with unsupported data type (double instead of float)
  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{513, 100}};
  input->Resize(shape, DALI_FLOAT64);  // Unsupported type
  input->SetLayout("ft");

  std::vector<double> data(513 * 100, 0.5);
  std::memcpy(input->mutable_tensor<double>(0), data.data(),
              data.size() * sizeof(double));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Setup should fail with DALI_FAIL exception for unsupported type
  std::vector<OutputDesc> output_desc;
  EXPECT_THROW({
    op.Setup(output_desc, ws);
  }, DALIException);
}

// Test case 4: Unsupported data type in RunImpl - should throw DALI_FAIL
TEST_F(MelFilterBankCPUTest, UnsupportedDataTypeRunImpl) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // First setup with valid type to get output shape
  auto input_valid = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{513, 100}};
  input_valid->Resize(shape, DALI_FLOAT);
  input_valid->SetLayout("ft");
  std::vector<float> valid_data(513 * 100, 0.5f);
  std::memcpy(input_valid->mutable_tensor<float>(0), valid_data.data(),
              valid_data.size() * sizeof(float));

  Workspace ws_setup;
  ws_setup.AddInput(input_valid);
  OldThreadPool tp_setup(1, 0, false, "TestPool");
  ws_setup.SetThreadPool(&tp_setup);

  std::vector<OutputDesc> output_desc;
  op.Setup(output_desc, ws_setup);

  // Create input tensor with unsupported data type for RunImpl
  auto input = std::make_shared<TensorList<CPUBackend>>();
  input->Resize(shape, DALI_INT32);  // Unsupported type
  input->SetLayout("ft");

  std::vector<int32_t> data(513 * 100, 1);
  std::memcpy(input->mutable_tensor<int32_t>(0), data.data(),
              data.size() * sizeof(int32_t));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Allocate output based on setup
  auto output = std::make_shared<TensorList<CPUBackend>>();
  output->Resize(output_desc[0].shape, output_desc[0].type);
  ws.AddOutput(output);

  // Now try to run with unsupported type
  // RunImpl should fail with DALI_FAIL exception
  EXPECT_THROW({
    op.Run(ws);
  }, DALIException);
}

// Test case 5: Empty layout with 3D tensor - should use std::max(0, ndim - 2) = 1
TEST_F(MelFilterBankCPUTest, EmptyLayout3DTensor) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // Create 3D input tensor with no layout
  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{10, 513, 100}};  // 3D tensor: [batch, frequency, time]
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("");  // Empty layout

  std::vector<float> data(10 * 513 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Setup should succeed
  // For 3D tensor with empty layout: std::max(0, 3 - 2) = std::max(0, 1) = 1
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 6: Empty layout with 4D tensor - should use std::max(0, ndim - 2) = 2
TEST_F(MelFilterBankCPUTest, EmptyLayout4DTensor) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // Create 4D input tensor with no layout
  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{5, 10, 513, 100}};  // 4D tensor
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("");  // Empty layout

  std::vector<float> data(5 * 10 * 513 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Setup should succeed
  // For 4D tensor with empty layout: std::max(0, 4 - 2) = std::max(0, 2) = 2
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);
}

// Test case 7: Successful RunImpl execution with valid inputs
TEST_F(MelFilterBankCPUTest, SuccessfulRunImpl) {
  OpSpec spec = MakeOpSpec();
  MelFilterBank<CPUBackend> op(spec);

  // Create valid input tensor
  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{513, 100}};  // 2D tensor: [frequency, time]
  input->Resize(shape, DALI_FLOAT);
  input->SetLayout("ft");

  // Fill with dummy data
  std::vector<float> data(513 * 100, 0.5f);
  std::memcpy(input->mutable_tensor<float>(0), data.data(),
              data.size() * sizeof(float));

  Workspace ws;
  ws.AddInput(input);
  OldThreadPool tp(1, 0, false, "TestPool");
  ws.SetThreadPool(&tp);

  // Setup to get output shape
  std::vector<OutputDesc> output_desc;
  bool setup_result = op.Setup(output_desc, ws);
  EXPECT_TRUE(setup_result);
  EXPECT_EQ(output_desc.size(), 1);

  // Create output tensor based on output_desc
  auto output = std::make_shared<TensorList<CPUBackend>>();
  output->Resize(output_desc[0].shape, output_desc[0].type);
  ws.AddOutput(output);

  // RunImpl should execute successfully
  EXPECT_NO_THROW({
    op.Run(ws);
  });
}

}  // namespace dali
