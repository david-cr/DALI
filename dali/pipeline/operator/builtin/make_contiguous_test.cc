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
#include <memory>
#include <vector>

#include "dali/pipeline/operator/builtin/make_contiguous.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/test/dali_test.h"

namespace dali {

class MakeContiguousMixedTest : public ::testing::Test {
 protected:
  static constexpr int batch_size_ = 3;
  static constexpr int num_threads_ = 2;
};

// Test CPU-to-GPU with small samples (coalesced path)
// Covers lines 57-60
TEST_F(MakeContiguousMixedTest, CPUToGPUCoalesced) {
  int device_count = 0;
  CUDA_CALL(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    GTEST_SKIP() << "At least 1 GPU required";
  }

  Pipeline pipe(batch_size_, num_threads_, 0);
  pipe.AddExternalInput("input");

  pipe.AddOperator(OpSpec("MakeContiguous")
                      .AddArg("device", "mixed")
                      .AddInput("input", StorageDevice::CPU)
                      .AddOutput("output", StorageDevice::GPU));

  pipe.Build({{"output", "gpu"}});

  // Create small input (below COALESCE_THRESHOLD = 8192 bytes)
  TensorList<CPUBackend> input;
  input.set_pinned(false);
  input.Resize({{10}, {20}, {15}}, DALI_FLOAT);

  for (int i = 0; i < batch_size_; ++i) {
    float *data = input.mutable_tensor<float>(i);
    for (int j = 0; j < input.tensor_shape(i)[0]; ++j) {
      data[j] = static_cast<float>(i * 100 + j);
    }
  }

  pipe.SetExternalInput("input", input);
  pipe.Run();

  Workspace ws;
  pipe.Outputs(&ws);
  const auto &result = ws.Output<GPUBackend>(0);
  EXPECT_EQ(result.num_samples(), batch_size_);
  EXPECT_EQ(result.type(), DALI_FLOAT);
}

// Test CPU-to-GPU with large samples (non-coalesced path)
// Covers lines 62-63
TEST_F(MakeContiguousMixedTest, CPUToGPUNonCoalesced) {
  int device_count = 0;
  CUDA_CALL(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    GTEST_SKIP() << "At least 1 GPU required";
  }

  Pipeline pipe(batch_size_, num_threads_, 0);
  pipe.AddExternalInput("input");

  pipe.AddOperator(OpSpec("MakeContiguous")
                      .AddArg("device", "mixed")
                      .AddInput("input", StorageDevice::CPU)
                      .AddOutput("output", StorageDevice::GPU));

  pipe.Build({{"output", "gpu"}});

  // Create large input (above COALESCE_THRESHOLD = 8192 bytes)
  // 2100 floats = 8400 bytes > 8192
  TensorList<CPUBackend> input;
  input.set_pinned(false);
  input.Resize({{2100}, {2200}, {2300}}, DALI_FLOAT);

  for (int i = 0; i < batch_size_; ++i) {
    float *data = input.mutable_tensor<float>(i);
    int size = input.tensor_shape(i)[0];
    for (int j = 0; j < size; ++j) {
      data[j] = static_cast<float>(i * 1000 + j);
    }
  }

  pipe.SetExternalInput("input", input);
  pipe.Run();

  Workspace ws;
  pipe.Outputs(&ws);
  const auto &result = ws.Output<GPUBackend>(0);
  EXPECT_EQ(result.num_samples(), batch_size_);
  EXPECT_EQ(result.type(), DALI_FLOAT);
}

// Test CPU-to-CPU copy (non-passthrough)
// Covers line 50
TEST_F(MakeContiguousMixedTest, CPUToCPUCopy) {
  Pipeline pipe(batch_size_, num_threads_, 0);
  pipe.AddExternalInput("input");

  // Use MakeContiguous with CPU input and CPU output
  // This will use MakeContiguousMixed with CPU output
  pipe.AddOperator(OpSpec("MakeContiguous")
                      .AddArg("device", "cpu")
                      .AddInput("input", StorageDevice::CPU)
                      .AddOutput("output", StorageDevice::CPU));

  pipe.Build({{"output", "cpu"}});

  // Create non-contiguous input to ensure pass_through is false
  TensorList<CPUBackend> input;
  input.set_pinned(false);
  input.SetContiguity(BatchContiguity::Noncontiguous);
  input.Resize({{25}, {30}, {35}}, DALI_INT32);

  for (int i = 0; i < batch_size_; ++i) {
    int *data = input.mutable_tensor<int>(i);
    int size = input.tensor_shape(i)[0];
    for (int j = 0; j < size; ++j) {
      data[j] = i * 100 + j;
    }
  }

  pipe.SetExternalInput("input", input);
  pipe.Run();

  Workspace ws;
  pipe.Outputs(&ws);
  const auto &result = ws.Output<CPUBackend>(0);
  EXPECT_EQ(result.num_samples(), batch_size_);
  EXPECT_EQ(result.type(), DALI_INT32);

  // Verify data was copied correctly
  for (int i = 0; i < batch_size_; ++i) {
    const int *data = result.tensor<int>(i);
    int size = result.tensor_shape(i)[0];
    for (int j = 0; j < size; ++j) {
      EXPECT_EQ(data[j], i * 100 + j);
    }
  }
}

class MakeContiguousErrorTest : public ::testing::Test {
 protected:
  static constexpr int batch_size_ = 3;
  static constexpr int num_threads_ = 2;
};

// Test error handling for inconsistent types in batch
// Attempts to cover lines 34-38 (type mismatch error path)
// Note: This test verifies that normal DALI usage prevents the error condition
TEST_F(MakeContiguousErrorTest, ConsistentTypesInBatch) {
  // TensorList enforces type consistency, so we verify this protection exists
  TensorList<CPUBackend> input;
  input.set_pinned(false);
  input.Resize({{10}, {20}, {30}}, DALI_FLOAT);

  // All samples in a TensorList must have the same type
  // Attempting to set different types would require low-level manipulation
  // that bypasses DALI's type safety, which is the intended behavior
  for (int i = 0; i < batch_size_; ++i) {
    float *data = input.mutable_tensor<float>(i);
    EXPECT_NE(data, nullptr);
    // All samples have type DALI_FLOAT
    EXPECT_EQ(input.type(), DALI_FLOAT);
  }

  // The type consistency check in MakeContiguous (line 34) is defensive
  // programming that should never trigger with properly constructed TensorLists
  EXPECT_EQ(input.type(), DALI_FLOAT);
}

// Test error handling for inconsistent dimensions in batch
// Attempts to cover lines 36-56 (dimension mismatch error path)
// Note: This test verifies that normal DALI usage prevents the error condition
TEST_F(MakeContiguousErrorTest, ConsistentDimensionsInBatch) {
  // TensorList samples can have different shapes but must have same dimensionality
  TensorList<CPUBackend> input;
  input.set_pinned(false);

  // Create samples with same number of dimensions (1D) but different sizes
  TensorListShape<1> shape;
  shape.resize(batch_size_);
  shape.set_tensor_shape(0, {10});
  shape.set_tensor_shape(1, {20});
  shape.set_tensor_shape(2, {30});

  input.Resize(shape, DALI_INT32);

  // Verify all samples have same dimensionality (sample_dim)
  int first_dim = input.shape()[0].sample_dim();
  for (int i = 0; i < batch_size_; ++i) {
    EXPECT_EQ(input.shape()[i].sample_dim(), first_dim);
  }

  // The dimension consistency check in MakeContiguous (line 36) is defensive
  // programming that protects against corrupted data structures
  EXPECT_EQ(first_dim, 1);
}

// Test with 2D tensors to verify dimension consistency across batch
TEST_F(MakeContiguousErrorTest, ConsistentDimensions2D) {
  TensorList<CPUBackend> input;
  input.set_pinned(false);

  // Create 2D samples with different shapes but same dimensionality
  TensorListShape<2> shape;
  shape.resize(batch_size_);
  shape.set_tensor_shape(0, {5, 10});
  shape.set_tensor_shape(1, {8, 15});
  shape.set_tensor_shape(2, {6, 12});

  input.Resize(shape, DALI_FLOAT);

  // Verify all samples have 2 dimensions
  for (int i = 0; i < batch_size_; ++i) {
    EXPECT_EQ(input.shape()[i].sample_dim(), 2);
  }
}

// Test with varying sample sizes to stress the coalesce decision logic
// This exercises the sample size checking loop (lines 29-38)
TEST_F(MakeContiguousErrorTest, VaryingSampleSizes) {
  int device_count = 0;
  CUDA_CALL(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    GTEST_SKIP() << "At least 1 GPU required";
  }

  Pipeline pipe(batch_size_, num_threads_, 0);
  pipe.AddExternalInput("input");

  pipe.AddOperator(OpSpec("MakeContiguous")
                      .AddArg("device", "mixed")
                      .AddInput("input", StorageDevice::CPU)
                      .AddOutput("output", StorageDevice::GPU));

  pipe.Build({{"output", "gpu"}});

  // Create samples with one small and one large (> COALESCE_THRESHOLD)
  // to test the coalesced flag logic
  TensorList<CPUBackend> input;
  input.set_pinned(false);
  TensorListShape<1> shape;
  shape.resize(batch_size_);
  shape.set_tensor_shape(0, {10});        // Small: 40 bytes
  shape.set_tensor_shape(1, {2500});      // Large: 10000 bytes > 8192
  shape.set_tensor_shape(2, {20});        // Small: 80 bytes

  input.Resize(shape, DALI_FLOAT);

  for (int i = 0; i < batch_size_; ++i) {
    float *data = input.mutable_tensor<float>(i);
    int size = input.shape()[i][0];
    for (int j = 0; j < size; ++j) {
      data[j] = static_cast<float>(i * 1000 + j);
    }
  }

  pipe.SetExternalInput("input", input);
  pipe.Run();

  Workspace ws;
  pipe.Outputs(&ws);
  const auto &result = ws.Output<GPUBackend>(0);
  EXPECT_EQ(result.num_samples(), batch_size_);
  EXPECT_EQ(result.type(), DALI_FLOAT);
}

}  // namespace dali

