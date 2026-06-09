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
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "dali/operators/debug/dump_image.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/common.h"
#include "dali/core/access_order.h"

namespace dali {
namespace testing {

class DumpImageCPUTest : public ::testing::Test {
 protected:
  OpSpec MakeOpSpec(const std::string &suffix = "test") {
    return OpSpec("DumpImage")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("suffix", suffix)
        .AddInput("input", StorageDevice::CPU)
        .AddOutput("output", StorageDevice::CPU);
  }

  void SetupWorkspace(Workspace &ws,
                      std::shared_ptr<TensorList<CPUBackend>> input,
                      ThreadPool &tp) {
    ws.AddInput(input);
    auto output = std::make_shared<TensorList<CPUBackend>>();
    ws.AddOutput(output);
    ws.SetBatchSizes(input->num_samples());
    ws.SetThreadPool(&tp);
  }
};

// ============================================================================
// Happy path: 3-channel HWC image (3D, c==3)
// ============================================================================

TEST_F(DumpImageCPUTest, RunImplRGB) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{4, 6, 3}};  // H=4, W=6, C=3
  input->Resize(shape, DALI_UINT8);
  std::memset(input->raw_mutable_tensor(0), 128, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Happy path: 1-channel (grayscale) HWC image (3D, c==1)
// ============================================================================

TEST_F(DumpImageCPUTest, RunImplGrayscale) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{4, 6, 1}};  // H=4, W=6, C=1
  input->Resize(shape, DALI_UINT8);
  std::memset(input->raw_mutable_tensor(0), 64, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Error: input.ndim() != 3  (e.g. 2D tensor)
// ============================================================================

TEST_F(DumpImageCPUTest, RunImplWrongNdimThrows) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{24, 3}};  // 2D instead of 3D
  input->Resize(shape, DALI_UINT8);
  std::memset(input->raw_mutable_tensor(0), 0, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

// ============================================================================
// Error: channels not 1 or 3  (e.g. c==2)
// ============================================================================

TEST_F(DumpImageCPUTest, RunImplWrongChannelsThrows) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{4, 6, 2}};  // H=4, W=6, C=2 (unsupported)
  input->Resize(shape, DALI_UINT8);
  std::memset(input->raw_mutable_tensor(0), 0, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

// ============================================================================
// Error: 4 channels (RGBA) - also unsupported
// ============================================================================

TEST_F(DumpImageCPUTest, RunImplFourChannelsThrows) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{4, 6, 4}};  // C=4
  input->Resize(shape, DALI_UINT8);
  std::memset(input->raw_mutable_tensor(0), 0, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

// ============================================================================
// DumpImage GPU Tests
// ============================================================================

class DumpImageGPUTest : public ::testing::Test {
 protected:
  OpSpec MakeGPUOpSpec(const std::string &suffix = "gpu_test") {
    return OpSpec("DumpImage")
        .AddArg("device", "gpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("suffix", suffix)
        .AddInput("input", StorageDevice::GPU)
        .AddOutput("output", StorageDevice::GPU);
  }

  void SetupGPUWorkspace(Workspace &ws,
                         std::shared_ptr<TensorList<GPUBackend>> input) {
    ws.AddInput(input);
    auto output = std::make_shared<TensorList<GPUBackend>>();
    ws.AddOutput(output);
    ws.SetBatchSizes(input->num_samples());
    ws.set_stream(nullptr);
  }

  // Create GPU input by filling CPU data then copying to GPU
  std::shared_ptr<TensorList<GPUBackend>> MakeGPUInput(
      const TensorListShape<> &shape, uint8_t fill_value = 128) {
    auto cpu_input = std::make_shared<TensorList<CPUBackend>>();
    cpu_input->Resize(shape, DALI_UINT8);
    for (int i = 0; i < cpu_input->num_samples(); i++) {
      std::memset(cpu_input->raw_mutable_tensor(i),
                  fill_value,
                  volume(cpu_input->tensor_shape(i)) * sizeof(uint8_t));
    }
    auto gpu_input = std::make_shared<TensorList<GPUBackend>>();
    cudaStream_t stream = nullptr;
    gpu_input->Copy(*cpu_input, AccessOrder(stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
    return gpu_input;
  }
};

// GPU: Happy path with 3-channel RGB image
TEST_F(DumpImageGPUTest, RunImplRGB) {
  auto spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{4, 6, 3}};
  auto input = MakeGPUInput(shape);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(nullptr));
}

// GPU: Happy path with 1-channel grayscale image
TEST_F(DumpImageGPUTest, RunImplGrayscale) {
  auto spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{4, 6, 1}};
  auto input = MakeGPUInput(shape, 64);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
  CUDA_CALL(cudaStreamSynchronize(nullptr));
}

// GPU: Error path - ndim != 3
TEST_F(DumpImageGPUTest, RunImplWrongNdimThrows) {
  auto spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{24, 3}};  // 2D instead of 3D
  auto input = MakeGPUInput(shape);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

// GPU: Error path - channels != 1 and != 3
TEST_F(DumpImageGPUTest, RunImplWrongChannelsThrows) {
  auto spec = MakeGPUOpSpec();
  auto op = InstantiateOperator(spec);

  TensorListShape<> shape = {{4, 6, 2}};  // C=2 unsupported
  auto input = MakeGPUInput(shape);

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

}  // namespace testing
}  // namespace dali
