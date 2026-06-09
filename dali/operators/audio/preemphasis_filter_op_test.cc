// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "dali/operators/audio/preemphasis_filter_op.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/common.h"
#include "dali/core/access_order.h"

namespace dali {
namespace testing {

// All types supported by PreemphasisFilter
static const std::vector<DALIDataType> kPreemphTypes = {
    DALI_UINT8, DALI_INT8, DALI_UINT16, DALI_INT16,
    DALI_UINT32, DALI_INT32, DALI_UINT64, DALI_INT64,
    DALI_FLOAT, DALI_FLOAT64
};

class PreemphasisFilterCPUTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  OpSpec MakeOpSpec(DALIDataType output_dtype,
                    const std::string &border = "clamp",
                    float coeff = 0.97f) {
    return OpSpec("PreemphasisFilter")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("preemph_coeff", coeff)
        .AddArg("dtype", output_dtype)
        .AddArg("border", border)
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

  // Run the operator with given input/output types, border mode, and coefficient
  void RunTest(DALIDataType input_dtype, DALIDataType output_dtype,
               const std::string &border = "clamp", float coeff = 0.97f) {
    OpSpec spec = MakeOpSpec(output_dtype, border, coeff);
    auto op = InstantiateOperator(spec);

    auto input = std::make_shared<TensorList<CPUBackend>>();
    TensorListShape<> shape = {{100}};
    input->Resize(shape, input_dtype);
    // Fill with non-zero pattern so filter logic is exercised
    auto *raw = static_cast<uint8_t *>(input->raw_mutable_tensor(0));
    for (size_t i = 0; i < input->nbytes(); i++) {
      raw[i] = static_cast<uint8_t>((i % 200) + 1);
    }

    Workspace ws;
    OldThreadPool tp(1, 0, false, "TestPool");
    SetupWorkspace(ws, input, tp);

    std::vector<OutputDesc> output_desc;
    ASSERT_TRUE(op->Setup(output_desc, ws));
    ASSERT_EQ(output_desc.size(), 1);

    ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

    EXPECT_NO_THROW(op->Run(ws));
  }
};

// ============================================================================
// Tests covering all input type x output type combinations in RunImpl
// Each test covers one input type with ALL 10 output types
// ============================================================================

TEST_F(PreemphasisFilterCPUTest, RunImplInputUInt8AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_UINT8, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputInt8AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_INT8, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputUInt16AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_UINT16, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputInt16AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_INT16, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputUInt32AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_UINT32, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputInt32AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_INT32, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputUInt64AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_UINT64, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputInt64AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_INT64, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputFloatAllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_FLOAT, out_type);
  }
}

TEST_F(PreemphasisFilterCPUTest, RunImplInputDoubleAllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunTest(DALI_FLOAT64, out_type);
  }
}

// ============================================================================
// Border mode tests
// ============================================================================

TEST_F(PreemphasisFilterCPUTest, BorderZero) {
  RunTest(DALI_FLOAT, DALI_FLOAT, "zero");
}

TEST_F(PreemphasisFilterCPUTest, BorderReflect) {
  RunTest(DALI_FLOAT, DALI_FLOAT, "reflect");
}

TEST_F(PreemphasisFilterCPUTest, BorderClamp) {
  RunTest(DALI_FLOAT, DALI_FLOAT, "clamp");
}

// ============================================================================
// Coefficient edge cases
// ============================================================================

// coeff == 0 triggers the identity-copy path
TEST_F(PreemphasisFilterCPUTest, CoeffZeroIdentityCopy) {
  RunTest(DALI_FLOAT, DALI_FLOAT, "clamp", 0.0f);
}

TEST_F(PreemphasisFilterCPUTest, CoeffZeroWithIntTypes) {
  RunTest(DALI_INT16, DALI_INT32, "clamp", 0.0f);
}

// ============================================================================
// Unsupported input type test
// ============================================================================

TEST_F(PreemphasisFilterCPUTest, UnsupportedInputTypeThrows) {
  OpSpec spec = MakeOpSpec(DALI_FLOAT);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input->Resize(shape, DALI_BOOL);  // Unsupported input type
  std::memset(input->raw_mutable_tensor(0), 0, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_THROW(op->Run(ws), DALIException);
}

// ============================================================================
// Batch processing test
// ============================================================================

TEST_F(PreemphasisFilterCPUTest, BatchProcessing) {
  OpSpec spec = MakeOpSpec(DALI_FLOAT);
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}, {200}, {50}};  // 3 samples
  input->Resize(shape, DALI_FLOAT);

  for (int i = 0; i < 3; i++) {
    auto *ptr = input->mutable_tensor<float>(i);
    for (int64_t j = 0; j < shape[i][0]; j++) {
      ptr[j] = static_cast<float>(j + 1);
    }
  }

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ws.Output<CPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// PreemphasisFilter GPU Tests
// ============================================================================

class PreemphasisFilterGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  OpSpec MakeGPUOpSpec(DALIDataType output_dtype,
                       const std::string &border = "clamp",
                       float coeff = 0.97f) {
    return OpSpec("PreemphasisFilter")
        .AddArg("device", "gpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("preemph_coeff", coeff)
        .AddArg("dtype", output_dtype)
        .AddArg("border", border)
        .AddInput("input", StorageDevice::GPU)
        .AddOutput("output", StorageDevice::GPU);
  }

  void SetupGPUWorkspace(Workspace &ws,
                         std::shared_ptr<TensorList<GPUBackend>> input) {
    ws.AddInput(input);
    auto output = std::make_shared<TensorList<GPUBackend>>();
    ws.AddOutput(output);
    ws.SetBatchSizes(input->num_samples());
    ws.set_stream(0);
  }

  void RunGPUTest(DALIDataType input_dtype, DALIDataType output_dtype,
                  const std::string &border = "clamp", float coeff = 0.97f) {
    OpSpec spec = MakeGPUOpSpec(output_dtype, border, coeff);
    auto op = InstantiateOperator(spec);

    // Create CPU input, fill with data, copy to GPU
    auto input_cpu = std::make_shared<TensorList<CPUBackend>>();
    TensorListShape<> shape = {{100}};
    input_cpu->Resize(shape, input_dtype);
    auto *raw = static_cast<uint8_t *>(input_cpu->raw_mutable_tensor(0));
    for (size_t i = 0; i < input_cpu->nbytes(); i++) {
      raw[i] = static_cast<uint8_t>((i % 200) + 1);
    }

    auto input = std::make_shared<TensorList<GPUBackend>>();
    cudaStream_t stream = nullptr;
    input->Copy(*input_cpu, AccessOrder(stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    Workspace ws;
    SetupGPUWorkspace(ws, input);

    std::vector<OutputDesc> output_desc;
    ASSERT_TRUE(op->Setup(output_desc, ws));
    ASSERT_EQ(output_desc.size(), 1);

    ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

    EXPECT_NO_THROW(op->Run(ws));
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
};

// GPU: All input type x output type combinations
TEST_F(PreemphasisFilterGPUTest, RunImplInputUInt8AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_UINT8, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputInt8AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_INT8, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputUInt16AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_UINT16, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputInt16AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_INT16, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputUInt32AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_UINT32, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputInt32AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_INT32, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputUInt64AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_UINT64, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputInt64AllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_INT64, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputFloatAllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_FLOAT, out_type);
  }
}

TEST_F(PreemphasisFilterGPUTest, RunImplInputDoubleAllOutputTypes) {
  for (auto out_type : kPreemphTypes) {
    RunGPUTest(DALI_FLOAT64, out_type);
  }
}

// GPU: Unsupported input type
TEST_F(PreemphasisFilterGPUTest, UnsupportedInputTypeThrows) {
  OpSpec spec = MakeGPUOpSpec(DALI_FLOAT);
  auto op = InstantiateOperator(spec);

  auto input_cpu = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};
  input_cpu->Resize(shape, DALI_BOOL);
  std::memset(input_cpu->raw_mutable_tensor(0), 0, input_cpu->nbytes());

  auto input = std::make_shared<TensorList<GPUBackend>>();
  cudaStream_t stream = nullptr;
  input->Copy(*input_cpu, AccessOrder(stream));
  CUDA_CALL(cudaStreamSynchronize(stream));

  Workspace ws;
  SetupGPUWorkspace(ws, input);

  std::vector<OutputDesc> output_desc;
  ASSERT_TRUE(op->Setup(output_desc, ws));
  ws.Output<GPUBackend>(0).Resize(output_desc[0].shape, output_desc[0].type);

  EXPECT_THROW(op->Run(ws), DALIException);
}

}  // namespace testing
}  // namespace dali
