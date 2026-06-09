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
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "dali/operators/decoder/host/host_decoder.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/common.h"

namespace dali {
namespace testing {

class HostDecoderTest : public ::testing::Test {
 protected:
  OpSpec MakeOpSpec() {
    return OpSpec("decoders__Image")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 1)
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
// Error: input.ndim() != 1  (e.g. 2D tensor triggers DALI_ENFORCE on line 30)
// ============================================================================

TEST_F(HostDecoderTest, WrongNdimThrows) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{24, 3}};  // 2D instead of required 1D
  input->Resize(shape, DALI_UINT8);
  std::memset(input->raw_mutable_tensor(0), 0, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  op->Setup(output_desc, ws);
  EXPECT_THROW(op->Run(ws), DALIException);
}

// ============================================================================
// Error: input type is not uint8  (triggers DALI_ENFORCE on line 31)
// ============================================================================

TEST_F(HostDecoderTest, WrongTypeThrows) {
  auto spec = MakeOpSpec();
  auto op = InstantiateOperator(spec);

  auto input = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<> shape = {{100}};  // 1D - correct ndim
  input->Resize(shape, DALI_INT32);   // wrong type (not uint8)
  std::memset(input->raw_mutable_tensor(0), 0, input->nbytes());

  Workspace ws;
  OldThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, input, tp);

  std::vector<OutputDesc> output_desc;
  op->Setup(output_desc, ws);
  EXPECT_THROW(op->Run(ws), DALIException);
}

}  // namespace testing
}  // namespace dali
