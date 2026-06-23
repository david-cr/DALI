// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

namespace {

std::shared_ptr<TensorList<CPUBackend>> MakeCpuTL(int batch, DALIDataType type = DALI_FLOAT) {
  auto tl = std::make_shared<TensorList<CPUBackend>>();
  tl->set_order(AccessOrder::host());
  tl->Resize(uniform_list_shape(batch, TensorShape<2>{2, 2}), type);
  return tl;
}

std::shared_ptr<TensorList<GPUBackend>> MakeGpuTL(int batch, DALIDataType type = DALI_FLOAT) {
  auto tl = std::make_shared<TensorList<GPUBackend>>();
  tl->Resize(uniform_list_shape(batch, TensorShape<2>{2, 2}), type);
  return tl;
}

}  // namespace

// MakeSampleView should populate the sample workspace with views into the batch
// for both CPU and GPU inputs/outputs and for argument inputs. A single call
// with one of each kind exercises both backend branches plus the argument-input
// loop in MakeSampleView.
TEST(SampleWorkspaceTest, MakeSampleViewAllKinds) {
  constexpr int kBatch = 4;
  constexpr int kDataIdx = 2;
  constexpr int kThreadIdx = 1;

  Workspace batch;
  batch.set_output_order(AccessOrder::host());

  batch.AddInput(MakeCpuTL(kBatch));
  batch.AddInput(MakeGpuTL(kBatch));

  batch.AddOutput(MakeCpuTL(kBatch));
  batch.AddOutput(MakeGpuTL(kBatch));

  auto arg = std::make_shared<TensorList<CPUBackend>>();
  arg->set_order(AccessOrder::host());
  arg->Resize(uniform_list_shape(kBatch, TensorShape<1>{1}), DALI_INT32);
  batch.AddArgumentInput("my_arg", arg);

  SampleWorkspace sample;
  MakeSampleView(sample, batch, kDataIdx, kThreadIdx);

  EXPECT_EQ(sample.data_idx(), kDataIdx);
  EXPECT_EQ(sample.thread_idx(), kThreadIdx);
  ASSERT_EQ(sample.NumInput(), 2);
  ASSERT_EQ(sample.NumOutput(), 2);
  EXPECT_TRUE(sample.InputIsType<CPUBackend>(0));
  EXPECT_TRUE(sample.InputIsType<GPUBackend>(1));
  EXPECT_TRUE(sample.OutputIsType<CPUBackend>(0));
  EXPECT_TRUE(sample.OutputIsType<GPUBackend>(1));
  EXPECT_EQ(sample.NumArgumentInput(), 1);
}

// FixBatchPropertiesConsistency walks CPU outputs and refreshes the TensorList
// properties for both the contiguous and non-contiguous cases.
TEST(SampleWorkspaceTest, FixBatchPropertiesConsistency) {
  constexpr int kBatch = 3;
  Workspace batch;
  batch.set_output_order(AccessOrder::host());
  batch.AddOutput(MakeCpuTL(kBatch));

  EXPECT_NO_THROW(FixBatchPropertiesConsistency(batch, /*contiguous=*/false));
  EXPECT_NO_THROW(FixBatchPropertiesConsistency(batch, /*contiguous=*/true));
}

}  // namespace dali
