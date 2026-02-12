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
#include <cstdint>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {
namespace testing {

// ============================================================================
// Erase operator coverage test: exercises all (type, ndim) combinations in
// the TYPE_SWITCH x VALUE_SWITCH at line 234 of erase.cc
// ERASE_SUPPORTED_TYPES: uint8, int8, uint16, int16, uint32, int32, uint64, int64, float, float16
// ERASE_SUPPORTED_NDIMS: 1, 2, 3, 4, 5
// ============================================================================

struct EraseTypeInfo {
  DALIDataType dali_type;
  const char *name;
};

static const EraseTypeInfo kEraseTypes[] = {
  {DALI_UINT8,   "uint8"},
  {DALI_INT8,    "int8"},
  {DALI_UINT16,  "uint16"},
  {DALI_INT16,   "int16"},
  {DALI_UINT32,  "uint32"},
  {DALI_INT32,   "int32"},
  {DALI_UINT64,  "uint64"},
  {DALI_INT64,   "int64"},
  {DALI_FLOAT,   "float32"},
  {DALI_FLOAT16, "float16"},
};

static constexpr int kNumEraseTypes = sizeof(kEraseTypes) / sizeof(kEraseTypes[0]);

// Supported ndims: 1, 2, 3, 4, 5
static constexpr int kSupportedNdims[] = {1, 2, 3, 4, 5};
static constexpr int kNumNdims = sizeof(kSupportedNdims) / sizeof(kSupportedNdims[0]);

class EraseTypeDimTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(EraseTypeDimTest, RunErase) {
  int type_idx = std::get<0>(GetParam());
  int ndim_idx = std::get<1>(GetParam());
  const auto &type_info = kEraseTypes[type_idx];
  int ndim = kSupportedNdims[ndim_idx];

  const int batch_size = 1;
  const int num_threads = 1;
  const int device_id = 0;

  Pipeline pipe(batch_size, num_threads, device_id);
  pipe.AddExternalInput("data");

  // Build a shape with the right number of dims, each dim = 4
  // For ndim >= 2, use axes={0, 1} for the erase region
  // For ndim == 1, use axes={0}
  std::vector<int> axes;
  std::vector<float> anchor;
  std::vector<float> shape_arg;
  if (ndim == 1) {
    axes = {0};
    anchor = {0.0f};
    shape_arg = {2.0f};
  } else {
    axes = {0, 1};
    anchor = {0.0f, 0.0f};
    shape_arg = {2.0f, 2.0f};
  }

  pipe.AddOperator(
    OpSpec("Erase")
      .AddArg("device", "cpu")
      .AddArg("axes", axes)
      .AddArg("anchor", anchor)
      .AddArg("shape", shape_arg)
      .AddArg("fill_value", std::vector<float>{0.0f})
      .AddInput("data", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU));

  pipe.Build({{"output", "cpu"}});

  // Create input data with the right type and ndim
  TensorListShape<> tl_shape(batch_size, ndim);
  for (int s = 0; s < batch_size; s++) {
    auto sh = tl_shape.tensor_shape_span(s);
    for (int d = 0; d < ndim; d++) {
      sh[d] = 4;
    }
  }

  TensorList<CPUBackend> input_batch;
  input_batch.Resize(tl_shape, type_info.dali_type);
  // Zero-fill the data
  std::memset(input_batch.raw_mutable_tensor(0), 1, input_batch.nbytes());

  pipe.SetExternalInput("data", input_batch);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  // Just verify we got output with correct shape
  const auto &output = ws.Output<CPUBackend>(0);
  EXPECT_EQ(output.num_samples(), batch_size);
  EXPECT_EQ(output.shape()[0].size(), ndim);
}

static std::string EraseTestName(
    const ::testing::TestParamInfo<std::tuple<int, int>> &info) {
  int type_idx = std::get<0>(info.param);
  int ndim_idx = std::get<1>(info.param);
  return std::string(kEraseTypes[type_idx].name) + "_" +
         std::to_string(kSupportedNdims[ndim_idx]) + "d";
}

INSTANTIATE_TEST_SUITE_P(
    AllTypeDims,
    EraseTypeDimTest,
    ::testing::Combine(::testing::Range(0, kNumEraseTypes),
                       ::testing::Range(0, kNumNdims)),
    EraseTestName);

}  // namespace testing
}  // namespace dali
