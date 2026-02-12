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
#include <tuple>
#include <vector>
#include "dali/core/common.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

// Declared by DALI_SCHEMA(SliceBase) in slice_base.cc
int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_SliceBase();

namespace testing {

// ============================================================================
// Cover the SliceBase schema registration function
// ============================================================================

TEST(SliceBaseSchemaTest, SliceBaseSchemaRegistered) {
  EXPECT_EQ(DALI_OPERATOR_SCHEMA_REQUIRED_FOR_SliceBase(), 42);
  EXPECT_NO_THROW(SchemaRegistry::GetSchema("SliceBase"));
}

// ============================================================================
// SliceBase CPU coverage: exercises all (input_type, ndim) same-type paths
// in the VALUE_SWITCH x TYPE_SWITCH at line 107 of slice_base.cc
// SLICE_TYPES: uint8, uint16, uint32, uint64, int8, int16, int32, int64,
//              float16, float, double
// SLICE_DIMS: 1, 2, 3, 4
// ============================================================================

struct SliceTypeInfo {
  DALIDataType dali_type;
  const char *name;
};

static const SliceTypeInfo kSliceTypes[] = {
  {DALI_UINT8,   "uint8"},
  {DALI_UINT16,  "uint16"},
  {DALI_UINT32,  "uint32"},
  {DALI_UINT64,  "uint64"},
  {DALI_INT8,    "int8"},
  {DALI_INT16,   "int16"},
  {DALI_INT32,   "int32"},
  {DALI_INT64,   "int64"},
  {DALI_FLOAT16, "float16"},
  {DALI_FLOAT,   "float32"},
  {DALI_FLOAT64, "float64"},
};

static constexpr int kNumSliceTypes = sizeof(kSliceTypes) / sizeof(kSliceTypes[0]);
static constexpr int kSliceDims[] = {1, 2, 3, 4};
static constexpr int kNumSliceDims = sizeof(kSliceDims) / sizeof(kSliceDims[0]);

// ---------------------------------------------------------------------------
// Same-type Slice: 11 types x 4 dims = 44 tests
// ---------------------------------------------------------------------------

class SliceBaseSameTypeTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(SliceBaseSameTypeTest, RunSlice) {
  int type_idx = std::get<0>(GetParam());
  int dim_idx = std::get<1>(GetParam());
  const auto &type_info = kSliceTypes[type_idx];
  int ndim = kSliceDims[dim_idx];

  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);
  pipe.AddExternalInput("data");

  // Slice the first 2 elements along axis 0
  pipe.AddOperator(
    OpSpec("Slice")
      .AddArg("device", "cpu")
      .AddArg("axes", std::vector<int>{0})
      .AddArg("start", std::vector<int>{0})
      .AddArg("end", std::vector<int>{2})
      .AddInput("data", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU));

  pipe.Build({{"output", "cpu"}});

  // Shape: each dim = 4
  TensorListShape<> tl_shape(batch_size, ndim);
  for (int s = 0; s < batch_size; s++) {
    auto sh = tl_shape.tensor_shape_span(s);
    for (int d = 0; d < ndim; d++) sh[d] = 4;
  }

  TensorList<CPUBackend> input;
  input.Resize(tl_shape, type_info.dali_type);
  std::memset(input.raw_mutable_tensor(0), 1, input.nbytes());

  pipe.SetExternalInput("data", input);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  const auto &output = ws.Output<CPUBackend>(0);
  EXPECT_EQ(output.num_samples(), batch_size);
  // axis 0 is sliced to 2; other dims remain 4
  EXPECT_EQ(output.shape()[0][0], 2);
  for (int d = 1; d < ndim; d++) {
    EXPECT_EQ(output.shape()[0][d], 4);
  }
}

static std::string SameTypeTestName(
    const ::testing::TestParamInfo<std::tuple<int, int>> &info) {
  int type_idx = std::get<0>(info.param);
  int dim_idx = std::get<1>(info.param);
  return std::string(kSliceTypes[type_idx].name) + "_" +
         std::to_string(kSliceDims[dim_idx]) + "d";
}

INSTANTIATE_TEST_SUITE_P(
    SameType,
    SliceBaseSameTypeTest,
    ::testing::Combine(::testing::Range(0, kNumSliceTypes),
                       ::testing::Range(0, kNumSliceDims)),
    SameTypeTestName);

// ---------------------------------------------------------------------------
// Cross-type Slice: each input type x 3 output types (float, float16, uint8)
// with ndim=2, to cover the inner TYPE_SWITCH for output_type.
// We only test cross-type (input_type != output_type) paths.
// 11 input types x 3 output types = 33 combos, minus same-type = ~30 tests
// ---------------------------------------------------------------------------

struct OutputTypeInfo {
  DALIDataType dali_type;
  const char *name;
};

static const OutputTypeInfo kOutputTypes[] = {
  {DALI_FLOAT,   "float"},
  {DALI_FLOAT16, "float16"},
  {DALI_UINT8,   "uint8"},
};

static constexpr int kNumOutputTypes = sizeof(kOutputTypes) / sizeof(kOutputTypes[0]);

class SliceBaseCrossTypeTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(SliceBaseCrossTypeTest, RunSliceCrossType) {
  int in_type_idx = std::get<0>(GetParam());
  int out_type_idx = std::get<1>(GetParam());
  const auto &in_info = kSliceTypes[in_type_idx];
  const auto &out_info = kOutputTypes[out_type_idx];

  // Skip same-type (already covered by SameType tests)
  if (in_info.dali_type == out_info.dali_type) {
    GTEST_SKIP() << "Same type, covered by SameType tests";
  }

  const int batch_size = 1;
  const int ndim = 2;  // use 2D for cross-type tests

  Pipeline pipe(batch_size, 1, 0);
  pipe.AddExternalInput("data");

  pipe.AddOperator(
    OpSpec("Slice")
      .AddArg("device", "cpu")
      .AddArg("dtype", out_info.dali_type)
      .AddArg("axes", std::vector<int>{0})
      .AddArg("start", std::vector<int>{0})
      .AddArg("end", std::vector<int>{2})
      .AddInput("data", StorageDevice::CPU)
      .AddOutput("output", StorageDevice::CPU));

  pipe.Build({{"output", "cpu"}});

  TensorListShape<> tl_shape(batch_size, ndim);
  tl_shape.tensor_shape_span(0)[0] = 4;
  tl_shape.tensor_shape_span(0)[1] = 4;

  TensorList<CPUBackend> input;
  input.Resize(tl_shape, in_info.dali_type);
  std::memset(input.raw_mutable_tensor(0), 1, input.nbytes());

  pipe.SetExternalInput("data", input);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  const auto &output = ws.Output<CPUBackend>(0);
  EXPECT_EQ(output.num_samples(), batch_size);
  EXPECT_EQ(output.type(), out_info.dali_type);
  EXPECT_EQ(output.shape()[0][0], 2);
  EXPECT_EQ(output.shape()[0][1], 4);
}

static std::string CrossTypeTestName(
    const ::testing::TestParamInfo<std::tuple<int, int>> &info) {
  int in_idx = std::get<0>(info.param);
  int out_idx = std::get<1>(info.param);
  return std::string(kSliceTypes[in_idx].name) + "_to_" + kOutputTypes[out_idx].name;
}

INSTANTIATE_TEST_SUITE_P(
    CrossType,
    SliceBaseCrossTypeTest,
    ::testing::Combine(::testing::Range(0, kNumSliceTypes),
                       ::testing::Range(0, kNumOutputTypes)),
    CrossTypeTestName);

}  // namespace testing
}  // namespace dali
