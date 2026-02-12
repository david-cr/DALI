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
// Erase GPU operator coverage test: exercises all (type, ndim, channel_dim)
// combinations in the TYPE_SWITCH x VALUE_SWITCH at line 164 of erase.cu
// ERASE_SUPPORTED_TYPES (GPU): uint8, int8, uint16, int16, uint32, int32,
//                              uint64, int64, float (no float16 on GPU)
// ERASE_SUPPORTED_NDIMS: 1, 2, 3, 4, 5
// channel_dim: -1 (no channel), 0 (first), Dims-1 (last)
// ============================================================================

struct EraseGpuTypeInfo {
  DALIDataType dali_type;
  const char *name;
};

static const EraseGpuTypeInfo kEraseGpuTypes[] = {
  {DALI_UINT8,   "uint8"},
  {DALI_INT8,    "int8"},
  {DALI_UINT16,  "uint16"},
  {DALI_INT16,   "int16"},
  {DALI_UINT32,  "uint32"},
  {DALI_INT32,   "int32"},
  {DALI_UINT64,  "uint64"},
  {DALI_INT64,   "int64"},
  {DALI_FLOAT,   "float32"},
};

static constexpr int kNumGpuTypes = sizeof(kEraseGpuTypes) / sizeof(kEraseGpuTypes[0]);
static constexpr int kSupportedNdims[] = {1, 2, 3, 4, 5};
static constexpr int kNumNdims = sizeof(kSupportedNdims) / sizeof(kSupportedNdims[0]);

// channel_dim_mode: 0 = no channel (-1), 1 = channel first (0), 2 = channel last (Dims-1)
static constexpr int kNumChannelModes = 3;

// Build a layout string for the given ndim and channel mode
static std::string MakeLayout(int ndim, int channel_mode) {
  // For ndim=1, channel modes 1 and 2 both just put C at the single dim
  if (ndim == 1) {
    if (channel_mode == 0) return "";  // no channel
    return "C";  // channel first or last (same for 1D)
  }
  // Use letters for non-channel dims: D, H, W, X, Y...
  const char dim_letters[] = {'H', 'W', 'D', 'X', 'Y'};
  std::string layout;
  int non_c_dims = (channel_mode == 0) ? ndim : ndim;
  if (channel_mode == 0) {
    // No channel dim
    for (int i = 0; i < ndim; i++)
      layout += dim_letters[i % 5];
  } else if (channel_mode == 1) {
    // Channel first
    layout += 'C';
    for (int i = 0; i < ndim - 1; i++)
      layout += dim_letters[i % 5];
  } else {
    // Channel last
    for (int i = 0; i < ndim - 1; i++)
      layout += dim_letters[i % 5];
    layout += 'C';
  }
  return layout;
}

// Parametrized test: (type_idx, ndim_idx, channel_mode)
class EraseGpuTypeDimTest
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(EraseGpuTypeDimTest, RunEraseGpu) {
  int type_idx = std::get<0>(GetParam());
  int ndim_idx = std::get<1>(GetParam());
  int channel_mode = std::get<2>(GetParam());
  const auto &type_info = kEraseGpuTypes[type_idx];
  int ndim = kSupportedNdims[ndim_idx];

  // For ndim == 1 with channel_mode == 1 (channel first) and 2 (channel last),
  // they are equivalent, skip one to avoid duplicates
  if (ndim == 1 && channel_mode == 2) {
    GTEST_SKIP() << "1D channel-last is same as channel-first, skipping";
  }

  const int batch_size = 1;
  const int num_threads = 1;
  const int device_id = 0;

  Pipeline pipe(batch_size, num_threads, device_id);
  pipe.AddExternalInput("data");

  // Build layout
  std::string layout = MakeLayout(ndim, channel_mode);

  // Axes for erase: use the first non-channel axis
  // For no-channel: axes = {0}
  // For channel-first: axes = {1} (first spatial dim)
  // For channel-last: axes = {0} (first spatial dim)
  std::vector<int> axes;
  if (channel_mode == 0 || channel_mode == 2) {
    axes = {0};
  } else {
    // channel first: first spatial dim is index 1
    axes = {1};
  }
  // Special case: ndim==1 with channel, the only dim is C
  if (ndim == 1 && channel_mode == 1) {
    axes = {0};
  }

  std::vector<float> anchor = {0.0f};
  std::vector<float> shape_arg = {2.0f};

  auto erase_spec = OpSpec("Erase")
      .AddArg("device", "gpu")
      .AddArg("axes", axes)
      .AddArg("anchor", anchor)
      .AddArg("shape", shape_arg)
      .AddArg("fill_value", std::vector<float>{0.0f})
      .AddInput("data", StorageDevice::GPU)
      .AddOutput("output", StorageDevice::GPU);

  pipe.AddOperator(erase_spec);
  pipe.Build({{"output", "gpu"}});

  // Create input data
  TensorListShape<> tl_shape(batch_size, ndim);
  for (int s = 0; s < batch_size; s++) {
    auto sh = tl_shape.tensor_shape_span(s);
    for (int d = 0; d < ndim; d++) {
      sh[d] = 4;
    }
  }

  TensorList<CPUBackend> input_batch;
  input_batch.Resize(tl_shape, type_info.dali_type);
  if (!layout.empty()) {
    input_batch.SetLayout(layout);
  }
  std::memset(input_batch.raw_mutable_tensor(0), 1, input_batch.nbytes());

  pipe.SetExternalInput("data", input_batch);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  const auto &output = ws.Output<GPUBackend>(0);
  EXPECT_EQ(output.num_samples(), batch_size);
  EXPECT_EQ(output.shape()[0].size(), ndim);
}

static std::string EraseGpuTestName(
    const ::testing::TestParamInfo<std::tuple<int, int, int>> &info) {
  int type_idx = std::get<0>(info.param);
  int ndim_idx = std::get<1>(info.param);
  int ch_mode = std::get<2>(info.param);
  const char *ch_str[] = {"noCh", "chFirst", "chLast"};
  return std::string(kEraseGpuTypes[type_idx].name) + "_" +
         std::to_string(kSupportedNdims[ndim_idx]) + "d_" + ch_str[ch_mode];
}

INSTANTIATE_TEST_SUITE_P(
    AllGpuTypeDimsCh,
    EraseGpuTypeDimTest,
    ::testing::Combine(::testing::Range(0, kNumGpuTypes),
                       ::testing::Range(0, kNumNdims),
                       ::testing::Range(0, kNumChannelModes)),
    EraseGpuTestName);

// ============================================================================
// Test with default fill value (no fill_value arg) — covers line 83-84
// ============================================================================

class EraseGpuDefaultFillTest : public ::testing::Test {};

TEST_F(EraseGpuDefaultFillTest, DefaultFillValue) {
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);
  pipe.AddExternalInput("data");

  // Don't provide fill_value - uses default path (line 83-84)
  pipe.AddOperator(
    OpSpec("Erase")
      .AddArg("device", "gpu")
      .AddArg("axes", std::vector<int>{0, 1})
      .AddArg("anchor", std::vector<float>{0.0f, 0.0f})
      .AddArg("shape", std::vector<float>{2.0f, 2.0f})
      .AddInput("data", StorageDevice::GPU)
      .AddOutput("output", StorageDevice::GPU));

  pipe.Build({{"output", "gpu"}});

  TensorListShape<> tl_shape(batch_size, 3);
  tl_shape.tensor_shape_span(0)[0] = 4;
  tl_shape.tensor_shape_span(0)[1] = 4;
  tl_shape.tensor_shape_span(0)[2] = 3;

  TensorList<CPUBackend> input_batch;
  input_batch.Resize(tl_shape, DALI_UINT8);
  input_batch.SetLayout("HWC");
  std::memset(input_batch.raw_mutable_tensor(0), 128, input_batch.nbytes());

  pipe.SetExternalInput("data", input_batch);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  EXPECT_EQ(ws.Output<GPUBackend>(0).num_samples(), batch_size);
}

}  // namespace testing
}  // namespace dali
