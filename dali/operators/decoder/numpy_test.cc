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
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/numpy.h"
#include "dali/core/float16.h"

namespace dali {

// Declared in numpy.cc
numpy::HeaderData ParseHeader(const std::string_view data);
void RunDecoding(SampleView<CPUBackend> outputSample, ConstSampleView<CPUBackend> inputView,
                 const numpy::HeaderData &header);

namespace testing {

// ============================================================================
// Helper: build a valid .npy v1 binary buffer in memory
// Format: 6-byte magic + 1 major + 1 minor + 2-byte header_len + header_str + data
// ============================================================================

static std::vector<uint8_t> BuildNpyBuffer(const std::string &descr,
                                           bool fortran_order,
                                           const std::vector<int64_t> &shape,
                                           const void *raw_data,
                                           size_t data_bytes) {
  // Build the header dict string
  std::string dict = "{'descr': '";
  dict += descr;
  dict += "', 'fortran_order': ";
  dict += (fortran_order ? "True" : "False");
  dict += ", 'shape': (";
  for (size_t i = 0; i < shape.size(); i++) {
    dict += std::to_string(shape[i]);
    dict += ",";
  }
  dict += "), }";

  // Pad dict with spaces so that 10 + header_len is a multiple of 64
  size_t total_prefix = 10 + dict.size() + 1;  // +1 for trailing newline
  size_t pad = (64 - (total_prefix % 64)) % 64;
  dict.append(pad, ' ');
  dict += '\n';

  uint16_t header_len = static_cast<uint16_t>(dict.size());

  std::vector<uint8_t> buf;
  buf.reserve(10 + header_len + data_bytes);

  // Magic
  buf.push_back(0x93);
  buf.push_back('N'); buf.push_back('U'); buf.push_back('M');
  buf.push_back('P'); buf.push_back('Y');
  // Version 1.0
  buf.push_back(1);
  buf.push_back(0);
  // Header length (LE)
  buf.push_back(static_cast<uint8_t>(header_len & 0xFF));
  buf.push_back(static_cast<uint8_t>((header_len >> 8) & 0xFF));
  // Header dict
  buf.insert(buf.end(), dict.begin(), dict.end());
  // Data
  const auto *p = static_cast<const uint8_t *>(raw_data);
  buf.insert(buf.end(), p, p + data_bytes);

  return buf;
}

template <typename T>
static std::vector<uint8_t> BuildNpyBufferTyped(const std::string &descr,
                                                bool fortran_order,
                                                const std::vector<int64_t> &shape,
                                                const std::vector<T> &data) {
  return BuildNpyBuffer(descr, fortran_order, shape,
                        data.data(), data.size() * sizeof(T));
}

// ============================================================================
// ParseHeader tests — covering error branches
// ============================================================================

TEST(NumpyDecoderTest, ParseHeaderBadMagic) {
  // Data that doesn't start with \x93NUMPY
  std::vector<uint8_t> bad = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A};
  EXPECT_THROW(
    ParseHeader(std::string_view(reinterpret_cast<const char *>(bad.data()), bad.size())),
    DALIException);
}

TEST(NumpyDecoderTest, ParseHeaderTooShort) {
  // Less than 10 bytes
  std::vector<uint8_t> tiny = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  EXPECT_THROW(
    ParseHeader(std::string_view(reinterpret_cast<const char *>(tiny.data()), tiny.size())),
    DALIException);
}

TEST(NumpyDecoderTest, ParseHeaderBadVersion) {
  // Version 2 not supported
  std::vector<uint8_t> v2 = {0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0, 0, 0};
  EXPECT_THROW(
    ParseHeader(std::string_view(reinterpret_cast<const char *>(v2.data()), v2.size())),
    DALIException);
}

TEST(NumpyDecoderTest, ParseHeaderLenExceedsInput) {
  // header_len says 9999 but data is only 12 bytes
  std::vector<uint8_t> buf = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0x0F, 0x27};
  buf.resize(12, 0);
  EXPECT_THROW(
    ParseHeader(std::string_view(reinterpret_cast<const char *>(buf.data()), buf.size())),
    DALIException);
}

TEST(NumpyDecoderTest, ParseHeaderSizeMismatch) {
  // Valid header but data size doesn't match expected
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto buf = BuildNpyBufferTyped<float>("<f4", false, {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  // Append extra bytes to make size mismatch
  buf.push_back(0xFF);
  buf.push_back(0xFF);
  EXPECT_THROW(
    ParseHeader(std::string_view(reinterpret_cast<const char *>(buf.data()), buf.size())),
    DALIException);
}

TEST(NumpyDecoderTest, ParseHeaderValid) {
  auto buf = BuildNpyBufferTyped<float>("<f4", false, {2, 3},
                                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto header = ParseHeader(
    std::string_view(reinterpret_cast<const char *>(buf.data()), buf.size()));
  EXPECT_EQ(header.type(), DALI_FLOAT);
  EXPECT_EQ(header.fortran_order, false);
  EXPECT_EQ(header.shape.size(), 2);
  EXPECT_EQ(header.shape[0], 2);
  EXPECT_EQ(header.shape[1], 3);
}

TEST(NumpyDecoderTest, ParseHeaderFortranOrder) {
  auto buf = BuildNpyBufferTyped<float>("<f4", true, {2, 3},
                                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto header = ParseHeader(
    std::string_view(reinterpret_cast<const char *>(buf.data()), buf.size()));
  EXPECT_EQ(header.fortran_order, true);
}

// ============================================================================
// RunDecoding tests — covering type conversion paths
// Each (OType, IType) pair covers a unique branch in the double TYPE_SWITCH
// ============================================================================

struct TypeDesc {
  DALIDataType dali_type;
  size_t size;
  const char *name;
};

static const TypeDesc kAllTypes[] = {
  {DALI_BOOL,    sizeof(bool),     "bool"},
  {DALI_UINT8,   sizeof(uint8_t),  "uint8"},
  {DALI_UINT16,  sizeof(uint16_t), "uint16"},
  {DALI_UINT32,  sizeof(uint32_t), "uint32"},
  {DALI_UINT64,  sizeof(uint64_t), "uint64"},
  {DALI_INT8,    sizeof(int8_t),   "int8"},
  {DALI_INT16,   sizeof(int16_t),  "int16"},
  {DALI_INT32,   sizeof(int32_t),  "int32"},
  {DALI_INT64,   sizeof(int64_t),  "int64"},
  {DALI_FLOAT,   sizeof(float),    "float32"},
  {DALI_FLOAT16, sizeof(float16),  "float16"},
  {DALI_FLOAT64, sizeof(double),   "float64"},
};

static constexpr int kNumTypes = sizeof(kAllTypes) / sizeof(kAllTypes[0]);

// Test all type conversion pairs for RunDecoding
// This covers the double-nested TYPE_SWITCH in RunDecoding (lines 132-146)
class RunDecodingTypeConversionTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(RunDecodingTypeConversionTest, ConvertTypes) {
  int out_idx = std::get<0>(GetParam());
  int in_idx = std::get<1>(GetParam());
  const auto &out_type = kAllTypes[out_idx];
  const auto &in_type = kAllTypes[in_idx];

  // Skip same-type pairs — those take the memcpy path, not TYPE_SWITCH
  if (out_type.dali_type == in_type.dali_type) {
    GTEST_SKIP() << "Same type " << out_type.name << " takes memcpy path";
  }

  const int64_t num_elems = 4;

  // Allocate input buffer filled with a small value (1)
  std::vector<uint8_t> input_buf(num_elems * in_type.size, 0);
  // Set each element to value 1 in its type representation
  for (int64_t i = 0; i < num_elems; i++) {
    uint8_t *elem = input_buf.data() + i * in_type.size;
    if (in_type.dali_type == DALI_BOOL) {
      *reinterpret_cast<bool *>(elem) = true;
    } else if (in_type.dali_type == DALI_UINT8) {
      *reinterpret_cast<uint8_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_UINT16) {
      *reinterpret_cast<uint16_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_UINT32) {
      *reinterpret_cast<uint32_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_UINT64) {
      *reinterpret_cast<uint64_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_INT8) {
      *reinterpret_cast<int8_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_INT16) {
      *reinterpret_cast<int16_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_INT32) {
      *reinterpret_cast<int32_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_INT64) {
      *reinterpret_cast<int64_t *>(elem) = 1;
    } else if (in_type.dali_type == DALI_FLOAT) {
      *reinterpret_cast<float *>(elem) = 1.0f;
    } else if (in_type.dali_type == DALI_FLOAT16) {
      *reinterpret_cast<float16 *>(elem) = static_cast<float16>(1.0f);
    } else if (in_type.dali_type == DALI_FLOAT64) {
      *reinterpret_cast<double *>(elem) = 1.0;
    }
  }

  // Allocate output buffer
  std::vector<uint8_t> output_buf(num_elems * out_type.size, 0);

  // Build header
  numpy::HeaderData header;
  header.shape = {num_elems};
  header.type_info = &TypeTable::GetTypeInfo(in_type.dali_type);
  header.fortran_order = false;
  header.data_offset = 0;

  TensorShape<> shape = {num_elems};

  SampleView<CPUBackend> output_view(output_buf.data(), shape, out_type.dali_type);
  ConstSampleView<CPUBackend> input_view(
      static_cast<const void *>(input_buf.data()), shape, in_type.dali_type);

  ASSERT_NO_THROW(RunDecoding(output_view, input_view, header))
      << "Failed converting " << in_type.name << " -> " << out_type.name;
}

static std::string TypeConversionTestName(
    const ::testing::TestParamInfo<std::tuple<int, int>> &info) {
  int o = std::get<0>(info.param);
  int i = std::get<1>(info.param);
  return std::string(kAllTypes[o].name) + "_from_" + kAllTypes[i].name;
}

INSTANTIATE_TEST_SUITE_P(
    AllTypeConversions,
    RunDecodingTypeConversionTest,
    ::testing::Combine(::testing::Range(0, kNumTypes), ::testing::Range(0, kNumTypes)),
    TypeConversionTestName);

// ============================================================================
// RunDecoding same-type path (memcpy) — covers the else branch at line 141
// ============================================================================

TEST(NumpyDecoderTest, RunDecodingSameType) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> output_data(4, 0.0f);

  numpy::HeaderData header;
  header.shape = {4};
  header.type_info = &TypeTable::GetTypeInfo(DALI_FLOAT);
  header.fortran_order = false;
  header.data_offset = 0;

  TensorShape<> shape = {4};
  SampleView<CPUBackend> output_view(output_data.data(), shape, DALI_FLOAT);
  ConstSampleView<CPUBackend> input_view(
      static_cast<const void *>(input_data.data()), shape, DALI_FLOAT);

  RunDecoding(output_view, input_view, header);

  EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// RunDecoding with fortran_order — covers lines 123-130
// ============================================================================

TEST(NumpyDecoderTest, RunDecodingFortranOrder) {
  // 2x3 matrix in Fortran (column-major) order
  // Input data viewed as (2, 3) row-major: [[1,2,3],[4,5,6]]
  // Transpose perm [1,0] -> (3, 2): [[1,4],[2,5],[3,6]] = flat [1,4,2,5,3,6]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> output_data(6, 0.0f);

  numpy::HeaderData header;
  header.shape = {2, 3};
  header.type_info = &TypeTable::GetTypeInfo(DALI_FLOAT);
  header.fortran_order = true;
  header.data_offset = 0;

  TensorShape<> output_shape = {3, 2};
  TensorShape<> input_shape = {2, 3};

  SampleView<CPUBackend> output_view(output_data.data(), output_shape, DALI_FLOAT);
  ConstSampleView<CPUBackend> input_view(
      static_cast<const void *>(input_data.data()), input_shape, DALI_FLOAT);

  RunDecoding(output_view, input_view, header);

  // After transpose perm [1,0] from (2,3) to (3,2):
  // out[j][i] = in[i][j]: [1,4,2,5,3,6]
  std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  EXPECT_EQ(output_data, expected);
}

// ============================================================================
// RunDecoding with fortran_order AND type conversion
// covers lines 123-130 + TYPE_SWITCH conversion
// ============================================================================

TEST(NumpyDecoderTest, RunDecodingFortranOrderWithConversion) {
  // 2x3 int32 matrix in Fortran order, converted to float32
  // Input: [[1,2,3],[4,5,6]] as (2,3) -> transpose to (3,2): [1,4,2,5,3,6]
  std::vector<int32_t> input_data = {1, 2, 3, 4, 5, 6};
  std::vector<float> output_data(6, 0.0f);

  numpy::HeaderData header;
  header.shape = {2, 3};
  header.type_info = &TypeTable::GetTypeInfo(DALI_INT32);
  header.fortran_order = true;
  header.data_offset = 0;

  TensorShape<> output_shape = {3, 2};
  TensorShape<> input_shape = {2, 3};

  SampleView<CPUBackend> output_view(output_data.data(), output_shape, DALI_FLOAT);
  ConstSampleView<CPUBackend> input_view(
      static_cast<const void *>(input_data.data()), input_shape, DALI_INT32);

  RunDecoding(output_view, input_view, header);

  // After fortran transpose [1,0] + conversion to float: [1,4,2,5,3,6]
  std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  EXPECT_EQ(output_data, expected);
}

}  // namespace testing
}  // namespace dali
