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
#include <memory>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/decoder/inflate/inflate_params.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace testing {

// ============================================================================
// parse_inflate_alg — valid "LZ4" (case-insensitive)
// ============================================================================

TEST(InflateParamsTest, ParseInflateAlgLZ4) {
  EXPECT_EQ(inflate::parse_inflate_alg("LZ4"), inflate::InflateAlg::LZ4);
}

TEST(InflateParamsTest, ParseInflateAlgLowerCase) {
  EXPECT_EQ(inflate::parse_inflate_alg("lz4"), inflate::InflateAlg::LZ4);
}

TEST(InflateParamsTest, ParseInflateAlgMixedCase) {
  EXPECT_EQ(inflate::parse_inflate_alg("Lz4"), inflate::InflateAlg::LZ4);
}

// ============================================================================
// parse_inflate_alg — unknown algorithm → DALI_FAIL
// ============================================================================

TEST(InflateParamsTest, ParseInflateAlgUnknownThrows) {
  EXPECT_THROW(inflate::parse_inflate_alg("gzip"), DALIException);
}

TEST(InflateParamsTest, ParseInflateAlgEmptyThrows) {
  EXPECT_THROW(inflate::parse_inflate_alg(""), DALIException);
}

// ============================================================================
// to_string(InflateAlg) — LZ4
// ============================================================================

TEST(InflateParamsTest, ToStringLZ4) {
  EXPECT_EQ(inflate::to_string(inflate::InflateAlg::LZ4), "LZ4");
}

// ============================================================================
// to_string(InflateAlg) — unknown/default
// ============================================================================

TEST(InflateParamsTest, ToStringUnknown) {
  // Force an unrecognized enum value via cast
  auto unknown = static_cast<inflate::InflateAlg>(999);
  EXPECT_EQ(inflate::to_string(unknown), "<unknown>");
}

// ============================================================================
// Operator registration — verify schema exists for experimental__Inflate
// ============================================================================

TEST(InflateGpuTest, SchemaExists) {
  EXPECT_TRUE(SchemaRegistry::TryGetSchema("experimental__Inflate") != nullptr);
}

// ============================================================================
// Operator instantiation with valid OpSpec — covers DALI_REGISTER_OPERATOR
// (line 145 of inflate_gpu.cc), Inflate<GPUBackend> constructor, and
// parse_inflate_alg call path
// ============================================================================

TEST(InflateGpuTest, InstantiateWithValidSpec) {
  auto spec = OpSpec("experimental__Inflate")
      .AddArg("device", "gpu")
      .AddArg("max_batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("shape", std::vector<int>{10})
      .AddInput("input", StorageDevice::GPU)
      .AddOutput("output", StorageDevice::GPU);

  std::unique_ptr<OperatorBase> op;
  EXPECT_NO_THROW(op = InstantiateOperator(spec));
  EXPECT_NE(op, nullptr);
}

// ============================================================================
// Operator instantiation with invalid algorithm → parse_inflate_alg throws
// in the Inflate constructor
// ============================================================================

TEST(InflateGpuTest, InstantiateWithInvalidAlgorithmThrows) {
  auto spec = OpSpec("experimental__Inflate")
      .AddArg("device", "gpu")
      .AddArg("max_batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("shape", std::vector<int>{10})
      .AddArg("algorithm", "gzip")
      .AddInput("input", StorageDevice::GPU)
      .AddOutput("output", StorageDevice::GPU);

  EXPECT_THROW(InstantiateOperator(spec), DALIException);
}

}  // namespace testing
}  // namespace dali
