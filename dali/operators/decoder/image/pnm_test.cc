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
#include <vector>
#include "dali/operators/decoder/image/pnm.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

// Helper: make a PNM buffer from a string
static std::vector<uint8_t> MakePnm(const std::string &header) {
  return std::vector<uint8_t>(header.begin(), header.end());
}

// ============================================================================
// PeekShapeImpl with null buffer → DALI_ENFORCE(pnm) throws (line 25)
// ============================================================================

TEST(PnmImageTest, PeekShapeNullBufferThrows) {
  PnmImage img(nullptr, 0, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShapeImpl with length ≤ 1 → DALI_ENFORCE(at_ptr < end_ptr) throws (line 30)
// ============================================================================

TEST(PnmImageTest, PeekShapeTooShortThrows) {
  uint8_t data[] = {'P'};
  PnmImage img(data, 1, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with valid P6 (RGB) format → 3 channels
// ============================================================================

TEST(PnmImageTest, PeekShapeP6RGB) {
  // P6\n100 200\n255\n
  auto buf = MakePnm("P6\n100 200\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 200);  // height
  EXPECT_EQ(shape[1], 100);  // width
  EXPECT_EQ(shape[2], 3);    // channels (RGB)
}

// ============================================================================
// PeekShape with P5 (grayscale PGM) format → 1 channel
// ============================================================================

TEST(PnmImageTest, PeekShapeP5Grayscale) {
  auto buf = MakePnm("P5\n320 240\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_GRAY);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 240);  // height
  EXPECT_EQ(shape[1], 320);  // width
  EXPECT_EQ(shape[2], 1);    // channels (grayscale)
}

// ============================================================================
// PeekShape with P3 (ASCII RGB) format → 3 channels
// ============================================================================

TEST(PnmImageTest, PeekShapeP3RGB) {
  auto buf = MakePnm("P3\n50 75\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 75);   // height
  EXPECT_EQ(shape[1], 50);   // width
  EXPECT_EQ(shape[2], 3);    // channels
}

// ============================================================================
// PeekShape with comments in the header → covers lines 48-52
// ============================================================================

TEST(PnmImageTest, PeekShapeWithComment) {
  // Comment after width digits, before separator.
  // The comment's trailing newline acts as the whitespace separator.
  auto buf = MakePnm("P6\n100#this_is_a_comment\n200\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 200);  // height
  EXPECT_EQ(shape[1], 100);  // width
  EXPECT_EQ(shape[2], 3);    // channels
}

TEST(PnmImageTest, PeekShapeWithMultipleComments) {
  // Two comments: one after width, one after height
  auto buf = MakePnm("P6\n100#comment1\n200#comment2\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 200);
  EXPECT_EQ(shape[1], 100);
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// PeekShape with multiple consecutive spaces → covers line 58 loop (true branch)
// ============================================================================

TEST(PnmImageTest, PeekShapeMultipleSpaces) {
  // Multiple spaces between width and height
  auto buf = MakePnm("P6\n100   200\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 200);
  EXPECT_EQ(shape[1], 100);
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// PeekShape with truncated data in main loop → line 44 failure
// ============================================================================

TEST(PnmImageTest, PeekShapeTruncatedInLoopThrows) {
  // Truncated: has magic and start of width but no height
  auto buf = MakePnm("P6\n10");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with non-digit character where digit expected → line 61 failure
// ============================================================================

TEST(PnmImageTest, PeekShapeNonDigitThrows) {
  // 'X' where a digit is expected (after the first whitespace)
  auto buf = MakePnm("P6\nXYZ 200\n255\n");
  PnmImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with P1 (PBM bitmap) format → 1 channel
// ============================================================================

TEST(PnmImageTest, PeekShapeP1Bitmap) {
  auto buf = MakePnm("P1\n10 20\n");
  PnmImage img(buf.data(), buf.size(), DALI_GRAY);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 20);   // height
  EXPECT_EQ(shape[1], 10);   // width
  EXPECT_EQ(shape[2], 1);    // channels (bitmap = 1)
}

}  // namespace testing
}  // namespace dali
