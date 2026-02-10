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
#include <fstream>
#include <string>
#include <vector>
#include "dali/operators/decoder/jpeg/jpeg_mem.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace jpeg {
namespace testing {

static std::vector<uint8_t> ReadFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static std::string TestJpegPath() {
  return dali::testing::dali_extra_path() +
         "/db/single/jpeg/100/swan-3584559_640.jpg";
}

// ============================================================================
// GetImageInfo tests
// ============================================================================

// Valid JPEG → returns true and sets dimensions
TEST(JpegMemTest, GetImageInfoValid) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  int width = 0, height = 0, components = 0;
  bool ok = GetImageInfo(data.data(), data.size(), &width, &height, &components);
  EXPECT_TRUE(ok);
  EXPECT_GT(width, 0);
  EXPECT_GT(height, 0);
  EXPECT_GT(components, 0);
}

// GetImageInfo with null output pointers → covers null branches (lines 549-551)
TEST(JpegMemTest, GetImageInfoNullOutputs) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  bool ok = GetImageInfo(data.data(), data.size(), nullptr, nullptr, nullptr);
  EXPECT_TRUE(ok);
}

// GetImageInfo with null data → returns false (line 554)
TEST(JpegMemTest, GetImageInfoNullData) {
  int width = 0, height = 0, components = 0;
  bool ok = GetImageInfo(nullptr, 100, &width, &height, &components);
  EXPECT_FALSE(ok);
  EXPECT_EQ(width, 0);
  EXPECT_EQ(height, 0);
  EXPECT_EQ(components, 0);
}

// GetImageInfo with zero datasize → returns false (line 554)
TEST(JpegMemTest, GetImageInfoZeroSize) {
  uint8_t data = 0;
  bool ok = GetImageInfo(&data, 0, nullptr, nullptr, nullptr);
  EXPECT_FALSE(ok);
}

// GetImageInfo with corrupt data → triggers setjmp/longjmp error (lines 568-570)
// or jpeg_read_header failure (lines 574-576)
TEST(JpegMemTest, GetImageInfoCorruptData) {
  std::vector<uint8_t> bad_data = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05};
  int width = -1, height = -1, components = -1;
  bool ok = GetImageInfo(bad_data.data(), bad_data.size(),
                         &width, &height, &components);
  EXPECT_FALSE(ok);
}

// ============================================================================
// Uncompress — error paths
// ============================================================================

// Bad ratio → returns nullptr (lines 86-87)
TEST(JpegMemTest, UncompressBadRatio) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.ratio = 3;  // invalid, must be 1, 2, 4, or 8
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_EQ(result, nullptr);
}

// Bad components → returns nullptr (lines 91-92)
TEST(JpegMemTest, UncompressBadComponents) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 5;  // invalid, must be 0, 1, or 3
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_EQ(result, nullptr);
}

// Null data → returns nullptr (line 96)
TEST(JpegMemTest, UncompressNullData) {
  UncompressFlags flags;
  auto result = Uncompress(nullptr, 100, flags);
  EXPECT_EQ(result, nullptr);
}

// Zero datasize → returns nullptr (line 96)
TEST(JpegMemTest, UncompressZeroSize) {
  uint8_t data = 0xFF;
  UncompressFlags flags;
  auto result = Uncompress(&data, 0, flags);
  EXPECT_EQ(result, nullptr);
}

// Incompatible stride (too small) → returns nullptr (lines 238-241)
TEST(JpegMemTest, UncompressStrideTooSmall) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.stride = 1;  // way too small
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_EQ(result, nullptr);
}

// Corrupt JPEG data → triggers CatchError/longjmp → returns nullptr (lines 113, 125)
TEST(JpegMemTest, UncompressCorruptData) {
  // A minimal JPEG start marker followed by garbage
  std::vector<uint8_t> bad_data = {0xFF, 0xD8, 0xFF, 0xE0,
                                   0x00, 0x10, 0x4A, 0x46,
                                   0x49, 0x46, 0x00, 0x01,
                                   0x01, 0x00, 0x00, 0x01,
                                   0x00, 0x01, 0x00, 0x00,
                                   0xFF, 0xDB};  // truncated DQT
  UncompressFlags flags;
  auto result = Uncompress(bad_data.data(), bad_data.size(), flags);
  EXPECT_EQ(result, nullptr);
}

// ============================================================================
// Uncompress — success paths
// ============================================================================

// Basic RGB decode with autodetect components → covers components==0 (line 141),
// normal decompression path, finish_decompress (line 443)
TEST(JpegMemTest, UncompressAutodetect) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 0;  // autodetect
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress with explicit 3 components (RGB) → covers case 3 (lines 148-151)
TEST(JpegMemTest, UncompressRGB) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress as grayscale → covers case 1 (lines 145-147)
TEST(JpegMemTest, UncompressGrayscale) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 1;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress with BGR color space → covers line 149 (JCS_EXT_BGR)
TEST(JpegMemTest, UncompressBGR) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.color_space = DALI_BGR;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress with ratio 2 → covers ratio check succeeding for non-1 values
TEST(JpegMemTest, UncompressRatio2) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.ratio = 2;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress with ratio 4 → covers another ratio branch
TEST(JpegMemTest, UncompressRatio4) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.ratio = 4;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress with ratio 8 → covers another ratio branch
TEST(JpegMemTest, UncompressRatio8) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.ratio = 8;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Decompress with fancy_upscaling = false → covers line 164
TEST(JpegMemTest, UncompressNoFancyUpscaling) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.fancy_upscaling = false;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// ============================================================================
// Uncompress — crop paths (requires LIBJPEG_TURBO_VERSION)
// ============================================================================

#if defined(LIBJPEG_TURBO_VERSION)
// Valid crop → covers lines 196-231
TEST(JpegMemTest, UncompressCropValid) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  // First get the image dimensions
  int width = 0, height = 0, components = 0;
  ASSERT_TRUE(GetImageInfo(data.data(), data.size(), &width, &height, &components));

  UncompressFlags flags;
  flags.components = 3;
  flags.crop = true;
  flags.crop_x = 10;
  flags.crop_y = 10;
  flags.crop_width = std::min(100, width - 20);
  flags.crop_height = std::min(100, height - 20);
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}

// Invalid crop window → returns nullptr (lines 202-210)
TEST(JpegMemTest, UncompressCropInvalid) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  UncompressFlags flags;
  flags.components = 3;
  flags.crop = true;
  flags.crop_x = 0;
  flags.crop_y = 0;
  flags.crop_width = 99999;  // exceeds image width
  flags.crop_height = 99999;
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_EQ(result, nullptr);
}

// Crop at origin (left_cor = 0) → covers line 214
TEST(JpegMemTest, UncompressCropAtOrigin) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  int width = 0, height = 0, components = 0;
  ASSERT_TRUE(GetImageInfo(data.data(), data.size(), &width, &height, &components));

  UncompressFlags flags;
  flags.components = 3;
  flags.crop = true;
  flags.crop_x = 0;  // left_cor will be 0
  flags.crop_y = 0;
  flags.crop_width = std::min(100, width);
  flags.crop_height = std::min(100, height);
  auto result = Uncompress(data.data(), data.size(), flags);
  EXPECT_NE(result, nullptr);
}
#endif

// ============================================================================
// Uncompress — truncated jpeg recovery paths
// ============================================================================

// Try recover truncated jpeg → covers lines 349-368 when data is truncated
TEST(JpegMemTest, UncompressTruncatedWithRecovery) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  // Truncate to ~1/4 of the file
  size_t truncated_size = data.size() / 4;
  ASSERT_GT(truncated_size, 100u);

  UncompressFlags flags;
  flags.components = 3;
  flags.try_recover_truncated_jpeg = true;
  flags.min_acceptable_fraction = 0.0f;  // accept any fraction
  auto result = Uncompress(data.data(), truncated_size, flags);
  // May or may not return data depending on how much was readable
  // The point is it doesn't crash and exercises the recovery path
}

// Truncated jpeg without recovery → covers lines 349-351 (no recovery),
// and Uncompress height_read_ != height_ path (lines 533-537)
TEST(JpegMemTest, UncompressTruncatedWithoutRecovery) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  // Truncate to ~1/4 of the file
  size_t truncated_size = data.size() / 4;
  ASSERT_GT(truncated_size, 100u);

  UncompressFlags flags;
  flags.components = 3;
  flags.try_recover_truncated_jpeg = false;
  flags.min_acceptable_fraction = 0.0f;  // accept any fraction
  auto result = Uncompress(data.data(), truncated_size, flags);
  // May return data with partial read (black-filled remainder)
  // or nullptr if too little was read
}

// Uncompress with fraction threshold → covers lines 521-528
TEST(JpegMemTest, UncompressFractionThreshold) {
  auto data = ReadFileBytes(TestJpegPath());
  ASSERT_FALSE(data.empty());

  // Truncate heavily
  size_t truncated_size = data.size() / 8;
  ASSERT_GT(truncated_size, 100u);

  UncompressFlags flags;
  flags.components = 3;
  flags.try_recover_truncated_jpeg = false;
  flags.min_acceptable_fraction = 0.9f;  // require 90% to accept
  auto result = Uncompress(data.data(), truncated_size, flags);
  // Likely returns nullptr because fraction_read < min_acceptable_fraction
}

}  // namespace testing
}  // namespace jpeg
}  // namespace dali
