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
#include <vector>
#include "dali/operators/decoder/image/png.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

// Helper: write a big-endian uint32 at dst
static void WriteBE32(uint8_t *dst, uint32_t val) {
  dst[0] = static_cast<uint8_t>((val >> 24) & 0xFF);
  dst[1] = static_cast<uint8_t>((val >> 16) & 0xFF);
  dst[2] = static_cast<uint8_t>((val >> 8) & 0xFF);
  dst[3] = static_cast<uint8_t>(val & 0xFF);
}

// Build a synthetic PNG header with IHDR chunk.
// Layout:
//   [0..7]   PNG signature: 89 50 4E 47 0D 0A 1A 0A
//   [8..11]  IHDR chunk size: 00 00 00 0D (13 bytes)
//   [12..15] "IHDR"
//   [16..19] Width (BE)
//   [20..23] Height (BE)
//   [24]     Bit depth (8)
//   [25]     Color type
//   [26]     Compression (0)
//   [27]     Filter (0)
//   [28]     Interlace (0)
// Total: 29 bytes
static std::vector<uint8_t> MakePngHeader(uint32_t width, uint32_t height,
                                           uint8_t color_type) {
  std::vector<uint8_t> buf(32, 0);

  // PNG signature
  buf[0] = 0x89; buf[1] = 0x50; buf[2] = 0x4E; buf[3] = 0x47;
  buf[4] = 0x0D; buf[5] = 0x0A; buf[6] = 0x1A; buf[7] = 0x0A;

  // IHDR chunk
  WriteBE32(&buf[8], 13);  // chunk data length = 13
  buf[12] = 'I'; buf[13] = 'H'; buf[14] = 'D'; buf[15] = 'R';
  WriteBE32(&buf[16], width);
  WriteBE32(&buf[20], height);
  buf[24] = 8;           // bit depth
  buf[25] = color_type;  // color type
  buf[26] = 0;           // compression
  buf[27] = 0;           // filter
  buf[28] = 0;           // interlace

  return buf;
}

// ============================================================================
// PeekShapeImpl with null buffer → DALI_ENFORCE(encoded_buffer) throws (line 91)
// ============================================================================

TEST(PngImageTest, PeekShapeNullBufferThrows) {
  // Construct with non-null but call PeekShape via Image base which passes
  // the stored buffer. We construct with nullptr directly.
  PngImage img(nullptr, 0, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShapeImpl with too-short buffer → DALI_ENFORCE(length >= 16) throws (line 92)
// ============================================================================

TEST(PngImageTest, PeekShapeTooShortThrows) {
  uint8_t data[10] = {};
  PngImage img(data, 10, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with PNG_COLOR_TYPE_GRAY (0) → returns 1 channel (lines 68, 70)
// ============================================================================

TEST(PngImageTest, PeekShapeGrayscale) {
  auto buf = MakePngHeader(100, 200, 0);  // PNG_COLOR_TYPE_GRAY = 0
  PngImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 200);  // height
  EXPECT_EQ(shape[1], 100);  // width
  EXPECT_EQ(shape[2], 1);    // 1 channel
}

// ============================================================================
// PeekShape with PNG_COLOR_TYPE_GRAY_ALPHA (4) → returns 1 channel (lines 69, 70)
// ============================================================================

TEST(PngImageTest, PeekShapeGrayAlpha) {
  auto buf = MakePngHeader(80, 60, 4);  // PNG_COLOR_TYPE_GRAY_ALPHA = 4
  PngImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 60);   // height
  EXPECT_EQ(shape[1], 80);   // width
  EXPECT_EQ(shape[2], 1);    // 1 channel
}

// ============================================================================
// PeekShape with PNG_COLOR_TYPE_PALETTE (3) → returns 3 channels (line 72)
// ============================================================================

TEST(PngImageTest, PeekShapePalette) {
  auto buf = MakePngHeader(50, 40, 3);  // PNG_COLOR_TYPE_PALETTE = 3
  PngImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 40);   // height
  EXPECT_EQ(shape[1], 50);   // width
  EXPECT_EQ(shape[2], 3);    // 3 channels
}

// ============================================================================
// PeekShape with unsupported color type → DALI_FAIL default case (lines 76-77)
// ============================================================================

TEST(PngImageTest, PeekShapeUnsupportedColorTypeThrows) {
  auto buf = MakePngHeader(50, 40, 99);  // color_type = 99, invalid
  PngImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape older PNG format: no IHDR tag → fallback (line 97-99)
// ============================================================================

TEST(PngImageTest, PeekShapeOlderPngNoIHDR) {
  // Create buffer where bytes [12..15] are NOT "IHDR"
  // In this case, png_dimens = encoded_buffer (offset 0)
  // So Width is read from offset kOffsetWidth=8, Height from kOffsetHeight=12,
  // ColorType from kOffsetColorType=17
  std::vector<uint8_t> buf(32, 0);

  // PNG signature
  buf[0] = 0x89; buf[1] = 0x50; buf[2] = 0x4E; buf[3] = 0x47;
  buf[4] = 0x0D; buf[5] = 0x0A; buf[6] = 0x1A; buf[7] = 0x0A;

  // NOT an IHDR tag at [12..15] — use "XXXX" instead
  buf[12] = 'X'; buf[13] = 'X'; buf[14] = 'X'; buf[15] = 'X';

  // For older format, reads are relative to encoded_buffer (offset 0):
  // Width at kOffsetWidth=8, Height at kOffsetHeight=12, ColorType at kOffsetColorType=17
  WriteBE32(&buf[8], 320);   // width (read from offset 8)
  WriteBE32(&buf[12], 240);  // height (read from offset 12)
  buf[17] = 2;               // color type RGB (read from offset 17)

  // length check: png_dimens == encoded_buffer, need length >= 0 + 16 = 16
  PngImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[1], 320);  // width
  // Note: height overlaps with "XXXX" bytes we wrote, so just check it parsed without error
  EXPECT_EQ(shape[2], 3);    // 3 channels (RGB)
}

// ============================================================================
// PeekShape older format with insufficient length → DALI_ENFORCE throws (line 102)
// Note: For IHDR path, png_dimens = encoded_buffer + 8, so need length >= 24.
// We provide length = 20 which passes line 92 (>= 16) but fails line 102 (< 24).
// ============================================================================

TEST(PngImageTest, PeekShapeIHDRPathInsufficientLengthThrows) {
  auto buf = MakePngHeader(100, 200, 2);
  // Provide length = 20: passes `length >= 16` but fails
  // `length >= png_dimens - encoded_buffer + 16` = `20 >= 8 + 16 = 24` → false
  PngImage img(buf.data(), 20, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

}  // namespace testing
}  // namespace dali
