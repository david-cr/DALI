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
#include "dali/operators/decoder/image/webp.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

// Build a minimal WebP-like buffer with a RIFF header and VP8 chunk identifier.
// Layout:
//   [0..3]   "RIFF"
//   [4..7]   file size - 8 (dummy)
//   [8..11]  "WEBP"
//   [12..15] VP8 chunk type (4 chars: e.g. "VP8 ", "VP8L", "VP8X")
//   [16..]   chunk data (depends on format)
static std::vector<uint8_t> MakeWebpHeader(const char chunk_type[4], size_t total_size = 64) {
  std::vector<uint8_t> buf(total_size, 0);
  // RIFF header
  buf[0] = 'R'; buf[1] = 'I'; buf[2] = 'F'; buf[3] = 'F';
  // file size (dummy, not checked by PeekShapeImpl)
  uint32_t fsize = static_cast<uint32_t>(total_size - 8);
  buf[4] = fsize & 0xFF; buf[5] = (fsize >> 8) & 0xFF;
  buf[6] = (fsize >> 16) & 0xFF; buf[7] = (fsize >> 24) & 0xFF;
  // WEBP signature
  buf[8] = 'W'; buf[9] = 'E'; buf[10] = 'B'; buf[11] = 'P';
  // VP8 chunk type at offset 12
  buf[12] = chunk_type[0]; buf[13] = chunk_type[1];
  buf[14] = chunk_type[2]; buf[15] = chunk_type[3];
  return buf;
}

// Helper to write a little-endian uint16
static void WriteLE16(uint8_t *dst, uint16_t val) {
  dst[0] = static_cast<uint8_t>(val & 0xFF);
  dst[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
}

// Helper to write a little-endian uint32
static void WriteLE32(uint8_t *dst, uint32_t val) {
  dst[0] = static_cast<uint8_t>(val & 0xFF);
  dst[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
  dst[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
  dst[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

// ============================================================================
// PeekShape with null buffer → DALI_ENFORCE(encoded_buffer) throws (line 25)
// ============================================================================

TEST(WebpImageTest, PeekShapeNullBufferThrows) {
  WebpImage img(nullptr, 0, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with buffer too short → DALI_ENFORCE(length >= 16) throws (line 26)
// ============================================================================

TEST(WebpImageTest, PeekShapeTooShortThrows) {
  uint8_t data[10] = {};
  WebpImage img(data, 10, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// VP8 lossy: buffer >= 16 but < 30 → DALI_ENFORCE(length >= 30) throws (line 33)
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LossyTooShortThrows) {
  auto buf = MakeWebpHeader("VP8 ", 20);  // length=20, which is < 30
  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// VP8 lossy: sync code mismatch → DALI_FAIL (lines 38-41)
// Sync code at vp8_data[11..13] should be 0x9D 0x01 0x2A
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LossySyncCodeMismatchThrows) {
  auto buf = MakeWebpHeader("VP8 ", 64);
  // vp8_data = buf.data() + 12, so sync code at buf[23..25]
  // Set wrong sync code
  buf[23] = 0x00; buf[24] = 0x00; buf[25] = 0x00;
  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// VP8 lossy: valid sync code → returns {H, W, 3} (lines 45-49)
// Already covered but included for completeness as a sanity test
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LossyValid) {
  auto buf = MakeWebpHeader("VP8 ", 64);
  // vp8_data starts at buf[12]
  // Sync code at vp8_data[11..13] = buf[23..25]
  buf[23] = 0x9D; buf[24] = 0x01; buf[25] = 0x2A;
  // Width at vp8_data[14..15] = buf[26..27] (LE uint16, masked with 0x3FFF)
  WriteLE16(&buf[26], 320);
  // Height at vp8_data[16..17] = buf[28..29] (LE uint16, masked with 0x3FFF)
  WriteLE16(&buf[28], 240);

  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 240);  // height
  EXPECT_EQ(shape[1], 320);  // width
  EXPECT_EQ(shape[2], 3);    // VP8 always RGB
}

// ============================================================================
// VP8L lossless: buffer >= 16 but < 25 → DALI_ENFORCE(length >= 25) throws (line 53)
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LLosslessTooShortThrows) {
  auto buf = MakeWebpHeader("VP8L", 20);  // length=20, which is < 25
  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// VP8L lossless: signature byte mismatch → DALI_FAIL (lines 56-58)
// Signature at vp8_data[8] = buf[20] should be 0x2F
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LSignatureMismatchThrows) {
  auto buf = MakeWebpHeader("VP8L", 64);
  // vp8_data[8] = buf[20], set to wrong value
  buf[20] = 0x00;
  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// VP8L lossless: valid → returns {H, W, 3 + alpha} (lines 62-68)
// Already covered but included as sanity test
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LLosslessValid) {
  auto buf = MakeWebpHeader("VP8L", 64);
  // Signature byte at vp8_data[8] = buf[20]
  buf[20] = 0x2F;
  // Features at vp8_data[9..12] = buf[21..24] (LE uint32)
  // W = (features & 0x3FFF) + 1, H = ((features & 0x0FFFC000) >> 14) + 1
  // alpha = (features & 0x10000000) >> 28
  // Want W=200, H=100, alpha=0:
  //   W-1=199=0xC7, H-1=99=0x63
  //   features = (99 << 14) | 199 = 0x18C0C7
  uint32_t features = (99 << 14) | 199;
  WriteLE32(&buf[21], features);

  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 100);  // height
  EXPECT_EQ(shape[1], 200);  // width
  EXPECT_EQ(shape[2], 3);    // no alpha → 3
}

// ============================================================================
// VP8L lossless with alpha bit → returns 4 channels (line 65, 68)
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8LLosslessWithAlpha) {
  auto buf = MakeWebpHeader("VP8L", 64);
  buf[20] = 0x2F;  // signature byte
  // W=200, H=100, alpha=1:
  //   features = 0x10000000 | (99 << 14) | 199
  uint32_t features = 0x10000000 | (99 << 14) | 199;
  WriteLE32(&buf[21], features);

  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 100);
  EXPECT_EQ(shape[1], 200);
  EXPECT_EQ(shape[2], 4);    // alpha → 3 + 1 = 4
}

// ============================================================================
// VP8X extended format → DALI_FAIL (lines 69-71)
// ============================================================================

TEST(WebpImageTest, PeekShapeVP8XExtendedThrows) {
  auto buf = MakeWebpHeader("VP8X", 64);
  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// Unrecognized header → DALI_FAIL (lines 73-75)
// ============================================================================

TEST(WebpImageTest, PeekShapeUnrecognizedHeaderThrows) {
  auto buf = MakeWebpHeader("XXXX", 64);
  WebpImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

}  // namespace testing
}  // namespace dali
