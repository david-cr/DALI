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
#include "dali/operators/decoder/image/bmp.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

// Helper: write a little-endian value at dst
template <typename T>
void WriteLEImg(uint8_t *dst, T value) {
  for (size_t i = 0; i < sizeof(T); i++) {
    dst[i] = static_cast<uint8_t>(value & 0xFF);
    if constexpr (sizeof(T) > 1) {
      value >>= 8;
    }
  }
}

// Build a complete valid 1x1 24bpp BMP image that cv::imdecode can decode
std::vector<uint8_t> MakeValid1x1Bmp() {
  // BMP file header (14) + DIB BITMAPINFOHEADER (40) + pixel data (4 = 3 BGR + 1 padding)
  std::vector<uint8_t> buf(58, 0);
  // BMP file header
  buf[0] = 'B'; buf[1] = 'M';
  WriteLEImg<uint32_t>(&buf[2], 58);      // file size
  WriteLEImg<uint32_t>(&buf[10], 54);     // pixel data offset
  // DIB header
  WriteLEImg<uint32_t>(&buf[14], 40);     // header size
  WriteLEImg<int32_t>(&buf[18], 1);       // width
  WriteLEImg<int32_t>(&buf[22], 1);       // height
  WriteLEImg<uint16_t>(&buf[26], 1);      // color planes
  WriteLEImg<uint16_t>(&buf[28], 24);     // bpp
  WriteLEImg<uint32_t>(&buf[30], 0);      // compression = RGB
  WriteLEImg<uint32_t>(&buf[34], 4);      // image size (padded row)
  // Pixel data at offset 54: one BGR pixel + 1 byte padding
  buf[54] = 128;  // B
  buf[55] = 64;   // G
  buf[56] = 32;   // R
  buf[57] = 0;    // padding
  return buf;
}

// ============================================================================
// GetImage() before Decode() → DALI_ENFORCE(decoded_) throws (line 36)
// ============================================================================

TEST(ImageBaseTest, GetImageBeforeDecodeThrows) {
  uint8_t dummy[64] = {};
  dummy[0] = 'B'; dummy[1] = 'M';
  BmpImage img(dummy, sizeof(dummy), DALI_RGB);
  EXPECT_THROW(img.GetImage(), DALIException);
}

// ============================================================================
// GetShape() before Decode() → DALI_ENFORCE(decoded_) throws (line 45)
// ============================================================================

TEST(ImageBaseTest, GetShapeBeforeDecodeThrows) {
  uint8_t dummy[64] = {};
  dummy[0] = 'B'; dummy[1] = 'M';
  BmpImage img(dummy, sizeof(dummy), DALI_RGB);
  EXPECT_THROW(img.GetShape(), DALIException);
}

// ============================================================================
// Decode() called twice → DALI_ENFORCE(!decoded_) throws (line 27)
// ============================================================================

TEST(ImageBaseTest, DecodeCalledTwiceThrows) {
  auto buf = MakeValid1x1Bmp();
  BmpImage img(buf.data(), buf.size(), DALI_RGB);

  // First decode should succeed
  EXPECT_NO_THROW(img.Decode());

  // Verify the image was decoded correctly
  auto shape = img.GetShape();
  EXPECT_EQ(shape[0], 1);  // height
  EXPECT_EQ(shape[1], 1);  // width
  EXPECT_EQ(shape[2], 3);  // channels (RGB)

  // Second decode should throw
  EXPECT_THROW(img.Decode(), DALIException);
}

}  // namespace testing
}  // namespace dali
