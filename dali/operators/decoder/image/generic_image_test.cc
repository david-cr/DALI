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
#include "dali/operators/decoder/image/generic_image.h"
#include "dali/operators/decoder/image/bmp.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

// Helper: write a little-endian value at dst
template <typename T>
void WriteLEGeneric(uint8_t *dst, T value) {
  for (size_t i = 0; i < sizeof(T); i++) {
    dst[i] = static_cast<uint8_t>(value & 0xFF);
    if constexpr (sizeof(T) > 1) {
      value >>= 8;
    }
  }
}

// Build a BITMAPINFOHEADER BMP buffer (header only, no pixel data)
// This produces a valid header that BmpImage::PeekShapeImpl can parse,
// but cv::imdecode will fail on because pixel data is missing/corrupt.
std::vector<uint8_t> MakeTruncatedBmp(int32_t width, int32_t height,
                                      uint16_t bpp = 24,
                                      uint32_t compression_type = 0) {
  // Only header (54 bytes), no pixel data
  std::vector<uint8_t> buf(54, 0);
  buf[0] = 'B'; buf[1] = 'M';
  // File size field (offset 2): claim full size but only provide header
  uint32_t row_bytes = ((width * bpp + 31) / 32) * 4;
  uint32_t pixel_data_size = row_bytes * abs(height);
  uint32_t file_size = 54 + pixel_data_size;
  WriteLEGeneric<uint32_t>(&buf[2], file_size);
  // Pixel data offset (offset 10)
  WriteLEGeneric<uint32_t>(&buf[10], 54);
  // DIB header at offset 14
  WriteLEGeneric<uint32_t>(&buf[14], 40);                // header_size = 40
  WriteLEGeneric<int32_t>(&buf[18], width);              // width
  WriteLEGeneric<int32_t>(&buf[22], height);             // height
  WriteLEGeneric<uint16_t>(&buf[26], 1);                 // color planes
  WriteLEGeneric<uint16_t>(&buf[28], bpp);               // bits per pixel
  WriteLEGeneric<uint32_t>(&buf[30], compression_type);  // compression type
  WriteLEGeneric<uint32_t>(&buf[34], pixel_data_size);   // image size
  return buf;
}

// ============================================================================
// GenericImage::PeekShapeImpl always throws DALI_FAIL (line 103-105)
// ============================================================================

TEST(GenericImageTest, PeekShapeImplAlwaysThrows) {
  // GenericImage::PeekShapeImpl calls DALI_FAIL because the format is unknown
  uint8_t dummy[64] = {};
  GenericImage img(dummy, sizeof(dummy), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// GenericImage::DecodeImpl line 50: cv::imdecode returns null Mat
// Exercise through BmpImage (inherits DecodeImpl from GenericImage)
// with a valid BMP header but truncated/missing pixel data.
// ============================================================================

TEST(GenericImageTest, DecodeImplUnsupportedImageThrows) {
  // Create BMP with valid header (PeekShapeImpl succeeds) but no pixel data
  // so cv::imdecode fails → decoded_image.data == nullptr
  auto buf = MakeTruncatedBmp(100, 100, 24);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);

  // PeekShape should succeed since the header is valid
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 100);
  EXPECT_EQ(shape[1], 100);
  EXPECT_EQ(shape[2], 3);

  // Decode should fail because cv::imdecode can't decode truncated data
  EXPECT_THROW(img.Decode(), DALIException);
}

// Also test with a different image type to ensure the error path is robust
TEST(GenericImageTest, DecodeImplTruncatedGrayBmpThrows) {
  auto buf = MakeTruncatedBmp(50, 50, 24);
  BmpImage img(buf.data(), buf.size(), DALI_GRAY);

  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 50);
  EXPECT_EQ(shape[1], 50);

  EXPECT_THROW(img.Decode(), DALIException);
}

// Test with completely garbage data that has a BMP header structure
// but invalid pixel content
TEST(GenericImageTest, DecodeImplGarbagePixelDataThrows) {
  // Build a BMP header for a 10x10 24bpp image
  // Then append garbage that's shorter than the expected pixel data
  auto buf = MakeTruncatedBmp(10, 10, 24);
  // Append some garbage bytes (not enough for 10*10*3 pixels)
  for (int i = 0; i < 50; i++) {
    buf.push_back(static_cast<uint8_t>(i ^ 0xAB));
  }
  BmpImage img(buf.data(), buf.size(), DALI_RGB);

  // PeekShape reads only the header - should work
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 10);
  EXPECT_EQ(shape[1], 10);
  EXPECT_EQ(shape[2], 3);

  // Decode should fail - not enough pixel data
  EXPECT_THROW(img.Decode(), DALIException);
}

}  // namespace testing
}  // namespace dali
