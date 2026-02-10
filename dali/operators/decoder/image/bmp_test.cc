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
void WriteLE(uint8_t *dst, T value) {
  for (size_t i = 0; i < sizeof(T); i++) {
    dst[i] = static_cast<uint8_t>(value & 0xFF);
    if constexpr (sizeof(T) > 1) {
      value >>= 8;
    }
  }
}

// ---------------------------------------------------------------------------
// Build a BITMAPCOREHEADER BMP buffer (header_size == 12)
// Layout: 14-byte BMP header + 12-byte DIB header + optional palette
// ---------------------------------------------------------------------------
std::vector<uint8_t> MakeBitmapCoreHeader(uint16_t width, uint16_t height,
                                          uint16_t bpp,
                                          bool color_palette = true) {
  size_t palette_size = 0;
  if (bpp <= 8) {
    size_t ncolors = 1u << bpp;
    palette_size = ncolors * 3;  // 3 bytes per entry for BITMAPCOREHEADER
  }
  std::vector<uint8_t> buf(26 + palette_size, 0);
  buf[0] = 'B'; buf[1] = 'M';
  // DIB header at offset 14
  WriteLE<uint32_t>(&buf[14], 12);           // header_size = 12
  WriteLE<uint16_t>(&buf[18], width);        // width
  WriteLE<uint16_t>(&buf[20], height);       // height
  WriteLE<uint16_t>(&buf[22], 1);            // color planes
  WriteLE<uint16_t>(&buf[24], bpp);          // bits per pixel
  // Fill palette if present
  if (bpp <= 8) {
    size_t ncolors = 1u << bpp;
    for (size_t i = 0; i < ncolors; i++) {
      if (color_palette) {
        // Make it a color palette: R != G != B for at least some entries
        buf[26 + i * 3 + 0] = static_cast<uint8_t>(i);       // B
        buf[26 + i * 3 + 1] = static_cast<uint8_t>(i + 1);   // G
        buf[26 + i * 3 + 2] = static_cast<uint8_t>(i + 2);   // R
      } else {
        // Grayscale palette: B == G == R
        uint8_t v = static_cast<uint8_t>(i);
        buf[26 + i * 3 + 0] = v;
        buf[26 + i * 3 + 1] = v;
        buf[26 + i * 3 + 2] = v;
      }
    }
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Build a BITMAPINFOHEADER BMP buffer (header_size == 40)
// Layout: 14-byte BMP header + 40-byte DIB header + optional palette
// ---------------------------------------------------------------------------
std::vector<uint8_t> MakeBitmapInfoHeader(int32_t width, int32_t height,
                                          uint16_t bpp,
                                          uint32_t compression_type = 0,
                                          uint32_t ncolors_arg = 0,
                                          bool color_palette = true) {
  size_t palette_size = 0;
  uint32_t effective_ncolors = ncolors_arg;
  if (bpp <= 8) {
    effective_ncolors = (ncolors_arg == 0) ? (1u << bpp) : ncolors_arg;
    palette_size = effective_ncolors * 4;  // 4 bytes per entry for BITMAPINFOHEADER
  }
  std::vector<uint8_t> buf(54 + palette_size, 0);
  buf[0] = 'B'; buf[1] = 'M';
  // DIB header at offset 14
  WriteLE<uint32_t>(&buf[14], 40);                  // header_size = 40
  WriteLE<int32_t>(&buf[18], width);                // width
  WriteLE<int32_t>(&buf[22], height);               // height
  WriteLE<uint16_t>(&buf[26], 1);                   // color planes
  WriteLE<uint16_t>(&buf[28], bpp);                 // bits per pixel
  WriteLE<uint32_t>(&buf[30], compression_type);    // compression type
  WriteLE<uint32_t>(&buf[34], 0);                   // image size (can be 0)
  WriteLE<int32_t>(&buf[38], 0);                    // h_resolution
  WriteLE<int32_t>(&buf[42], 0);                    // v_resolution
  WriteLE<uint32_t>(&buf[46], ncolors_arg);         // ncolors
  WriteLE<uint32_t>(&buf[50], 0);                   // important_colors
  // Fill palette if present
  if (bpp <= 8) {
    for (uint32_t i = 0; i < effective_ncolors; i++) {
      if (color_palette) {
        buf[54 + i * 4 + 0] = static_cast<uint8_t>(i);       // B
        buf[54 + i * 4 + 1] = static_cast<uint8_t>(i + 1);   // G
        buf[54 + i * 4 + 2] = static_cast<uint8_t>(i + 2);   // R
        buf[54 + i * 4 + 3] = 0;                               // A
      } else {
        uint8_t v = static_cast<uint8_t>(i);
        buf[54 + i * 4 + 0] = v;
        buf[54 + i * 4 + 1] = v;
        buf[54 + i * 4 + 2] = v;
        buf[54 + i * 4 + 3] = 0;
      }
    }
  }
  return buf;
}

// ============================================================================
// DALI_ENFORCE error paths in PeekShapeImpl
// ============================================================================

TEST(BmpImageTest, PeekShapeNullBufferThrows) {
  // Construct BmpImage with nullptr → PeekShape triggers DALI_ENFORCE(bmp != nullptr)
  BmpImage img(nullptr, 100, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

TEST(BmpImageTest, PeekShapeTooShortThrows) {
  // Buffer shorter than 18 bytes → DALI_ENFORCE(length >= 18)
  uint8_t buf[10] = {};
  BmpImage img(buf, sizeof(buf), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// BITMAPCOREHEADER (header_size == 12) branch
// ============================================================================

TEST(BmpImageTest, PeekShapeBitmapCoreHeader24bpp) {
  // 24bpp BITMAPCOREHEADER → w, h parsed, but c is not computed in this branch
  auto buf = MakeBitmapCoreHeader(40, 30, 24);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 30);  // h
  EXPECT_EQ(shape[1], 40);  // w
  EXPECT_EQ(shape[2], 0);   // c (not computed for BITMAPCOREHEADER)
}

TEST(BmpImageTest, PeekShapeBitmapCoreHeader8bppColorPalette) {
  // 8bpp with palette in BITMAPCOREHEADER → c is not computed in this branch
  auto buf = MakeBitmapCoreHeader(16, 16, 8, /*color_palette=*/true);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 16);
  EXPECT_EQ(shape[1], 16);
  EXPECT_EQ(shape[2], 0);  // c not computed for BITMAPCOREHEADER
}

TEST(BmpImageTest, PeekShapeBitmapCoreHeader8bppGrayPalette) {
  // 8bpp with grayscale palette in BITMAPCOREHEADER → c not computed
  auto buf = MakeBitmapCoreHeader(16, 16, 8, /*color_palette=*/false);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 16);
  EXPECT_EQ(shape[1], 16);
  EXPECT_EQ(shape[2], 0);  // c not computed for BITMAPCOREHEADER
}

TEST(BmpImageTest, PeekShapeBitmapCoreHeader4bpp) {
  // 4bpp in BITMAPCOREHEADER → palette parsed but c not computed
  auto buf = MakeBitmapCoreHeader(32, 32, 4, /*color_palette=*/true);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 32);
  EXPECT_EQ(shape[1], 32);
  EXPECT_EQ(shape[2], 0);  // c not computed for BITMAPCOREHEADER
}

// ============================================================================
// BITMAPINFOHEADER - number_of_channels: bpp=32, compression=RGB → 4
// ============================================================================

TEST(BmpImageTest, PeekShapeInfoHeader32bppRGB) {
  // 32bpp, compression_type=0 (RGB) → number_of_channels returns 4
  auto buf = MakeBitmapInfoHeader(100, 80, 32, /*compression_type=*/0);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 80);
  EXPECT_EQ(shape[1], 100);
  EXPECT_EQ(shape[2], 4);
}

// ============================================================================
// BITMAPINFOHEADER - number_of_channels: compression=BITFIELDS, bpp=16 → 3
// ============================================================================

TEST(BmpImageTest, PeekShapeInfoHeaderBitfields16bpp) {
  // compression_type=3 (BITFIELDS), bpp=16 → number_of_channels returns 3
  auto buf = MakeBitmapInfoHeader(64, 48, 16, /*compression_type=*/3);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 48);
  EXPECT_EQ(shape[1], 64);
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// BITMAPINFOHEADER - number_of_channels: compression=BITFIELDS, bpp=32 → 4
// ============================================================================

TEST(BmpImageTest, PeekShapeInfoHeaderBitfields32bpp) {
  // compression_type=3 (BITFIELDS), bpp=32 → number_of_channels returns 4
  auto buf = MakeBitmapInfoHeader(64, 48, 32, /*compression_type=*/3);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 48);
  EXPECT_EQ(shape[1], 64);
  EXPECT_EQ(shape[2], 4);
}

// ============================================================================
// BITMAPINFOHEADER - unsupported configuration → DALI_WARN + return 0
// ============================================================================

TEST(BmpImageTest, PeekShapeInfoHeaderUnsupportedConfig) {
  // compression_type=2 (RLE4), bpp=16 → falls through to DALI_WARN, returns c=0
  auto buf = MakeBitmapInfoHeader(64, 48, 16, /*compression_type=*/2);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 48);
  EXPECT_EQ(shape[1], 64);
  EXPECT_EQ(shape[2], 0);
}

// ============================================================================
// BITMAPINFOHEADER - ncolors==0 with bpp<=8 → ternary true branch (line 116)
// ============================================================================

TEST(BmpImageTest, PeekShapeInfoHeader8bppNcolorsZero) {
  // bpp=8, ncolors=0 in header → triggers ncolors = 1_uz << bpp = 256
  auto buf = MakeBitmapInfoHeader(20, 20, 8, /*compression_type=*/0,
                                  /*ncolors_arg=*/0, /*color_palette=*/true);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 20);
  EXPECT_EQ(shape[1], 20);
  EXPECT_EQ(shape[2], 3);  // color palette → 3
}

// ============================================================================
// BITMAPINFOHEADER - ncolors!=0 with bpp<=8 → ternary false branch (already
// covered, but with explicit ncolors value for completeness)
// ============================================================================

TEST(BmpImageTest, PeekShapeInfoHeader4bppNcolorsExplicit) {
  // bpp=4, ncolors=16 explicit → ternary false branch
  auto buf = MakeBitmapInfoHeader(20, 20, 4, /*compression_type=*/0,
                                  /*ncolors_arg=*/16, /*color_palette=*/true);
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 20);
  EXPECT_EQ(shape[1], 20);
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// Edge: header doesn't match either branch (length < 26 for core, < 50 for info)
// ============================================================================

TEST(BmpImageTest, PeekShapeUnknownHeaderFallthrough) {
  // header_size = 20 (not 12, not >= 40) → neither branch, h=w=c=0
  std::vector<uint8_t> buf(54, 0);
  buf[0] = 'B'; buf[1] = 'M';
  WriteLE<uint32_t>(&buf[14], 20);  // unknown header_size
  BmpImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 0);
  EXPECT_EQ(shape[1], 0);
  EXPECT_EQ(shape[2], 0);
}

}  // namespace testing
}  // namespace dali
