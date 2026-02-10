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
#include <fstream>
#include <vector>
#include "dali/operators/decoder/image/tiff_libtiff.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace testing {

// Helper: read a file into a byte vector
static std::vector<uint8_t> ReadFileBytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << "Cannot open: " << path;
  auto size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(size);
  f.read(reinterpret_cast<char *>(buf.data()), size);
  return buf;
}

// ============================================================================
// Constructor with invalid data → DALI_ENFORCE(tif_) throws (line 243)
// ============================================================================

TEST(TiffLibtiffTest, InvalidDataConstructorThrows) {
  // Pure garbage data that TIFFClientOpen cannot parse
  uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03,
                       0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B};
  EXPECT_THROW(TiffImage_Libtiff(garbage, sizeof(garbage), DALI_RGB),
               DALIException);
}

// ============================================================================
// Constructor with zero-length data → DALI_ENFORCE(tif_) throws (line 243)
// ============================================================================

TEST(TiffLibtiffTest, EmptyBufferConstructorThrows) {
  uint8_t dummy = 0;
  EXPECT_THROW(TiffImage_Libtiff(&dummy, 0, DALI_RGB), DALIException);
}

// ============================================================================
// PeekShape with valid 8-bit RGB TIFF (lines 272-276)
// Also covers constructor success path, BufDecoderHelper::read,
// BufDecoderHelper::seek, BufDecoderHelper::size, BufDecoderHelper::map
// ============================================================================

TEST(TiffLibtiffTest, PeekShapeValidRGB) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF not found: " << path;

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_GT(shape[0], 0);  // height > 0
  EXPECT_GT(shape[1], 0);  // width > 0
  EXPECT_EQ(shape[2], 3);  // RGB → 3 channels
}

// ============================================================================
// PeekShape with grayscale TIFF
// ============================================================================

TEST(TiffLibtiffTest, PeekShapeGrayscale) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640_gray.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF not found: " << path;

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_GRAY);
  auto shape = img.PeekShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
  EXPECT_EQ(shape[2], 1);  // Grayscale → 1 channel
}

// ============================================================================
// PeekShape with palette TIFF → shape_[2] overridden to 3 (lines 265-268)
// ============================================================================

TEST(TiffLibtiffTest, PeekShapePalette) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-300572_640_palette.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF not found: " << path;

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
  EXPECT_EQ(shape[2], 3);  // palette → 3 channels
}

// ============================================================================
// CanDecode returns true for standard 8-bit, non-tiled, top-left (line 373-378)
// ============================================================================

TEST(TiffLibtiffTest, CanDecodeStandard) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_TRUE(img.CanDecode(DALI_RGB));
}

// ============================================================================
// CanDecode returns false for tiled TIFF (line 374: is_tiled_ check)
// ============================================================================

TEST(TiffLibtiffTest, CanDecodeTiledFalse) {
  std::string path = dali_extra_path() +
                     "/db/imgcodec/tiff/tiled/cat-111793_640_tiled_1024x1024.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF not found: " << path;

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_FALSE(img.CanDecode(DALI_RGB));
}

// ============================================================================
// CanDecode returns false for non-top-left orientation (line 376)
// ============================================================================

TEST(TiffLibtiffTest, CanDecodeNonTopLeftFalse) {
  std::string path = dali_extra_path() +
                     "/db/imgcodec/tiff/orientation/cat-1046544_640_rotate_270.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF not found: " << path;

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_FALSE(img.CanDecode(DALI_RGB));
}

// ============================================================================
// CanDecode returns false for palette TIFF (line 377: palette_ check)
// ============================================================================

TEST(TiffLibtiffTest, CanDecodePaletteFalse) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-300572_640_palette.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_FALSE(img.CanDecode(DALI_RGB));
}

// ============================================================================
// Decode valid RGB TIFF as DALI_RGB (lines 282-370, line 308)
// Covers: DecodeImpl main path, ConvertLineFromRGBX RGB branch (line 175-178)
// ============================================================================

TEST(TiffLibtiffTest, DecodeRGB) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// Decode valid RGB TIFF as DALI_GRAY (line 305-306)
// Covers: ConvertLineFromRGBX GRAY branch (line 169-170)
// ============================================================================

TEST(TiffLibtiffTest, DecodeGray) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_GRAY);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 1);  // grayscale output
}

// ============================================================================
// Decode valid RGB TIFF as DALI_BGR (line 309)
// Covers: ConvertLineFromRGBX BGR branch (lines 179-182)
// ============================================================================

TEST(TiffLibtiffTest, DecodeBGR) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_BGR);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// Decode valid RGB TIFF as DALI_YCbCr (line 310)
// Covers: ConvertLineFromRGBX YCbCr branch (lines 171-174)
// ============================================================================

TEST(TiffLibtiffTest, DecodeYCbCr) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_YCbCr);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// Decode valid RGB TIFF as DALI_ANY_DATA (lines 313-315)
// Covers: ConvertLineFromRGBX ANY_DATA branch (lines 162-165)
// ============================================================================

TEST(TiffLibtiffTest, DecodeAnyData) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_ANY_DATA);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 3);  // preserves original channel count
}

// ============================================================================
// Decode grayscale TIFF as DALI_GRAY
// Covers: ConvertLineFromMonochrome GRAY branch (lines 198-199)
// ============================================================================

TEST(TiffLibtiffTest, DecodeGrayscaleAsGray) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640_gray.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_GRAY);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 1);
}

// ============================================================================
// Decode grayscale TIFF as DALI_RGB
// Covers: ConvertLineFromMonochrome RGB branch (lines 203-204)
// ============================================================================

TEST(TiffLibtiffTest, DecodeGrayscaleAsRGB) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640_gray.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// Decode grayscale TIFF as DALI_YCbCr
// Covers: ConvertLineFromMonochrome YCbCr branch (lines 200-202)
// ============================================================================

TEST(TiffLibtiffTest, DecodeGrayscaleAsYCbCr) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640_gray.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_YCbCr);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// Decode grayscale TIFF as DALI_ANY_DATA
// Covers: ConvertLineFromMonochrome ANY_DATA/default branch (lines 205-209)
// ============================================================================

TEST(TiffLibtiffTest, DecodeGrayscaleAsAnyData) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640_gray.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_ANY_DATA);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_EQ(shape[2], 1);  // preserves original grayscale
}

// ============================================================================
// Decode tiled TIFF → CanDecode false → falls back to GenericImage (line 286)
// ============================================================================

TEST(TiffLibtiffTest, DecodeTiledFallsBackToGeneric) {
  std::string path = dali_extra_path() +
                     "/db/imgcodec/tiff/tiled/cat-111793_640_tiled_1024x1024.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  // Should fall back to GenericImage::DecodeImpl for tiled TIFFs
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
}

// ============================================================================
// Decode non-topleft orientation → CanDecode false → GenericImage fallback (line 286)
// ============================================================================

TEST(TiffLibtiffTest, DecodeNonTopLeftFallsBackToGeneric) {
  // Use mirror_horizontal (ORIENTATION_TOPRIGHT=2) which differs from
  // ORIENTATION_TOPLEFT but does NOT change image dimensions, avoiding
  // shape mismatches when GenericImage/OpenCV applies orientation correction.
  std::string path = dali_extra_path() +
                     "/db/imgcodec/tiff/orientation/cat-1046544_640_mirror_horizontal.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_FALSE(img.CanDecode(DALI_RGB));  // orientation != TOPLEFT
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
}

// ============================================================================
// Decode palette TIFF → CanDecode false → GenericImage fallback
// Covers palette_ path in constructor (lines 265-268) and CanDecode (line 377)
// ============================================================================

TEST(TiffLibtiffTest, DecodePaletteFallsBackToGeneric) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/0/cat-300572_640_palette.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  img.Decode();
  auto decoded = img.GetImage();
  EXPECT_NE(decoded, nullptr);
  auto shape = img.GetShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
}

// ============================================================================
// Decode compressed TIFF with rows_per_strip > 1 → sequential read path (lines 356-360)
// Uses a zstd-compressed TIFF file if available
// ============================================================================

TEST(TiffLibtiffTest, DecodeCompressedSequentialRead) {
  std::string path = dali_extra_path() +
                     "/db/single/tiff/various_encoding/cat-1046544_640_zstd.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Compressed TIFF not found: " << path;

  TiffImage_Libtiff img(buf.data(), buf.size(), DALI_RGB);
  // If CanDecode is true and compression != NONE with rows_per_strip > 1,
  // the sequential pre-read path (lines 356-360) is exercised.
  if (img.CanDecode(DALI_RGB)) {
    img.Decode();
    auto decoded = img.GetImage();
    EXPECT_NE(decoded, nullptr);
    auto shape = img.GetShape();
    EXPECT_GT(shape[0], 0);
    EXPECT_GT(shape[1], 0);
    EXPECT_EQ(shape[2], 3);
  }
}

}  // namespace testing
}  // namespace dali
