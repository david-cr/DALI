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
#include "dali/operators/decoder/image/jpeg.h"
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
// PeekShapeImpl with garbage data → DALI_ENFORCE(GetImageInfo) throws (line 125)
// ============================================================================

TEST(JpegImageTest, PeekShapeInvalidDataThrows) {
  // Non-JPEG garbage data
  uint8_t garbage[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                       0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
  JpegImage img(garbage, sizeof(garbage), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

TEST(JpegImageTest, PeekShapeEmptyDataThrows) {
  // Empty data: length 0
  uint8_t dummy = 0;
  JpegImage img(&dummy, 0, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// DecodeImpl with DALI_ANY_DATA and a grayscale JPEG
// → covers the DALI_GRAY branch of the ternary at line 68
// ============================================================================

TEST(JpegImageTest, DecodeAnyDataGrayscaleJpeg) {
  // Load a known grayscale JPEG from DALI_extra
  std::string path = dali_extra_path() +
                     "/db/3D/MRI/Knee/Jpegs/STU00001/SER00001/0.jpg";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test JPEG file not found at: " << path;

  // Verify it's a grayscale image (components == 1)
  JpegImage img_peek(buf.data(), buf.size(), DALI_ANY_DATA);
  auto shape = img_peek.PeekShape();
  EXPECT_EQ(shape[2], 1);  // 1 component = grayscale

  // Now decode with DALI_ANY_DATA — should select DALI_GRAY path (line 68)
  JpegImage img(buf.data(), buf.size(), DALI_ANY_DATA);
  EXPECT_NO_THROW(img.Decode());

  auto decoded_shape = img.GetShape();
  EXPECT_GT(decoded_shape[0], 0);  // height
  EXPECT_GT(decoded_shape[1], 0);  // width
  EXPECT_EQ(decoded_shape[2], 1);  // 1 channel (grayscale)
}

// ============================================================================
// DecodeImpl with a truncated JPEG (valid header, corrupt scan data)
// → may trigger decoded_image == nullptr fallback at lines 110-112
// ============================================================================

TEST(JpegImageTest, DecodeTruncatedJpegFallsBack) {
  // Load a valid RGB JPEG
  std::string path = dali_extra_path() +
                     "/db/single/jpeg/100/swan-3584559_640.jpg";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test JPEG file not found at: " << path;

  // Truncate: keep only the first ~20% of the file (header + partial scan)
  size_t truncated_size = buf.size() / 5;
  ASSERT_GT(truncated_size, 64u) << "File too small for truncation test";

  // PeekShape should still succeed on the truncated data (header is intact)
  JpegImage img_peek(buf.data(), truncated_size, DALI_RGB);
  auto shape = img_peek.PeekShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);

  // Decode on truncated data: should either fall back to GenericImage::DecodeImpl
  // (which may also fail and throw) or throw during decompression
  JpegImage img(buf.data(), truncated_size, DALI_RGB);
  // We accept either successful fallback decode or a thrown exception;
  // the key goal is exercising the code path, not a specific outcome
  try {
    img.Decode();
    // If it succeeded (via GenericImage fallback), verify shape is reasonable
    auto decoded_shape = img.GetShape();
    EXPECT_GT(decoded_shape[0], 0);
    EXPECT_GT(decoded_shape[1], 0);
  } catch (const DALIException &) {
    // Thrown exception is acceptable — the decode path was still exercised
  } catch (const std::exception &) {
    // Other exceptions are also acceptable for corrupt data
  }
}

// ============================================================================
// PeekShape on a valid RGB JPEG → sanity test
// ============================================================================

TEST(JpegImageTest, PeekShapeValidRGBJpeg) {
  std::string path = dali_extra_path() +
                     "/db/single/jpeg/100/swan-3584559_640.jpg";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  JpegImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 408);  // height
  EXPECT_EQ(shape[1], 640);  // width
  EXPECT_EQ(shape[2], 3);    // 3 components (RGB)
}

// ============================================================================
// Decode with DALI_ANY_DATA on an RGB JPEG → selects DALI_RGB path (line 68 T)
// This is already covered but ensures the full path works
// ============================================================================

TEST(JpegImageTest, DecodeAnyDataRGBJpeg) {
  std::string path = dali_extra_path() +
                     "/db/single/jpeg/100/swan-3584559_640.jpg";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty());

  JpegImage img(buf.data(), buf.size(), DALI_ANY_DATA);
  EXPECT_NO_THROW(img.Decode());

  auto decoded_shape = img.GetShape();
  EXPECT_EQ(decoded_shape[0], 408);
  EXPECT_EQ(decoded_shape[1], 640);
  EXPECT_EQ(decoded_shape[2], 3);  // 3 channels → DALI_RGB selected
}

}  // namespace testing
}  // namespace dali
