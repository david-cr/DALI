// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include "dali/test/dali_test.h"
#include "dali/operators/decoder/image/tiff.h"
#include "dali/core/error_handling.h"
#include "dali/test/dali_test_config.h"

namespace dali {

class TiffDecoderTest : public DALITest {
 protected:
  std::string bin = {'\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47', '\x48', '\x49',
                     '\x4a', '\x4b', '\x4c', '\x4d', '\x4e', '\x4f', '\x50'};
};

TEST_F(TiffDecoderTest, TiffBufferBigEndianTest) {
  TiffBuffer buf(bin);
  EXPECT_EQ(65, buf.Read<int8_t>());
  EXPECT_EQ(16961, buf.Read<int16_t>());
  EXPECT_EQ(1145258561, buf.Read<int32_t>());
  EXPECT_EQ(5208208757389214273, buf.Read<int64_t>());
  EXPECT_EQ(65, buf.Read<uint8_t>());
  EXPECT_EQ(16961, buf.Read<uint16_t>());
  EXPECT_EQ(1145258561, buf.Read<uint32_t>());
  EXPECT_EQ(5208208757389214273, buf.Read<uint64_t>());
}


TEST_F(TiffDecoderTest, TiffBufferLittleEndianTest) {
  TiffBuffer buf(bin, true);
  EXPECT_EQ(65, buf.Read<int8_t>());
  EXPECT_EQ(16706, buf.Read<int16_t>());
  EXPECT_EQ(1094861636, buf.Read<int32_t>());
  EXPECT_EQ(4702394921427289928, buf.Read<int64_t>());
  EXPECT_EQ(65, buf.Read<uint8_t>());
  EXPECT_EQ(16706, buf.Read<uint16_t>());
  EXPECT_EQ(1094861636, buf.Read<uint32_t>());
  EXPECT_EQ(4702394921427289928, buf.Read<uint64_t>());
}


TEST_F(TiffDecoderTest, TiffBufferOffsetTest) {
  TiffBuffer buf_big(bin);
  EXPECT_EQ(75, buf_big.Read<int8_t>(10));
  EXPECT_EQ(19274, buf_big.Read<int16_t>(9));
  TiffBuffer buf_little(bin, true);
  EXPECT_EQ(75, buf_little.Read<int8_t>(10));
  EXPECT_EQ(1145390663, buf_little.Read<int32_t>(3));
  EXPECT_EQ(4774735094265366601, buf_little.Read<int64_t>(1));
}

// ============================================================================
// Helpers for building synthetic TIFF data
// ============================================================================

namespace {

static void WriteBE16(uint8_t *dst, uint16_t val) {
  dst[0] = static_cast<uint8_t>((val >> 8) & 0xFF);
  dst[1] = static_cast<uint8_t>(val & 0xFF);
}

static void WriteBE32(uint8_t *dst, uint32_t val) {
  dst[0] = static_cast<uint8_t>((val >> 24) & 0xFF);
  dst[1] = static_cast<uint8_t>((val >> 16) & 0xFF);
  dst[2] = static_cast<uint8_t>((val >> 8) & 0xFF);
  dst[3] = static_cast<uint8_t>(val & 0xFF);
}

static void WriteLE16(uint8_t *dst, uint16_t val) {
  dst[0] = static_cast<uint8_t>(val & 0xFF);
  dst[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
}

static void WriteLE32(uint8_t *dst, uint32_t val) {
  dst[0] = static_cast<uint8_t>(val & 0xFF);
  dst[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
  dst[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
  dst[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

constexpr uint16_t kWidthTag = 256;
constexpr uint16_t kHeightTag = 257;
constexpr uint16_t kPhotometricTag = 262;
constexpr uint16_t kSamplesPerPixelTag = 277;
constexpr uint16_t kTypeWord = 3;
constexpr uint16_t kTypeDWord = 4;
constexpr uint16_t kPhotometricPalette = 3;

struct IfdEntry {
  uint16_t tag;
  uint16_t type;
  uint32_t count;
  uint32_t value;
};

// Build a synthetic big-endian TIFF (MM header) with the given IFD entries.
// Values are stored in big-endian byte order.
static std::vector<uint8_t> MakeBETiff(const std::vector<IfdEntry>& entries) {
  // Header (8) + IFD count (2) + entries (12 each) + next IFD ptr (4) + padding
  size_t size = 8 + 2 + entries.size() * 12 + 4 + 8;
  std::vector<uint8_t> buf(size, 0);

  // Big-endian header: "MM" + 42
  buf[0] = 0x4D; buf[1] = 0x4D; buf[2] = 0x00; buf[3] = 0x2A;
  // IFD offset = 8
  WriteBE32(&buf[4], 8);
  // Entry count
  WriteBE16(&buf[8], static_cast<uint16_t>(entries.size()));

  size_t offset = 10;
  for (const auto& e : entries) {
    WriteBE16(&buf[offset], e.tag);
    WriteBE16(&buf[offset + 2], e.type);
    WriteBE32(&buf[offset + 4], e.count);
    if (e.type == kTypeWord) {
      WriteBE16(&buf[offset + 8], static_cast<uint16_t>(e.value));
    } else {
      WriteBE32(&buf[offset + 8], e.value);
    }
    offset += 12;
  }
  return buf;
}

// Build a synthetic little-endian TIFF (II header) with the given IFD entries.
// Values are stored in little-endian byte order.
static std::vector<uint8_t> MakeLETiff(const std::vector<IfdEntry>& entries) {
  size_t size = 8 + 2 + entries.size() * 12 + 4 + 8;
  std::vector<uint8_t> buf(size, 0);

  // Little-endian header: "II" + 42
  buf[0] = 0x49; buf[1] = 0x49; buf[2] = 0x2A; buf[3] = 0x00;
  // IFD offset = 8
  WriteLE32(&buf[4], 8);
  // Entry count
  WriteLE16(&buf[8], static_cast<uint16_t>(entries.size()));

  size_t offset = 10;
  for (const auto& e : entries) {
    WriteLE16(&buf[offset], e.tag);
    WriteLE16(&buf[offset + 2], e.type);
    WriteLE32(&buf[offset + 4], e.count);
    if (e.type == kTypeWord) {
      WriteLE16(&buf[offset + 8], static_cast<uint16_t>(e.value));
    } else {
      WriteLE32(&buf[offset + 8], e.value);
    }
    offset += 12;
  }
  return buf;
}

// Helper: read a file into a byte vector
static std::vector<uint8_t> ReadFileBytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << "Cannot open: " << path;
  auto sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char *>(buf.data()), sz);
  return buf;
}

}  // namespace

// ============================================================================
// PeekShapeImpl with null buffer → DALI_ENFORCE throws (line 49)
// ============================================================================

TEST(TiffImageTest, PeekShapeNullBufferThrows) {
  TiffImage img(nullptr, 0, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// Constructor (line 45-46) + PeekShape with big-endian RGB TIFF
// Covers: is_little_endian returning true (line 40), TYPE_WORD (line 70-71),
// WIDTH_TAG (78-80), HEIGHT_TAG (81-83), SAMPLESPERPIXEL_TAG (84-88),
// final DALI_ENFORCE (line 98), return (line 101)
// ============================================================================

TEST(TiffImageTest, PeekShapeBigEndianRGB) {
  auto buf = MakeBETiff({
    {kWidthTag,           kTypeWord, 1, 320},
    {kHeightTag,          kTypeWord, 1, 240},
    {kSamplesPerPixelTag, kTypeWord, 1, 3},
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 240);  // height
  EXPECT_EQ(shape[1], 320);  // width
  EXPECT_EQ(shape[2], 3);    // channels
}

// ============================================================================
// PeekShape with little-endian grayscale TIFF
// Covers: is_little_endian returning false (line 36-37)
// ============================================================================

TEST(TiffImageTest, PeekShapeLittleEndianGrayscale) {
  auto buf = MakeLETiff({
    {kWidthTag,           kTypeWord, 1, 100},
    {kHeightTag,          kTypeWord, 1, 50},
    {kSamplesPerPixelTag, kTypeWord, 1, 1},
  });

  TiffImage img(buf.data(), buf.size(), DALI_GRAY);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 50);   // height
  EXPECT_EQ(shape[1], 100);  // width
  EXPECT_EQ(shape[2], 1);    // channel
}

// ============================================================================
// PeekShape with TYPE_DWORD values (line 72-73)
// ============================================================================

TEST(TiffImageTest, PeekShapeDWordValues) {
  auto buf = MakeBETiff({
    {kWidthTag,           kTypeDWord, 1, 640},
    {kHeightTag,          kTypeDWord, 1, 480},
    {kSamplesPerPixelTag, kTypeDWord, 1, 3},
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 480);
  EXPECT_EQ(shape[1], 640);
  EXPECT_EQ(shape[2], 3);
}

// ============================================================================
// PeekShape with palette image → nchannels=3 (lines 89-92)
// Also covers early break (lines 94-95) and palette_read path
// ============================================================================

TEST(TiffImageTest, PeekShapePalette) {
  auto buf = MakeBETiff({
    {kWidthTag,       kTypeWord, 1, 200},
    {kHeightTag,      kTypeWord, 1, 150},
    {kPhotometricTag, kTypeWord, 1, kPhotometricPalette},
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 150);
  EXPECT_EQ(shape[1], 200);
  EXPECT_EQ(shape[2], 3);  // palette → 3 channels
}

// ============================================================================
// PeekShape where PHOTOMETRIC_PALETTE comes after SAMPLESPERPIXEL
// → covers the !palette_read guard being true, then palette overriding (line 84)
// ============================================================================

TEST(TiffImageTest, PeekShapePaletteAfterSamplesPerPixel) {
  auto buf = MakeBETiff({
    {kWidthTag,           kTypeWord, 1, 200},
    {kHeightTag,          kTypeWord, 1, 150},
    {kSamplesPerPixelTag, kTypeWord, 1, 1},   // palette TIFF has spp=1
    {kPhotometricTag,     kTypeWord, 1, kPhotometricPalette},
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 150);
  EXPECT_EQ(shape[1], 200);
  EXPECT_EQ(shape[2], 3);  // palette overrides spp=1 → 3
}

// ============================================================================
// PeekShape where SAMPLESPERPIXEL comes after PHOTOMETRIC_PALETTE
// → covers the !palette_read being false → SAMPLESPERPIXEL is skipped (line 84)
// ============================================================================

TEST(TiffImageTest, PeekShapeSamplesPerPixelAfterPalette) {
  auto buf = MakeBETiff({
    {kWidthTag,           kTypeWord, 1, 200},
    {kHeightTag,          kTypeWord, 1, 150},
    {kPhotometricTag,     kTypeWord, 1, kPhotometricPalette},
    {kSamplesPerPixelTag, kTypeWord, 1, 1},   // should be ignored
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 150);
  EXPECT_EQ(shape[1], 200);
  EXPECT_EQ(shape[2], 3);  // palette wins, spp=1 is skipped
}

// ============================================================================
// PeekShape with unsupported value type → DALI_FAIL (line 74-75)
// ============================================================================

TEST(TiffImageTest, PeekShapeUnsupportedValueTypeThrows) {
  // Use value_type=5 (RATIONAL), which is neither TYPE_WORD nor TYPE_DWORD
  auto buf = MakeBETiff({
    {kWidthTag, 5, 1, 320},  // type=5 is unsupported
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with value_count != 1 → DALI_ENFORCE (line 67)
// ============================================================================

TEST(TiffImageTest, PeekShapeValueCountNotOneThrows) {
  auto buf = MakeBETiff({
    {kWidthTag, kTypeWord, 2, 320},  // count=2 is invalid
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with missing dimensions → DALI_ENFORCE (line 98-99)
// Only width present, missing height and samples_per_pixel
// ============================================================================

TEST(TiffImageTest, PeekShapeMissingDimensionsThrows) {
  // A valid TIFF header but only width tag — missing height and spp
  auto buf = MakeBETiff({
    {kWidthTag, kTypeWord, 1, 320},
  });

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShape with real TIFF file → full end-to-end sanity
// ============================================================================

TEST(TiffImageTest, PeekShapeRealFile) {
  std::string path = testing::dali_extra_path() +
                     "/db/single/tiff/0/cat-111793_640.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF file not found: " << path;

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
  EXPECT_GT(shape[2], 0);
}

// ============================================================================
// PeekShape with real palette TIFF file
// ============================================================================

TEST(TiffImageTest, PeekShapeRealPaletteFile) {
  std::string path = testing::dali_extra_path() +
                     "/db/single/tiff/0/cat-300572_640_palette.tiff";
  auto buf = ReadFileBytes(path);
  ASSERT_FALSE(buf.empty()) << "Test TIFF file not found: " << path;

  TiffImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_GT(shape[0], 0);
  EXPECT_GT(shape[1], 0);
  EXPECT_EQ(shape[2], 3);  // palette → 3 channels
}

}  // namespace dali
