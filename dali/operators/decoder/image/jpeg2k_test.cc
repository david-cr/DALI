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
#include "dali/operators/decoder/image/jpeg2k.h"
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

// Helper: write a big-endian uint16 at dst
static void WriteBE16(uint8_t *dst, uint16_t val) {
  dst[0] = static_cast<uint8_t>((val >> 8) & 0xFF);
  dst[1] = static_cast<uint8_t>(val & 0xFF);
}

// Build a valid minimal JP2 header buffer.
// Structure:
//   [0..11]   Signature box:  size=12, type='jP  ', content=0x0D0A870A
//   [12..31]  File Type box:  size=20, type='ftyp', content='jp2 '+0+compat
//   [32..39]  JP2 Header box header: size=30, type='jp2h'
//   [40..61]  Image Header sub-box:  size=22, type='ihdr',
//             height(4), width(4), channels(2), bpc(1), comp(1), cs(1), ip(1)
// Total: 62 bytes
static std::vector<uint8_t> MakeValidJP2Header(uint32_t height, uint32_t width,
                                                uint16_t channels) {
  std::vector<uint8_t> buf(62, 0);

  // Signature box (12 bytes at offset 0)
  WriteBE32(&buf[0], 12);       // box size
  buf[4] = 'j'; buf[5] = 'P'; buf[6] = ' '; buf[7] = ' ';  // box type
  buf[8] = 0x0D; buf[9] = 0x0A; buf[10] = 0x87; buf[11] = 0x0A;  // JP2 signature

  // File Type box (20 bytes at offset 12)
  WriteBE32(&buf[12], 20);      // box size
  buf[16] = 'f'; buf[17] = 't'; buf[18] = 'y'; buf[19] = 'p';  // box type
  buf[20] = 'j'; buf[21] = 'p'; buf[22] = '2'; buf[23] = ' ';  // brand
  WriteBE32(&buf[24], 0);       // minor version
  buf[28] = 'j'; buf[29] = 'p'; buf[30] = '2'; buf[31] = ' ';  // compatibility

  // JP2 Header super-box (header at offset 32, size = 8 + 22 = 30)
  WriteBE32(&buf[32], 30);      // box size
  buf[36] = 'j'; buf[37] = 'p'; buf[38] = '2'; buf[39] = 'h';  // box type

  // Image Header sub-box (22 bytes at offset 40)
  WriteBE32(&buf[40], 22);      // box size
  buf[44] = 'i'; buf[45] = 'h'; buf[46] = 'd'; buf[47] = 'r';  // box type
  WriteBE32(&buf[48], height);  // height
  WriteBE32(&buf[52], width);   // width
  WriteBE16(&buf[56], channels);// num components
  buf[58] = 7;                  // bpc (8 bits, unsigned)
  buf[59] = 7;                  // compression type (jpeg2000)
  buf[60] = 0;                  // colorspace unknown
  buf[61] = 0;                  // intellectual property

  return buf;
}

// ============================================================================
// CheckIsJPEG2k with too-small size → return false (line 56-57)
// ============================================================================

TEST(Jpeg2kImageTest, CheckIsJPEG2kTooSmallReturnsFalse) {
  uint8_t data[4] = {0};
  // size < kBlockHdrSize (8) → should return false
  EXPECT_FALSE(CheckIsJPEG2k(data, 4));
  EXPECT_FALSE(CheckIsJPEG2k(data, 0));
  EXPECT_FALSE(CheckIsJPEG2k(data, 7));
}

TEST(Jpeg2kImageTest, CheckIsJPEG2kValidSignature) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  EXPECT_TRUE(CheckIsJPEG2k(buf.data(), buf.size()));
}

TEST(Jpeg2kImageTest, CheckIsJPEG2kWrongSignature) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Corrupt signature type: change 'jP  ' to 'xP  '
  buf[4] = 'x';
  EXPECT_FALSE(CheckIsJPEG2k(buf.data(), buf.size()));
}

// ============================================================================
// PeekShapeImpl on valid JP2 → sanity test
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeValidJP2) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  Jpeg2kImage img(buf.data(), buf.size(), DALI_RGB);
  auto shape = img.PeekShape();
  EXPECT_EQ(shape[0], 100);  // height
  EXPECT_EQ(shape[1], 200);  // width
  EXPECT_EQ(shape[2], 3);    // channels
}

// ============================================================================
// advance_one_block failure: wrong block type at sig box (line 46)
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeWrongSigBlockTypeThrows) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Corrupt the signature box type: 'jP  ' → 'XX  '
  buf[4] = 'X'; buf[5] = 'X';
  Jpeg2kImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// advance_one_block failure: wrong block type at ftyp box (line 46)
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeWrongFtypBlockTypeThrows) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Corrupt the file type box type: 'ftyp' → 'ZZZZ'
  buf[16] = 'Z'; buf[17] = 'Z'; buf[18] = 'Z'; buf[19] = 'Z';
  Jpeg2kImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// advance_one_block failure: block size causes index to exceed data (line 49)
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeBlockSizeExceedsDataThrows) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Make the signature box size very large (bigger than buffer)
  WriteBE32(&buf[0], 999);  // sig box size = 999, way beyond 62-byte buffer
  Jpeg2kImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShapeImpl: wrong jp2_header_type (line 68)
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeWrongJP2HeaderTypeThrows) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Corrupt the jp2h type at offset 36-39: 'jp2h' → 'xxxx'
  buf[36] = 'x'; buf[37] = 'x'; buf[38] = 'x'; buf[39] = 'x';
  Jpeg2kImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShapeImpl: wrong jp2_im_header_type (line 70)
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeWrongImHeaderTypeThrows) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Corrupt the ihdr type at offset 44-47: 'ihdr' → 'bad!'
  buf[44] = 'b'; buf[45] = 'a'; buf[46] = 'd'; buf[47] = '!';
  Jpeg2kImage img(buf.data(), buf.size(), DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

// ============================================================================
// PeekShapeImpl: data too short to read height/width/channels (line 71)
// ============================================================================

TEST(Jpeg2kImageTest, PeekShapeTruncatedImHeaderThrows) {
  auto buf = MakeValidJP2Header(100, 200, 3);
  // Truncate right after the ihdr box header (offset 48),
  // so there's no room for height+width+channels
  size_t truncated_size = 50;  // need at least 48 + 8 + 4 + 4 + 2 = 66
  Jpeg2kImage img(buf.data(), truncated_size, DALI_RGB);
  EXPECT_THROW(img.PeekShape(), DALIException);
}

}  // namespace testing
}  // namespace dali
