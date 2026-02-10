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
#include <setjmp.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "dali/operators/decoder/jpeg/jpeg_handle.h"

namespace dali {
namespace testing {

// ============================================================================
// SetDest (3-arg overload) → covers lines 122-124
// ============================================================================

TEST(JpegHandleTest, SetDestWithoutDestination) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  std::vector<uint8_t> buffer(1024);
  jpeg::SetDest(&cinfo, buffer.data(), buffer.size());

  EXPECT_NE(cinfo.dest, nullptr);
  jpeg_destroy_compress(&cinfo);
}

// ============================================================================
// SetDest (4-arg overload) with string destination → covers lines 127-143
// ============================================================================

TEST(JpegHandleTest, SetDestWithDestination) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  std::string dest_str;
  std::vector<uint8_t> buffer(1024);
  jpeg::SetDest(&cinfo, buffer.data(), buffer.size(), &dest_str);

  EXPECT_NE(cinfo.dest, nullptr);
  jpeg_destroy_compress(&cinfo);
}

// ============================================================================
// SetDest called twice → covers cinfo->dest != nullptr path (line 130)
// ============================================================================

TEST(JpegHandleTest, SetDestCalledTwice) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  std::string dest_str;
  std::vector<uint8_t> buffer(1024);
  jpeg::SetDest(&cinfo, buffer.data(), buffer.size(), &dest_str);
  auto first_dest = cinfo.dest;
  // Call again — dest is already set, so the allocation branch is skipped
  jpeg::SetDest(&cinfo, buffer.data(), buffer.size(), &dest_str);
  EXPECT_EQ(cinfo.dest, first_dest);

  jpeg_destroy_compress(&cinfo);
}

// ============================================================================
// Full compress with string destination → covers MemInitDestination (lines 86-95),
// MemEmptyOutputBuffer (lines 98-107), and MemTermDestination (lines 110-119)
// including the dest->dest branch (lines 92-93, 101-102, 113-115)
// ============================================================================

TEST(JpegHandleTest, CompressWithStringDestination) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  std::string dest_str;
  // Use a tiny buffer so MemEmptyOutputBuffer gets called multiple times
  std::vector<uint8_t> buffer(64);
  jpeg::SetDest(&cinfo, buffer.data(), buffer.size(), &dest_str);

  cinfo.image_width = 8;
  cinfo.image_height = 8;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);  // calls MemInitDestination

  std::vector<uint8_t> row(8 * 3, 128);
  JSAMPROW row_ptr = row.data();
  while (cinfo.next_scanline < cinfo.image_height) {
    jpeg_write_scanlines(&cinfo, &row_ptr, 1);  // may call MemEmptyOutputBuffer
  }
  jpeg_finish_compress(&cinfo);  // calls MemTermDestination

  EXPECT_GT(dest_str.size(), 0u);
  jpeg_destroy_compress(&cinfo);
}

// ============================================================================
// Full compress with null destination → covers MemInitDestination with
// dest->dest == nullptr (line 92 false branch), MemTermDestination with
// dest->dest == nullptr (line 113 false branch)
// ============================================================================

TEST(JpegHandleTest, CompressWithNullDestination) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  std::vector<uint8_t> buffer(4096);
  jpeg::SetDest(&cinfo, buffer.data(), buffer.size());  // nullptr dest

  cinfo.image_width = 4;
  cinfo.image_height = 4;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);

  std::vector<uint8_t> row(4 * 3, 200);
  JSAMPROW row_ptr = row.data();
  while (cinfo.next_scanline < cinfo.image_height) {
    jpeg_write_scanlines(&cinfo, &row_ptr, 1);
  }
  jpeg_finish_compress(&cinfo);

  jpeg_destroy_compress(&cinfo);
}

// ============================================================================
// CatchError → covers lines 31-36
// Uses setjmp/longjmp since CatchError calls longjmp
// ============================================================================

TEST(JpegHandleTest, CatchErrorJumps) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  jmp_buf jmpbuf;
  cinfo.client_data = &jmpbuf;
  cinfo.err->error_exit = jpeg::CatchError;

  bool caught = false;
  if (setjmp(jmpbuf) == 0) {
    // Trigger an error by calling error_exit directly
    cinfo.err->error_exit(reinterpret_cast<j_common_ptr>(&cinfo));
  } else {
    caught = true;
  }
  EXPECT_TRUE(caught);
  // CatchError calls jpeg_destroy, so no cleanup needed
}

// ============================================================================
// MemSkipInputData with negative jump → covers line 191-192
// ============================================================================

TEST(JpegHandleTest, SkipInputDataNegativeJump) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[] = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10,
                    0x4A, 0x46, 0x49, 0x46, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0xFF, 0xD9};
  jpeg::SetSrc(&cinfo, data, sizeof(data), false);

  // Init the source to set up next_input_byte and bytes_in_buffer
  cinfo.src->init_source(&cinfo);
  auto orig_next = cinfo.src->next_input_byte;
  auto orig_bytes = cinfo.src->bytes_in_buffer;

  // Negative jump → should return without changing state
  cinfo.src->skip_input_data(&cinfo, -5);

  EXPECT_EQ(cinfo.src->next_input_byte, orig_next);
  EXPECT_EQ(cinfo.src->bytes_in_buffer, orig_bytes);

  jpeg_destroy_decompress(&cinfo);
}

// ============================================================================
// MemSkipInputData with jump > bytes_in_buffer → covers lines 194-196
// This calls MemFillInputBuffer internally. Since try_recover is true
// and next_input_byte != kEOIBuffer, it inserts a fake EOI (lines 170-176).
// ============================================================================

TEST(JpegHandleTest, SkipInputDataOverflow) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[] = {0xFF, 0xD8, 0xFF, 0xD9};
  jpeg::SetSrc(&cinfo, data, sizeof(data), true);  // try_recover = true

  // Init the source
  cinfo.src->init_source(&cinfo);
  // bytes_in_buffer = 4, next_input_byte = data after init

  // Simulate having consumed some data so next_input_byte != data
  // (avoids the empty-file ERREXIT path in MemFillInputBuffer)
  cinfo.src->next_input_byte = data + 2;
  cinfo.src->bytes_in_buffer = 2;

  // Jump beyond available bytes → calls MemFillInputBuffer
  // With try_recover_truncated_jpeg = true and next_input_byte != kEOIBuffer,
  // it should insert a fake EOI
  cinfo.src->skip_input_data(&cinfo, 1000);

  // After overflow + fill, bytes_in_buffer should be 2 (fake EOI)
  EXPECT_EQ(cinfo.src->bytes_in_buffer, 2u);

  jpeg_destroy_decompress(&cinfo);
}

// ============================================================================
// MemFillInputBuffer with bytes_in_buffer > 0, try_recover = false
// → covers line 167, returns FALSE (line 169)
// ============================================================================

TEST(JpegHandleTest, FillInputBufferWithDataLeftNoRecover) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[] = {0xFF, 0xD8, 0xFF, 0xD9};
  jpeg::SetSrc(&cinfo, data, sizeof(data), false);  // try_recover = false

  cinfo.src->init_source(&cinfo);
  // bytes_in_buffer = 4 > 0 → takes the "else if (bytes_in_buffer)" branch
  auto result = cinfo.src->fill_input_buffer(&cinfo);
  EXPECT_EQ(result, FALSE);

  jpeg_destroy_decompress(&cinfo);
}

// ============================================================================
// MemFillInputBuffer with bytes_in_buffer > 0, try_recover = true
// → covers line 167, returns TRUE (line 169)
// ============================================================================

TEST(JpegHandleTest, FillInputBufferWithDataLeftRecover) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[] = {0xFF, 0xD8, 0xFF, 0xD9};
  jpeg::SetSrc(&cinfo, data, sizeof(data), true);  // try_recover = true

  cinfo.src->init_source(&cinfo);
  // bytes_in_buffer = 4 > 0 → returns TRUE for recovery
  auto result = cinfo.src->fill_input_buffer(&cinfo);
  EXPECT_EQ(result, TRUE);

  jpeg_destroy_decompress(&cinfo);
}

// ============================================================================
// MemFillInputBuffer — empty file case (bytes_in_buffer == 0 &&
// next_input_byte == data) → covers lines 163-166, calls ERREXIT
// ============================================================================

TEST(JpegHandleTest, FillInputBufferEmptyFile) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[1] = {0};
  jpeg::SetSrc(&cinfo, data, 0, false);  // datasize = 0

  // Manually init: sets next_input_byte = data, bytes_in_buffer = 0
  cinfo.src->init_source(&cinfo);

  jmp_buf jmpbuf;
  cinfo.client_data = &jmpbuf;
  cinfo.err->error_exit = jpeg::CatchError;

  bool caught = false;
  if (setjmp(jmpbuf) == 0) {
    cinfo.src->fill_input_buffer(&cinfo);
  } else {
    caught = true;
  }
  EXPECT_TRUE(caught);
  // CatchError calls jpeg_destroy, no cleanup needed
}

// ============================================================================
// MemFillInputBuffer — truncated data with recovery (bytes_in_buffer == 0,
// next_input_byte != data, next_input_byte != kEOIBuffer, try_recover = true)
// → covers lines 170-176 (insert fake EOI)
// ============================================================================

TEST(JpegHandleTest, FillInputBufferTruncatedWithRecovery) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[] = {0xFF, 0xD8, 0xFF, 0xE0};
  jpeg::SetSrc(&cinfo, data, sizeof(data), true);  // try_recover = true

  cinfo.src->init_source(&cinfo);
  // Simulate having consumed all data: advance pointer past data
  cinfo.src->next_input_byte = data + sizeof(data);
  cinfo.src->bytes_in_buffer = 0;

  // This is the truncated case: bytes_in_buffer == 0,
  // next_input_byte != data (not empty file),
  // next_input_byte != kEOIBuffer, try_recover = true
  auto result = cinfo.src->fill_input_buffer(&cinfo);
  EXPECT_EQ(result, TRUE);
  EXPECT_EQ(cinfo.src->bytes_in_buffer, 2u);  // fake EOI inserted

  jpeg_destroy_decompress(&cinfo);
}

// ============================================================================
// MemFillInputBuffer — final error case (already at fake EOI, bytes_in_buffer == 0)
// → covers lines 177-181, calls ERREXIT
// ============================================================================

TEST(JpegHandleTest, FillInputBufferFinalError) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  uint8_t data[] = {0xFF, 0xD8, 0xFF, 0xE0};
  jpeg::SetSrc(&cinfo, data, sizeof(data), true);  // try_recover = true

  cinfo.src->init_source(&cinfo);

  // First: consume all data and get a fake EOI inserted
  cinfo.src->next_input_byte = data + sizeof(data);
  cinfo.src->bytes_in_buffer = 0;
  auto result = cinfo.src->fill_input_buffer(&cinfo);
  EXPECT_EQ(result, TRUE);

  // Now consume the fake EOI
  cinfo.src->bytes_in_buffer = 0;
  // next_input_byte is now pointing to kEOIBuffer (static) → matches the else branch

  jmp_buf jmpbuf;
  cinfo.client_data = &jmpbuf;
  cinfo.err->error_exit = jpeg::CatchError;

  bool caught = false;
  if (setjmp(jmpbuf) == 0) {
    cinfo.src->fill_input_buffer(&cinfo);
  } else {
    caught = true;
  }
  EXPECT_TRUE(caught);
  // CatchError calls jpeg_destroy, no cleanup needed
}

}  // namespace testing
}  // namespace dali
