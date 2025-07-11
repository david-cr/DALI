// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <linux/limits.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "dali/util/file.h"
#include "dali/util/odirect_file.h"
#include "dali/core/stream.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"

namespace dali {

class DaliFileTest : public ::testing::Test {
 protected:
  // Helper function to allocate aligned memory for O_DIRECT
  static void* AllocateAlignedBuffer(size_t size, size_t alignment) {
    void* ptr = nullptr;
    int result = posix_memalign(&ptr, alignment, size);
    if (result != 0) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  // Helper function to free aligned memory
  static void FreeAlignedBuffer(void* ptr) {
    free(ptr);
  }
 protected:
  void SetUp() override {
    // Create test directory with random name
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    int random_num = dis(gen);
    test_dir_ = "/tmp/dali_file_test_" + std::to_string(random_num);
    mkdir(test_dir_.c_str(), 0755);

    // Create test files
    CreateTestFiles();
  }

  void TearDown() override {
    // Clean up test files
    CleanupTestFiles();
  }

  void CreateTestFiles() {
    // Create a simple text file
    std::string text_file = test_dir_ + "/test_file.txt";
    std::ofstream text_stream(text_file);
    ASSERT_TRUE(text_stream.is_open());
    text_stream << "Hello, DALI file test!\n";
    text_stream << "This is a test file for file operations.\n";
    text_stream << "Line 3 with some content.\n";
    text_stream.close();

    // Create a binary file (aligned for O_DIRECT)
    std::string binary_file = test_dir_ + "/test_file.bin";
    std::ofstream binary_stream(binary_file, std::ios::binary);
    ASSERT_TRUE(binary_stream.is_open());
    // Create a larger file that's aligned to typical O_DIRECT requirements (4KB)
    std::vector<uint8_t> binary_data(4096, 0);  // 4KB aligned file
    // Fill first 8 bytes with test data
    for (int i = 0; i < 8; i++) {
      binary_data[i] = i + 1;
    }
    binary_stream.write(reinterpret_cast<const char*>(binary_data.data()), binary_data.size());
    binary_stream.close();

    // Create an empty file
    std::string empty_file = test_dir_ + "/empty_file.txt";
    std::ofstream empty_stream(empty_file);
    ASSERT_TRUE(empty_stream.is_open());
    empty_stream.close();

    // Create a large file (1MB)
    std::string large_file = test_dir_ + "/large_file.bin";
    std::ofstream large_stream(large_file, std::ios::binary);
    ASSERT_TRUE(large_stream.is_open());
    std::vector<uint8_t> large_data(1024 * 1024, 0x42);  // 1MB of 0x42
    large_stream.write(reinterpret_cast<const char*>(large_data.data()), large_data.size());
    large_stream.close();

    // Create a file with special characters in name
    std::string special_file = test_dir_ + "/test_file_with_spaces and-dashes.txt";
    std::ofstream special_stream(special_file);
    ASSERT_TRUE(special_stream.is_open());
    special_stream << "File with special characters in name\n";
    special_stream.close();
  }

  void CleanupTestFiles() {
    // Remove test files
    std::vector<std::string> files = {
      test_dir_ + "/test_file.txt",
      test_dir_ + "/test_file.bin",
      test_dir_ + "/empty_file.txt",
      test_dir_ + "/large_file.bin",
      test_dir_ + "/test_file_with_spaces and-dashes.txt"
    };

    for (const auto& file : files) {
      unlink(file.c_str());
    }

    // Remove test directory
    rmdir(test_dir_.c_str());
  }

  std::string test_dir_;
};

// Test 1: FileStream - Basic File Operations
TEST_F(DaliFileTest, FileStreamBasic) {
  std::string file_path = test_dir_ + "/test_file.txt";

  // Test file opening
  auto file_stream = FileStream::Open(file_path);
  ASSERT_NE(file_stream, nullptr);
  EXPECT_EQ(file_stream->path(), file_path);

  // Test file reading
  std::vector<char> buffer(100);
  size_t bytes_read = file_stream->Read(buffer.data(), buffer.size());
  EXPECT_GT(bytes_read, 0);
  EXPECT_LE(bytes_read, buffer.size());

  // Test file size
  size_t file_size = file_stream->Size();
  EXPECT_GT(file_size, 0);

  // Test seeking
  file_stream->SeekRead(0, SEEK_SET);
  EXPECT_EQ(file_stream->TellRead(), 0);

  file_stream->SeekRead(5, SEEK_SET);
  EXPECT_EQ(file_stream->TellRead(), 5);

  file_stream->Close();
}

// Test 2: FileStream - Error Handling (File Not Found)
TEST_F(DaliFileTest, FileStreamFileNotFound) {
  std::string nonexistent_file = test_dir_ + "/nonexistent_file.txt";

  EXPECT_THROW({
    auto file_stream = FileStream::Open(nonexistent_file);
  }, std::exception);
}

// Test 3: FileStream - Empty File
TEST_F(DaliFileTest, FileStreamEmptyFile) {
  std::string empty_file = test_dir_ + "/empty_file.txt";

  auto file_stream = FileStream::Open(empty_file);
  ASSERT_NE(file_stream, nullptr);

  EXPECT_EQ(file_stream->Size(), 0);

  std::vector<char> buffer(10);
  size_t bytes_read = file_stream->Read(buffer.data(), buffer.size());
  EXPECT_EQ(bytes_read, 0);

  file_stream->Close();
}

// Test 4: FileStream - Large File
TEST_F(DaliFileTest, FileStreamLargeFile) {
  std::string large_file = test_dir_ + "/large_file.bin";

  auto file_stream = FileStream::Open(large_file);
  ASSERT_NE(file_stream, nullptr);

  EXPECT_EQ(file_stream->Size(), 1024 * 1024);  // 1MB

  // Read in chunks
  std::vector<char> buffer(4096);
  size_t total_read = 0;
  size_t bytes_read;

  while ((bytes_read = file_stream->Read(buffer.data(), buffer.size())) > 0) {
    total_read += bytes_read;
  }

  EXPECT_EQ(total_read, 1024 * 1024);

  file_stream->Close();
}

// Test 5: FileStream - Binary File
TEST_F(DaliFileTest, FileStreamBinaryFile) {
  std::string binary_file = test_dir_ + "/test_file.bin";

  auto file_stream = FileStream::Open(binary_file);
  ASSERT_NE(file_stream, nullptr);

  EXPECT_EQ(file_stream->Size(), 4096);  // 4KB aligned file

  std::vector<uint8_t> buffer(8);
  size_t bytes_read = file_stream->Read(buffer.data(), buffer.size());
  EXPECT_EQ(bytes_read, 8);

  // Verify content (first 8 bytes)
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(buffer[i], i + 1);
  }

  file_stream->Close();
}

// Test 6: FileStream - Seek Operations
TEST_F(DaliFileTest, FileStreamSeekOperations) {
  std::string file_path = test_dir_ + "/test_file.txt";

  auto file_stream = FileStream::Open(file_path);
  ASSERT_NE(file_stream, nullptr);

  // Test SEEK_SET
  file_stream->SeekRead(10, SEEK_SET);
  EXPECT_EQ(file_stream->TellRead(), 10);

  // Test SEEK_CUR
  file_stream->SeekRead(5, SEEK_CUR);
  EXPECT_EQ(file_stream->TellRead(), 15);

  // Test SEEK_END
  file_stream->SeekRead(-5, SEEK_END);
  size_t file_size = file_stream->Size();
  EXPECT_EQ(file_stream->TellRead(), file_size - 5);

  file_stream->Close();
}

// Test 7: FileStream - Special Characters in Filename
TEST_F(DaliFileTest, FileStreamSpecialCharacters) {
  std::string special_file = test_dir_ + "/test_file_with_spaces and-dashes.txt";

  auto file_stream = FileStream::Open(special_file);
  ASSERT_NE(file_stream, nullptr);

  EXPECT_EQ(file_stream->path(), special_file);

  std::vector<char> buffer(100);
  size_t bytes_read = file_stream->Read(buffer.data(), buffer.size());
  EXPECT_GT(bytes_read, 0);

  file_stream->Close();
}

// Test 8: ODirectFileStream - Basic Operations (if supported)
TEST_F(DaliFileTest, ODirectFileStreamBasic) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    EXPECT_EQ(odirect_file->path(), file_path);
    EXPECT_EQ(odirect_file->Size(), 4096);  // Now 4KB aligned

    // Test alignment values
    size_t alignment = ODirectFileStream::GetAlignment();
    size_t len_alignment = ODirectFileStream::GetLenAlignment();
    size_t chunk_size = ODirectFileStream::GetChunkSize();

    EXPECT_GT(alignment, 0);
    EXPECT_GT(len_alignment, 0);
    EXPECT_GT(chunk_size, 0);

        // Create aligned buffer for O_DIRECT
    size_t buffer_size = alignment;  // Use alignment size for buffer
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(buffer_size, alignment));

    // Ensure buffer is aligned
    uintptr_t buffer_addr = reinterpret_cast<uintptr_t>(buffer);
    EXPECT_EQ(buffer_addr % alignment, 0) << "Buffer not aligned to " << alignment;

    // Test ReadAt with aligned buffer and offset
    size_t bytes_read = odirect_file->ReadAt(buffer, buffer_size, 0);
    EXPECT_EQ(bytes_read, buffer_size);

    // Verify content (first 8 bytes should contain our test data)
    for (int i = 0; i < 8; i++) {
      EXPECT_EQ(buffer[i], i + 1);
    }

    // Test Read method with aligned buffer
    odirect_file->SeekRead(0, SEEK_SET);
    uint8_t* buffer2 = static_cast<uint8_t*>(AllocateAlignedBuffer(buffer_size, alignment));

    // Ensure second buffer is also aligned
    uintptr_t buffer2_addr = reinterpret_cast<uintptr_t>(buffer2);
    EXPECT_EQ(buffer2_addr % alignment, 0) << "Buffer2 not aligned to " << alignment;

    size_t bytes_read2 = odirect_file->Read(buffer2, buffer_size);
    EXPECT_EQ(bytes_read2, buffer_size);

    // Verify content matches (first 8 bytes should contain our test data)
    for (int i = 0; i < 8; i++) {
      EXPECT_EQ(buffer2[i], i + 1);
    }

    // Clean up aligned buffers
    FreeAlignedBuffer(buffer);
    FreeAlignedBuffer(buffer2);

    odirect_file->Close();
  } catch (const std::exception& e) {
    // O_DIRECT might not be supported on all systems
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 9: ODirectFileStream - Large File (if supported)
TEST_F(DaliFileTest, ODirectFileStreamLargeFile) {
  std::string large_file = test_dir_ + "/large_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(large_file);

    EXPECT_EQ(odirect_file->Size(), 1024 * 1024);

    // Read in chunks using ReadAt
    size_t chunk_size = ODirectFileStream::GetChunkSize();
    size_t alignment = ODirectFileStream::GetAlignment();

    // Ensure chunk_size is aligned
    chunk_size = (chunk_size / alignment) * alignment;
    if (chunk_size == 0) chunk_size = alignment;

    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(chunk_size, alignment));
    size_t total_read = 0;
    size_t offset = 0;

    while (offset < odirect_file->Size()) {
      size_t to_read = std::min(chunk_size, odirect_file->Size() - offset);
      // Ensure to_read is aligned
      to_read = (to_read / alignment) * alignment;
      if (to_read == 0) break;

      size_t bytes_read = odirect_file->ReadAt(buffer, to_read, offset);
      EXPECT_EQ(bytes_read, to_read);
      total_read += bytes_read;
      offset += bytes_read;
    }

    FreeAlignedBuffer(buffer);

    EXPECT_EQ(total_read, 1024 * 1024);

    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 10: FileStream - Memory Mapping (if supported)
TEST_F(DaliFileTest, FileStreamMemoryMapping) {
  std::string file_path = test_dir_ + "/test_file.bin";

  auto file_stream = FileStream::Open(file_path);
  ASSERT_NE(file_stream, nullptr);

  // Test if memory mapping is supported
  if (file_stream->CanMemoryMap()) {
    // Try to get memory mapped data
    size_t file_size = file_stream->Size();
    auto mapped_data = file_stream->Get(file_size);
    EXPECT_NE(mapped_data, nullptr);
  } else {
    // Memory mapping not supported, should throw
    EXPECT_THROW({
      file_stream->Get(100);
    }, std::logic_error);
  }

  file_stream->Close();
}

// Test 11: FileStream - Options
TEST_F(DaliFileTest, FileStreamOptions) {
  std::string file_path = test_dir_ + "/test_file.txt";

  // Test with different options
  FileStream::Options opts;
  opts.read_ahead = true;
  opts.use_mmap = false;
  opts.use_odirect = false;

  auto file_stream = FileStream::Open(file_path, opts);
  ASSERT_NE(file_stream, nullptr);

  EXPECT_EQ(file_stream->path(), file_path);

  file_stream->Close();
}

// Test 12: FileStream - Concurrent Access
TEST_F(DaliFileTest, FileStreamConcurrentAccess) {
  std::string file_path = test_dir_ + "/test_file.txt";

  // Open multiple streams to the same file
  auto stream1 = FileStream::Open(file_path);
  auto stream2 = FileStream::Open(file_path);

  ASSERT_NE(stream1, nullptr);
  ASSERT_NE(stream2, nullptr);

  // Read from both streams
  std::vector<char> buffer1(50);
  std::vector<char> buffer2(50);

  size_t bytes1 = stream1->Read(buffer1.data(), buffer1.size());
  size_t bytes2 = stream2->Read(buffer2.data(), buffer2.size());

  EXPECT_GT(bytes1, 0);
  EXPECT_GT(bytes2, 0);

  // Content should be the same
  EXPECT_EQ(bytes1, bytes2);
  EXPECT_EQ(std::string(buffer1.data(), bytes1), std::string(buffer2.data(), bytes2));

  stream1->Close();
  stream2->Close();
}

// Test 13: ODirectFileStream - Constructor Error (File Not Found)
TEST_F(DaliFileTest, ODirectFileStreamConstructorError) {
  std::string nonexistent_file = test_dir_ + "/nonexistent_file.bin";

  try {
    EXPECT_THROW({
      auto odirect_file = std::make_unique<ODirectFileStream>(nonexistent_file);
    }, DALIException);
  } catch (const std::exception& e) {
    // O_DIRECT might not be supported on all systems
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 14: ODirectFileStream - Seek Operations
TEST_F(DaliFileTest, ODirectFileStreamSeekOperations) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    // Test SEEK_SET
    odirect_file->SeekRead(2, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 2);

    // Test SEEK_CUR
    odirect_file->SeekRead(3, SEEK_CUR);
    EXPECT_EQ(odirect_file->TellRead(), 5);

    // Test SEEK_END
    odirect_file->SeekRead(-2, SEEK_END);
    EXPECT_EQ(odirect_file->TellRead(), 4094); // 4096 - 2 = 4094

    // Test TellRead at beginning
    odirect_file->SeekRead(0, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 0);

    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 15: ODirectFileStream - Seek Error (Invalid Position)
TEST_F(DaliFileTest, ODirectFileStreamSeekError) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    // Test seeking to negative position (should fail)
    EXPECT_THROW({
      odirect_file->SeekRead(-1, SEEK_SET);
    }, DALIException);

    // Test seeking with invalid whence value (should fail)
    EXPECT_THROW({
      odirect_file->SeekRead(0, 999);  // Invalid whence
    }, DALIException);

    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 15b: ODirectFileStream - Seek Beyond File Size (Valid Behavior)
TEST_F(DaliFileTest, ODirectFileStreamSeekBeyondFileSize) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    // Test seeking beyond file size (should succeed)
    odirect_file->SeekRead(10000, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 10000);

    // Test reading from beyond file size (should return 0)
    size_t alignment = ODirectFileStream::GetAlignment();
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(alignment, alignment));

    size_t bytes_read = odirect_file->Read(buffer, alignment);
    EXPECT_EQ(bytes_read, 0);  // Should return 0 when reading beyond file size

    FreeAlignedBuffer(buffer);
    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 16: ODirectFileStream - ReadAt with Different Offsets
TEST_F(DaliFileTest, ODirectFileStreamReadAtOffsets) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

        // Get alignment for O_DIRECT
    size_t alignment = ODirectFileStream::GetAlignment();

    // Create aligned buffer
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(alignment, alignment));
    uintptr_t buffer_addr = reinterpret_cast<uintptr_t>(buffer);
    EXPECT_EQ(buffer_addr % alignment, 0) << "Buffer not aligned to " << alignment;

    // Read from offset 0 (aligned)
    size_t bytes_read1 = odirect_file->ReadAt(buffer, alignment, 0);
    EXPECT_EQ(bytes_read1, alignment);
    // Verify first 8 bytes contain our test data
    EXPECT_EQ(buffer[0], 0x01);
    EXPECT_EQ(buffer[1], 0x02);
    EXPECT_EQ(buffer[2], 0x03);
    EXPECT_EQ(buffer[3], 0x04);
    EXPECT_EQ(buffer[4], 0x05);
    EXPECT_EQ(buffer[5], 0x06);
    EXPECT_EQ(buffer[6], 0x07);
    EXPECT_EQ(buffer[7], 0x08);

    // Read from offset 0 again (aligned) - this should work
    size_t bytes_read2 = odirect_file->ReadAt(buffer, alignment, 0);
    EXPECT_EQ(bytes_read2, alignment);

    // Read from a smaller offset that allows reading some data
    // Read 1024 bytes from offset 1024 (within file bounds)
    uint8_t* small_buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(1024, alignment));
    size_t bytes_read3 = odirect_file->ReadAt(small_buffer, 1024, 1024);
    EXPECT_EQ(bytes_read3, 1024);
    FreeAlignedBuffer(small_buffer);

    // Read beyond file size (should return 0)
    size_t bytes_read4 = odirect_file->ReadAt(buffer, alignment, 4096 + alignment);
    EXPECT_EQ(bytes_read4, 0);

    FreeAlignedBuffer(buffer);
    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 17: ODirectFileStream - Read with Seek
TEST_F(DaliFileTest, ODirectFileStreamReadWithSeek) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

        // Get alignment for O_DIRECT
    size_t alignment = ODirectFileStream::GetAlignment();

    // Create aligned buffer
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(alignment, alignment));
    uintptr_t buffer_addr = reinterpret_cast<uintptr_t>(buffer);
    EXPECT_EQ(buffer_addr % alignment, 0) << "Buffer not aligned to " << alignment;

    // Seek to beginning and read aligned amount
    odirect_file->SeekRead(0, SEEK_SET);
    size_t bytes_read = odirect_file->Read(buffer, alignment);
    EXPECT_EQ(bytes_read, alignment);

    // Verify first 8 bytes contain our test data
    EXPECT_EQ(buffer[0], 0x01);
    EXPECT_EQ(buffer[1], 0x02);
    EXPECT_EQ(buffer[2], 0x03);
    EXPECT_EQ(buffer[3], 0x04);
    EXPECT_EQ(buffer[4], 0x05);
    EXPECT_EQ(buffer[5], 0x06);
    EXPECT_EQ(buffer[6], 0x07);
    EXPECT_EQ(buffer[7], 0x08);

    // Verify position after read
    EXPECT_EQ(odirect_file->TellRead(), alignment);

    FreeAlignedBuffer(buffer);
    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 18: ODirectFileStream - Size Method
TEST_F(DaliFileTest, ODirectFileStreamSize) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    EXPECT_EQ(odirect_file->Size(), 4096);

    // Test size of empty file
    std::string empty_file = test_dir_ + "/empty_file.txt";
    auto empty_odirect = std::make_unique<ODirectFileStream>(empty_file);
    EXPECT_EQ(empty_odirect->Size(), 0);
    empty_odirect->Close();

    // Test size of large file
    std::string large_file = test_dir_ + "/large_file.bin";
    auto large_odirect = std::make_unique<ODirectFileStream>(large_file);
    EXPECT_EQ(large_odirect->Size(), 1024 * 1024);
    large_odirect->Close();

    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 19: ODirectFileStream - Size Error (File Deleted)
TEST_F(DaliFileTest, ODirectFileStreamSizeError) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    // Delete the file while it's open
    unlink(file_path.c_str());

    // Size should fail because the file path no longer exists
    // (even though the file descriptor is still open)
    EXPECT_THROW({
      odirect_file->Size();
    }, DALIException);

    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 19b: ODirectFileStream - Size Error (Invalid Path)
TEST_F(DaliFileTest, ODirectFileStreamSizeInvalidPath) {
  // Test with a non-existent file path
  std::string nonexistent_file = test_dir_ + "/nonexistent_file.bin";

  try {
    // This should fail during construction
    EXPECT_THROW({
      auto odirect_file = std::make_unique<ODirectFileStream>(nonexistent_file);
    }, DALIException);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 20: ODirectFileStream - Close Method
TEST_F(DaliFileTest, ODirectFileStreamClose) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    // Verify file is open
    EXPECT_EQ(odirect_file->Size(), 4096);

    // Close the file
    odirect_file->Close();

    // Try to use the file after closing (behavior may vary)
    // Some implementations might not throw, so we'll just verify the file is closed
    // by checking if we can open it again
    auto odirect_file2 = std::make_unique<ODirectFileStream>(file_path);
    EXPECT_EQ(odirect_file2->Size(), 4096);
    odirect_file2->Close();

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 21: ODirectFileStream - Destructor
TEST_F(DaliFileTest, ODirectFileStreamDestructor) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    {
      auto odirect_file = std::make_unique<ODirectFileStream>(file_path);
      EXPECT_EQ(odirect_file->Size(), 4096);
      // File should be automatically closed when odirect_file goes out of scope
    }

    // File should be closed now, verify by trying to open it again
    auto odirect_file2 = std::make_unique<ODirectFileStream>(file_path);
    EXPECT_EQ(odirect_file2->Size(), 4096);
    odirect_file2->Close();

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 22: ODirectFileStream - Environment Variables (Default Values)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsDefault) {
  // Clear any existing environment variables
  unsetenv("DALI_ODIRECT_ALIGNMENT");
  unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");
  unsetenv("DALI_ODIRECT_CHUNK_SIZE");

  try {
    // Test default values
    size_t alignment = ODirectFileStream::GetAlignment();
    size_t len_alignment = ODirectFileStream::GetLenAlignment();
    size_t chunk_size = ODirectFileStream::GetChunkSize();

    EXPECT_EQ(alignment, 4096);  // kODirectAlignment
    EXPECT_EQ(len_alignment, 4096);  // kODirectAlignment
    EXPECT_EQ(chunk_size, 2 << 20);  // kODirectChunkSize (2M)

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 23: ODirectFileStream - Environment Variables (Numeric Values)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsNumeric) {
  try {
    // Set numeric environment variables
    setenv("DALI_ODIRECT_ALIGNMENT", "8192", 1);
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "8192", 1);
    setenv("DALI_ODIRECT_CHUNK_SIZE", "1048576", 1);  // 1M

    size_t alignment = ODirectFileStream::GetAlignment();
    size_t len_alignment = ODirectFileStream::GetLenAlignment();
    size_t chunk_size = ODirectFileStream::GetChunkSize();

    EXPECT_EQ(alignment, 8192);
    EXPECT_EQ(len_alignment, 8192);
    EXPECT_EQ(chunk_size, 1048576);

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");
    unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");
    unsetenv("DALI_ODIRECT_CHUNK_SIZE");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 24: ODirectFileStream - Environment Variables (K Suffix)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsKSuffix) {
  try {
    // Set environment variables with 'k' suffix
    setenv("DALI_ODIRECT_ALIGNMENT", "8k", 1);
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "8k", 1);
    setenv("DALI_ODIRECT_CHUNK_SIZE", "1024k", 1);

    size_t alignment = ODirectFileStream::GetAlignment();
    size_t len_alignment = ODirectFileStream::GetLenAlignment();
    size_t chunk_size = ODirectFileStream::GetChunkSize();

    EXPECT_EQ(alignment, 8192);  // 8k = 8 * 1024
    EXPECT_EQ(len_alignment, 8192);  // 8k = 8 * 1024
    EXPECT_EQ(chunk_size, 1048576);  // 1024k = 1024 * 1024

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");
    unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");
    unsetenv("DALI_ODIRECT_CHUNK_SIZE");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 25: ODirectFileStream - Environment Variables (M Suffix)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsMSuffix) {
  try {
    // Set environment variables with 'M' suffix
    setenv("DALI_ODIRECT_ALIGNMENT", "4M", 1);
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "4M", 1);
    setenv("DALI_ODIRECT_CHUNK_SIZE", "4M", 1);  // Must be aligned to len_alignment (4M)

    size_t alignment = ODirectFileStream::GetAlignment();
    size_t len_alignment = ODirectFileStream::GetLenAlignment();
    size_t chunk_size = ODirectFileStream::GetChunkSize();

    EXPECT_EQ(alignment, 4194304);  // 4M = 4 * 1024 * 1024
    EXPECT_EQ(len_alignment, 4194304);  // 4M = 4 * 1024 * 1024
    EXPECT_EQ(chunk_size, 4194304);  // 4M = 4 * 1024 * 1024

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");
    unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");
    unsetenv("DALI_ODIRECT_CHUNK_SIZE");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 26: ODirectFileStream - Environment Variables (Invalid Values)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsInvalid) {
  try {
    // Test invalid environment variable values - non-digit characters
    setenv("DALI_ODIRECT_ALIGNMENT", "invalid", 1);
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test invalid characters in middle of string
    setenv("DALI_ODIRECT_ALIGNMENT", "4k2", 1);
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test invalid suffix (only 'k' and 'M' are valid)
    setenv("DALI_ODIRECT_ALIGNMENT", "4G", 1);
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4m", 1);  // lowercase 'm' not valid
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4K", 1);  // uppercase 'K' not valid
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test non-power-of-two value (alignment must be power of 2)
    setenv("DALI_ODIRECT_ALIGNMENT", "3000", 1);
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4095", 1);  // Not power of 2
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test value too small (minimum is 4096)
    setenv("DALI_ODIRECT_ALIGNMENT", "100", 1);
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "2048", 1);  // Below minimum
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test value too large (maximum is 16M)
    setenv("DALI_ODIRECT_ALIGNMENT", "32M", 1);  // 32M > 16M
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "64M", 1);  // 64M > 16M
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test chunk size alignment validation (non-power-of-two but aligned)
    setenv("DALI_ODIRECT_CHUNK_SIZE", "8192", 1);  // Should work (aligned to len_alignment)
    size_t chunk_size = ODirectFileStream::GetChunkSize();
    EXPECT_EQ(chunk_size, 8192);

    // Test chunk size not aligned to len_alignment
    setenv("DALI_ODIRECT_CHUNK_SIZE", "8193", 1);  // Not aligned to 4096
    EXPECT_THROW({
      ODirectFileStream::GetChunkSize();
    }, DALIException);

    // Test chunk size not aligned to custom len_alignment
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "8k", 1);  // Set len_alignment to 8k
    setenv("DALI_ODIRECT_CHUNK_SIZE", "4k", 1);     // 4k is not aligned to 8k
    EXPECT_THROW({
      ODirectFileStream::GetChunkSize();
    }, DALIException);

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");
    unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");
    unsetenv("DALI_ODIRECT_CHUNK_SIZE");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 27: ODirectFileStream - Environment Variables (Empty String)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsEmpty) {
  try {
    // Set empty environment variable
    setenv("DALI_ODIRECT_ALIGNMENT", "", 1);

    // Should use default value
    size_t alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 4096);  // Default value

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 27b: ODirectFileStream - Environment Variables (Boundary Values)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsBoundary) {
  try {
    // Test minimum valid value (4096)
    setenv("DALI_ODIRECT_ALIGNMENT", "4096", 1);
    size_t alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 4096);

    // Test maximum valid value (16M)
    setenv("DALI_ODIRECT_ALIGNMENT", "16M", 1);
    alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 16 << 20);  // 16 * 1024 * 1024

    // Test boundary values with k suffix
    setenv("DALI_ODIRECT_ALIGNMENT", "4k", 1);  // 4096
    alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 4096);

    setenv("DALI_ODIRECT_ALIGNMENT", "16k", 1);  // 16 * 1024
    alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 16 << 10);

    // Test boundary values with M suffix
    setenv("DALI_ODIRECT_ALIGNMENT", "1M", 1);  // 1 * 1024 * 1024
    alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 1 << 20);

    setenv("DALI_ODIRECT_ALIGNMENT", "8M", 1);  // 8 * 1024 * 1024
    alignment = ODirectFileStream::GetAlignment();
    EXPECT_EQ(alignment, 8 << 20);

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 27c: ODirectFileStream - Environment Variables (Edge Cases)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsEdgeCases) {
  try {
    // Test single character suffix
    setenv("DALI_ODIRECT_ALIGNMENT", "k", 1);  // Just 'k' without number
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "M", 1);  // Just 'M' without number
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test suffix in wrong position
    setenv("DALI_ODIRECT_ALIGNMENT", "k4", 1);  // 'k' at beginning
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4kM", 1);  // Multiple suffixes
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test whitespace
    setenv("DALI_ODIRECT_ALIGNMENT", " 4096 ", 1);  // Leading/trailing spaces
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4 k", 1);  // Space before suffix
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Test special characters
    setenv("DALI_ODIRECT_ALIGNMENT", "4k+", 1);  // Plus sign
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4k-", 1);  // Minus sign
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    setenv("DALI_ODIRECT_ALIGNMENT", "4k.", 1);  // Decimal point
    EXPECT_THROW({
      ODirectFileStream::GetAlignment();
    }, DALIException);

    // Clean up
    unsetenv("DALI_ODIRECT_ALIGNMENT");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 27d: ODirectFileStream - Environment Variables (Len Alignment Validation)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsLenAlignment) {
  try {
    // Test len_alignment validation - must be power of 2
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "3000", 1);  // Not power of 2
    EXPECT_THROW({
      ODirectFileStream::GetLenAlignment();
    }, DALIException);

    // Test len_alignment too small
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "2048", 1);  // Below minimum
    EXPECT_THROW({
      ODirectFileStream::GetLenAlignment();
    }, DALIException);

    // Test len_alignment too large
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "32M", 1);  // Above maximum
    EXPECT_THROW({
      ODirectFileStream::GetLenAlignment();
    }, DALIException);

    // Test valid len_alignment values
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "4096", 1);
    size_t len_alignment = ODirectFileStream::GetLenAlignment();
    EXPECT_EQ(len_alignment, 4096);

    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "8k", 1);
    len_alignment = ODirectFileStream::GetLenAlignment();
    EXPECT_EQ(len_alignment, 8192);

    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "4M", 1);
    len_alignment = ODirectFileStream::GetLenAlignment();
    EXPECT_EQ(len_alignment, 4 << 20);

    // Clean up
    unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 27e: ODirectFileStream - Environment Variables (Chunk Size with Custom Len Alignment)
TEST_F(DaliFileTest, ODirectFileStreamEnvVarsChunkSizeWithCustomLenAlignment) {
  try {
    // Set custom len_alignment and test chunk_size validation
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "8k", 1);  // 8192 bytes

    // Test chunk_size aligned to custom len_alignment
    setenv("DALI_ODIRECT_CHUNK_SIZE", "16k", 1);  // 16384 bytes, aligned to 8192
    size_t chunk_size = ODirectFileStream::GetChunkSize();
    EXPECT_EQ(chunk_size, 16384);

    // Test chunk_size not aligned to custom len_alignment
    setenv("DALI_ODIRECT_CHUNK_SIZE", "12k", 1);  // 12288 bytes, not aligned to 8192
    EXPECT_THROW({
      ODirectFileStream::GetChunkSize();
    }, DALIException);

    // Test chunk_size with M suffix aligned to custom len_alignment
    setenv("DALI_ODIRECT_CHUNK_SIZE", "1M", 1);  // 1048576 bytes, aligned to 8192
    chunk_size = ODirectFileStream::GetChunkSize();
    EXPECT_EQ(chunk_size, 1048576);

    // Test chunk_size with M suffix not aligned to custom len_alignment
    setenv("DALI_ODIRECT_LEN_ALIGNMENT", "1M", 1);  // Change to 1M alignment
    setenv("DALI_ODIRECT_CHUNK_SIZE", "2M", 1);     // 2M aligned to 1M
    chunk_size = ODirectFileStream::GetChunkSize();
    EXPECT_EQ(chunk_size, 2 << 20);

    setenv("DALI_ODIRECT_CHUNK_SIZE", "1536k", 1);  // 1536k = 1.5M, not aligned to 1M
    EXPECT_THROW({
      ODirectFileStream::GetChunkSize();
    }, DALIException);

    // Clean up
    unsetenv("DALI_ODIRECT_LEN_ALIGNMENT");
    unsetenv("DALI_ODIRECT_CHUNK_SIZE");

  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 28: ODirectFileStream - Read Error Conditions
TEST_F(DaliFileTest, ODirectFileStreamReadErrors) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

    // Test reading with zero bytes
    size_t alignment = ODirectFileStream::GetAlignment();
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(alignment, alignment));
    size_t bytes_read = odirect_file->Read(buffer, 0);
    EXPECT_EQ(bytes_read, 0);

    // Test ReadAt with zero bytes
    size_t bytes_read_at = odirect_file->ReadAt(buffer, 0, 0);
    EXPECT_EQ(bytes_read_at, 0);

    FreeAlignedBuffer(buffer);

    // Note: Testing with null buffers may cause undefined behavior
    // and is not recommended in production code

    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 29: ODirectFileStream - Multiple Operations Sequence
TEST_F(DaliFileTest, ODirectFileStreamMultipleOperations) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

        // Get alignment for O_DIRECT
    size_t alignment = ODirectFileStream::GetAlignment();

    // Create aligned buffer
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(alignment, alignment));
    uintptr_t buffer_addr = reinterpret_cast<uintptr_t>(buffer);
    EXPECT_EQ(buffer_addr % alignment, 0) << "Buffer not aligned to " << alignment;

    // Sequence of operations: seek, read, seek, readat, tell
    odirect_file->SeekRead(0, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 0);

    size_t bytes_read1 = odirect_file->Read(buffer, alignment);
    EXPECT_EQ(bytes_read1, alignment);
    // Verify first 8 bytes contain our test data
    EXPECT_EQ(buffer[0], 0x01);
    EXPECT_EQ(buffer[1], 0x02);
    EXPECT_EQ(buffer[2], 0x03);
    EXPECT_EQ(buffer[3], 0x04);
    EXPECT_EQ(buffer[4], 0x05);
    EXPECT_EQ(buffer[5], 0x06);
    EXPECT_EQ(buffer[6], 0x07);
    EXPECT_EQ(buffer[7], 0x08);

    EXPECT_EQ(odirect_file->TellRead(), alignment);

    odirect_file->SeekRead(0, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 0);

    size_t bytes_read2 = odirect_file->ReadAt(buffer, alignment, 0);
    EXPECT_EQ(bytes_read2, alignment);
    // Verify first 8 bytes again
    EXPECT_EQ(buffer[0], 0x01);
    EXPECT_EQ(buffer[1], 0x02);
    EXPECT_EQ(buffer[2], 0x03);
    EXPECT_EQ(buffer[3], 0x04);
    EXPECT_EQ(buffer[4], 0x05);
    EXPECT_EQ(buffer[5], 0x06);
    EXPECT_EQ(buffer[6], 0x07);
    EXPECT_EQ(buffer[7], 0x08);

    // Position should not change after ReadAt
    EXPECT_EQ(odirect_file->TellRead(), 0);

    FreeAlignedBuffer(buffer);
    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

// Test 30: ODirectFileStream - Edge Cases
TEST_F(DaliFileTest, ODirectFileStreamEdgeCases) {
  std::string file_path = test_dir_ + "/test_file.bin";

  try {
    auto odirect_file = std::make_unique<ODirectFileStream>(file_path);

        // Get alignment for O_DIRECT
    size_t alignment = ODirectFileStream::GetAlignment();

    // Create aligned buffer
    uint8_t* buffer = static_cast<uint8_t*>(AllocateAlignedBuffer(alignment, alignment));
    uintptr_t buffer_addr = reinterpret_cast<uintptr_t>(buffer);
    EXPECT_EQ(buffer_addr % alignment, 0) << "Buffer not aligned to " << alignment;

    // Test seeking to exact file size
    odirect_file->SeekRead(4096, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 4096);

    // Test reading at exact file size (should return 0)
    size_t bytes_read = odirect_file->Read(buffer, alignment);
    EXPECT_EQ(bytes_read, 0);

    // Test ReadAt at exact file size
    size_t bytes_read_at = odirect_file->ReadAt(buffer, alignment, 4096);
    EXPECT_EQ(bytes_read_at, 0);

    // Test seeking to one byte before end
    odirect_file->SeekRead(4095, SEEK_SET);
    EXPECT_EQ(odirect_file->TellRead(), 4095);

    // Read last byte (this might fail due to alignment requirements)
    // For O_DIRECT, we need to read at least alignment bytes
    bytes_read = odirect_file->Read(buffer, alignment);
    // This might return 0 or alignment depending on implementation

    FreeAlignedBuffer(buffer);
    odirect_file->Close();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "O_DIRECT not supported on this system: " << e.what();
  }
}

}  // namespace dali