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
#include <cstdlib>
#include <thread>
#include "dali/operators/decoder/nvjpeg/nvjpeg_memory.h"

namespace dali {
namespace nvjpeg_memory {
namespace testing {

// ============================================================================
// GetHostBuffer — covers lines 265-270 (0% → should reach ~100%)
// ============================================================================

TEST(NvjpegMemoryTest, GetHostBufferReturnsNonNull) {
  auto tid = std::this_thread::get_id();
  // Pre-add a buffer so GetHostBuffer can find it in the pool
  AddHostBuffer(tid, 1024);
  void *ptr = GetHostBuffer(tid, 512);
  EXPECT_NE(ptr, nullptr);
  // Return the buffer to the pool for cleanup
  DeleteAllBuffers(tid);
}

// ============================================================================
// DeviceNew via GetDeviceAllocator — size == 0 path (lines 292-294)
// ============================================================================

TEST(NvjpegMemoryTest, DeviceAllocatorZeroSize) {
  auto alloc = GetDeviceAllocator();
  void *ptr = reinterpret_cast<void *>(0xDEAD);
  int ret = alloc.dev_malloc(&ptr, 0);
  EXPECT_EQ(ret, 0);  // cudaSuccess
  EXPECT_EQ(ptr, nullptr);
}

// ============================================================================
// DeviceNew via GetDeviceAllocator — normal allocation (lines 297-299)
// and free via ReturnBufferToPool (line 284-286)
// ============================================================================

TEST(NvjpegMemoryTest, DeviceAllocatorNormalAlloc) {
  auto tid = std::this_thread::get_id();
  // Pre-add a device buffer
  AddBuffer<mm::memory_kind::device>(tid, 2048);

  auto alloc = GetDeviceAllocator();
  void *ptr = nullptr;
  int ret = alloc.dev_malloc(&ptr, 1024);
  EXPECT_EQ(ret, 0);  // cudaSuccess
  EXPECT_NE(ptr, nullptr);

  // Free via the allocator (returns to pool)
  ret = alloc.dev_free(ptr);
  EXPECT_EQ(ret, 0);

  DeleteAllBuffers(tid);
}

// ============================================================================
// HostNew via GetPinnedAllocator — size == 0 path (lines 333-335)
// ============================================================================

TEST(NvjpegMemoryTest, PinnedAllocatorZeroSize) {
  auto alloc = GetPinnedAllocator();
  void *ptr = reinterpret_cast<void *>(0xDEAD);
  int ret = alloc.pinned_malloc(&ptr, 0, 0);
  EXPECT_EQ(ret, 0);  // cudaSuccess
  EXPECT_EQ(ptr, nullptr);
}

// ============================================================================
// HostNew via GetPinnedAllocator — normal allocation (lines 337-342)
// ============================================================================

TEST(NvjpegMemoryTest, PinnedAllocatorNormalAlloc) {
  auto tid = std::this_thread::get_id();
  AddHostBuffer(tid, 2048);

  auto alloc = GetPinnedAllocator();
  void *ptr = nullptr;
  int ret = alloc.pinned_malloc(&ptr, 1024, 0);
  EXPECT_EQ(ret, 0);  // cudaSuccess
  EXPECT_NE(ptr, nullptr);

  // Free via the allocator
  ret = alloc.pinned_free(ptr);
  EXPECT_EQ(ret, 0);

  DeleteAllBuffers(tid);
}

// ============================================================================
// PrintMemStats with DALI_LOG_FILE — covers line 119 (log_filename branch)
// ============================================================================

TEST(NvjpegMemoryTest, PrintMemStatsToFile) {
  // Set DALI_LOG_FILE to a temp file
  const char *tmpfile = "/tmp/nvjpeg_mem_stats_test.log";
  setenv("DALI_LOG_FILE", tmpfile, 1);

  SetEnableMemStats(true);
  PrintMemStats();

  // Unset env to avoid side effects
  unsetenv("DALI_LOG_FILE");

  // Verify the file was written
  FILE *f = fopen(tmpfile, "r");
  ASSERT_NE(f, nullptr);
  char buf[256] = {};
  size_t read = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  EXPECT_GT(read, 0u);
  // Clean up
  remove(tmpfile);
}

// ============================================================================
// PrintMemStats with stats disabled — covers the !mem_stats_enabled_ path
// ============================================================================

TEST(NvjpegMemoryTest, PrintMemStatsDisabled) {
  SetEnableMemStats(false);
  // Should do nothing and not crash
  PrintMemStats();
  // Re-enable for other tests
  SetEnableMemStats(true);
}

// ============================================================================
// SetEnableMemStats — already covered, but ensures toggle works
// ============================================================================

TEST(NvjpegMemoryTest, SetEnableMemStats) {
  SetEnableMemStats(false);
  SetEnableMemStats(true);
}

// ============================================================================
// DeleteAllBuffers on a thread with no buffers — covers line 228-229
// ============================================================================

TEST(NvjpegMemoryTest, DeleteAllBuffersNoBuffers) {
  // Use a thread id that has no buffers allocated
  std::thread::id empty_tid;
  // Should not crash
  DeleteAllBuffers(empty_tid);
}

}  // namespace testing
}  // namespace nvjpeg_memory
}  // namespace dali
