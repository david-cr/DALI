// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sys/resource.h>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>
#include "dali/core/os/shared_mem.h"

namespace dali {

namespace detail {
// Defined in shared_mem.cc, but not declared in the header.
bool dir_exists(const char *str);
}  // namespace detail

namespace {

constexpr uint64_t kPageSize = 4096;

// ftruncate takes a signed off_t, so a size with the top bit set is always rejected.
constexpr uint64_t kUntruncatableSize = 1ull << 63;

/** Returns a descriptor that cannot possibly be open.
 *
 * The kernel never hands out descriptors at or above the RLIMIT_NOFILE soft limit, so closing
 * this one is guaranteed to fail instead of stealing a descriptor from somebody else.
 */
int ClosedFd() {
  rlimit lim{};
  if (getrlimit(RLIMIT_NOFILE, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY &&
      lim.rlim_cur < (1u << 30))
    return static_cast<int>(lim.rlim_cur);
  return 1 << 20;
}

}  // namespace

TEST(SharedMem, DirExists) {
  EXPECT_TRUE(detail::dir_exists("/"));
  EXPECT_FALSE(detail::dir_exists("/no/such/directory/here"));
}

TEST(SharedMem, DestroyInvalidFileHandle) {
  EXPECT_THROW(FileHandle::DestroyHandle(ClosedFd()), std::runtime_error);
  EXPECT_NO_THROW(FileHandle::DestroyHandle(-1)) << "A null handle has nothing to close.";
}

TEST(SharedMem, DestroyInvalidShmHandle) {
  EXPECT_THROW(ShmHandle::DestroyHandle(ClosedFd()), std::runtime_error);
  EXPECT_NO_THROW(ShmHandle::DestroyHandle(-1)) << "A null handle has nothing to close.";
}

TEST(SharedMem, DestroyInvalidMapping) {
  // munmap rejects an address that isn't page aligned - and unmaps nothing.
  MappedMemoryChunk misaligned{kPageSize, reinterpret_cast<uint8_t *>(1)};
  EXPECT_THROW(MemoryMapping::DestroyHandle(misaligned), std::runtime_error);

  MappedMemoryChunk empty{0, nullptr};
  EXPECT_NO_THROW(MemoryMapping::DestroyHandle(empty)) << "A null mapping has nothing to unmap.";
}

TEST(SharedMem, ErrorResizeUnmapped) {
  MemoryMapping mapping;
  EXPECT_THROW(mapping.resize(kPageSize), std::logic_error);
}

TEST(SharedMem, CreateFromExistingHandle) {
  SharedMem owner(-1, kPageSize);
  ASSERT_GE(owner.handle(), 0);
  owner.get_raw_ptr()[0] = 42;

  int dup_fd = dup(owner.handle());
  ASSERT_GE(dup_fd, 0);
  SharedMem attached(dup_fd, kPageSize);  // takes over dup_fd instead of allocating new memory
  EXPECT_EQ(attached.size(), kPageSize);
  EXPECT_EQ(attached.get_raw_ptr()[0], 42) << "Both objects should map the same memory.";
}

TEST(SharedMem, ErrorTruncateOnCreate) {
  EXPECT_THROW(SharedMem(-1, kUntruncatableSize), std::runtime_error);
}

TEST(SharedMem, ErrorTruncateOnResize) {
  SharedMem sm(-1, kPageSize);
  EXPECT_THROW(sm.resize(kUntruncatableSize, true), std::runtime_error);
}

TEST(SharedMem, ErrorResizeWithoutHandle) {
  SharedMem sm(-1, kPageSize);
  sm.close_handle();
  ASSERT_EQ(sm.handle(), -1);
  EXPECT_THROW(sm.resize(2 * kPageSize, true), std::logic_error);
}

TEST(SharedMem, ErrorMapWithoutHandle) {
  SharedMem sm(-1, kPageSize);
  sm.close();
  ASSERT_EQ(sm.get_raw_ptr(), nullptr);
  EXPECT_THROW(sm.resize(2 * kPageSize, false), std::logic_error);
}

}  // namespace dali
