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
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "dali/operators/decoder/cache/cached_decoder_impl.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

class CachedDecoderImplTest : public ::testing::Test {
 protected:
  // Create OpSpec with caching disabled (no cache_size or cache_size=0)
  OpSpec MakeNoCacheSpec() {
    return OpSpec("decoders__Image")
        .AddArg("device", "mixed")
        .AddArg("device_id", 0)
        .AddArg("max_batch_size", 4)
        .AddArg("num_threads", 1)
        .AddArg("cache_size", 0);
  }

  // Create OpSpec with caching enabled
  OpSpec MakeCacheSpec(int cache_size_mb = 16,
                       int cache_threshold = 0,
                       const std::string &cache_type = "threshold",
                       bool cache_debug = false,
                       bool cache_batch_copy = true) {
    return OpSpec("decoders__Image")
        .AddArg("device", "mixed")
        .AddArg("device_id", 0)
        .AddArg("max_batch_size", 4)
        .AddArg("num_threads", 1)
        .AddArg("cache_size", cache_size_mb)
        .AddArg("cache_threshold", cache_threshold)
        .AddArg("cache_type", cache_type)
        .AddArg("cache_debug", cache_debug)
        .AddArg("cache_batch_copy", cache_batch_copy);
  }
};

// ============================================================================
// No-cache path (cache_ == nullptr)
// ============================================================================

TEST_F(CachedDecoderImplTest, ConstructNoCacheDisabled) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  EXPECT_FALSE(impl.IsCacheEnabled());
}

TEST_F(CachedDecoderImplTest, NoCacheCacheLoadReturnsFalse) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy;
  EXPECT_FALSE(impl.CacheLoad("test.jpg", &dummy, nullptr));
}

TEST_F(CachedDecoderImplTest, NoCacheDeferCacheLoadReturnsFalse) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy;
  EXPECT_FALSE(impl.DeferCacheLoad("test.jpg", &dummy));
}

TEST_F(CachedDecoderImplTest, NoCacheLoadDeferredNoOp) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  EXPECT_NO_THROW(impl.LoadDeferred(nullptr));
}

TEST_F(CachedDecoderImplTest, NoCacheIsInCacheReturnsFalse) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  EXPECT_FALSE(impl.IsInCache("test.jpg"));
}

TEST_F(CachedDecoderImplTest, NoCacheCacheImageShapeEmpty) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  auto shape = impl.CacheImageShape("test.jpg");
  EXPECT_EQ(shape.size(), 3);
}

TEST_F(CachedDecoderImplTest, NoCacheCacheStoreNoOp) {
  auto spec = MakeNoCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy = 0;
  ImageCache::ImageShape shape{1, 1, 3};
  EXPECT_NO_THROW(impl.CacheStore("test.jpg", &dummy, shape, nullptr));
}

// ============================================================================
// Cache-enabled path
// ============================================================================

TEST_F(CachedDecoderImplTest, ConstructWithCacheEnabled) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);
  EXPECT_TRUE(impl.IsCacheEnabled());
}

TEST_F(CachedDecoderImplTest, CacheStoreAndIsInCache) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);

  // Initially not in cache
  EXPECT_FALSE(impl.IsInCache("img1.jpg"));

  // Store a small image in the cache (GPU memory)
  const int H = 2, W = 2, C = 3;
  const int nbytes = H * W * C;
  uint8_t host_data[nbytes];
  std::memset(host_data, 0xAB, nbytes);

  uint8_t *gpu_data = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_data, nbytes));
  CUDA_CALL(cudaMemcpy(gpu_data, host_data, nbytes, cudaMemcpyHostToDevice));

  ImageCache::ImageShape shape{H, W, C};
  impl.CacheStore("img1.jpg", gpu_data, shape, nullptr);
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Now should be in cache
  EXPECT_TRUE(impl.IsInCache("img1.jpg"));

  CUDA_CALL(cudaFree(gpu_data));
}

TEST_F(CachedDecoderImplTest, CacheStoreAndCacheImageShape) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);

  const int H = 4, W = 8, C = 3;
  const int nbytes = H * W * C;

  uint8_t *gpu_data = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_data, nbytes));
  CUDA_CALL(cudaMemset(gpu_data, 0, nbytes));

  ImageCache::ImageShape shape{H, W, C};
  impl.CacheStore("img2.jpg", gpu_data, shape, nullptr);
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  auto cached_shape = impl.CacheImageShape("img2.jpg");
  EXPECT_EQ(cached_shape[0], H);
  EXPECT_EQ(cached_shape[1], W);
  EXPECT_EQ(cached_shape[2], C);

  CUDA_CALL(cudaFree(gpu_data));
}

TEST_F(CachedDecoderImplTest, CacheStoreAndCacheLoad) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);

  const int H = 2, W = 3, C = 3;
  const int nbytes = H * W * C;
  uint8_t host_data[nbytes];
  for (int i = 0; i < nbytes; i++) host_data[i] = static_cast<uint8_t>(i);

  uint8_t *gpu_data = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_data, nbytes));
  CUDA_CALL(cudaMemcpy(gpu_data, host_data, nbytes, cudaMemcpyHostToDevice));

  ImageCache::ImageShape shape{H, W, C};
  impl.CacheStore("img3.jpg", gpu_data, shape, nullptr);
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Now load from cache into a different GPU buffer
  uint8_t *gpu_out = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_out, nbytes));
  CUDA_CALL(cudaMemset(gpu_out, 0, nbytes));

  bool loaded = impl.CacheLoad("img3.jpg", gpu_out, nullptr);
  EXPECT_TRUE(loaded);
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Verify data matches
  uint8_t host_out[nbytes];
  CUDA_CALL(cudaMemcpy(host_out, gpu_out, nbytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < nbytes; i++) {
    EXPECT_EQ(host_out[i], host_data[i]) << "Mismatch at byte " << i;
  }

  CUDA_CALL(cudaFree(gpu_data));
  CUDA_CALL(cudaFree(gpu_out));
}

TEST_F(CachedDecoderImplTest, CacheLoadEmptyNameReturnsFalse) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy;
  EXPECT_FALSE(impl.CacheLoad("", &dummy, nullptr));
}

TEST_F(CachedDecoderImplTest, CacheLoadNonExistentReturnsFalse) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy;
  EXPECT_FALSE(impl.CacheLoad("nonexistent.jpg", &dummy, nullptr));
}

TEST_F(CachedDecoderImplTest, DeferCacheLoadAndLoadDeferred) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);

  const int H = 2, W = 2, C = 3;
  const int nbytes = H * W * C;

  uint8_t *gpu_data = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_data, nbytes));
  CUDA_CALL(cudaMemset(gpu_data, 0x42, nbytes));

  ImageCache::ImageShape shape{H, W, C};
  impl.CacheStore("img_defer.jpg", gpu_data, shape, nullptr);
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // DeferCacheLoad should return true for a cached image
  uint8_t *gpu_out = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_out, nbytes));
  CUDA_CALL(cudaMemset(gpu_out, 0, nbytes));

  bool deferred = impl.DeferCacheLoad("img_defer.jpg", gpu_out);
  EXPECT_TRUE(deferred);

  // LoadDeferred executes the actual copy
  EXPECT_NO_THROW(impl.LoadDeferred(nullptr));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Verify data was copied
  uint8_t host_out[nbytes];
  CUDA_CALL(cudaMemcpy(host_out, gpu_out, nbytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < nbytes; i++) {
    EXPECT_EQ(host_out[i], 0x42) << "Mismatch at byte " << i;
  }

  CUDA_CALL(cudaFree(gpu_data));
  CUDA_CALL(cudaFree(gpu_out));
}

TEST_F(CachedDecoderImplTest, DeferCacheLoadEmptyNameReturnsFalse) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy;
  EXPECT_FALSE(impl.DeferCacheLoad("", &dummy));
}

TEST_F(CachedDecoderImplTest, DeferCacheLoadNonExistentReturnsFalse) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy;
  EXPECT_FALSE(impl.DeferCacheLoad("nonexistent.jpg", &dummy));
}

TEST_F(CachedDecoderImplTest, CacheStoreEmptyNameNoOp) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);
  uint8_t dummy = 0;
  ImageCache::ImageShape shape{1, 1, 3};
  // Should not crash - empty name is just ignored
  EXPECT_NO_THROW(impl.CacheStore("", &dummy, shape, nullptr));
}

TEST_F(CachedDecoderImplTest, CacheStoreDuplicateIgnored) {
  auto spec = MakeCacheSpec();
  CachedDecoderImpl impl(spec);

  const int nbytes = 12;
  uint8_t *gpu_data = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_data, nbytes));
  CUDA_CALL(cudaMemset(gpu_data, 0xAA, nbytes));

  ImageCache::ImageShape shape{2, 2, 3};
  impl.CacheStore("dup.jpg", gpu_data, shape, nullptr);
  CUDA_CALL(cudaStreamSynchronize(nullptr));
  EXPECT_TRUE(impl.IsInCache("dup.jpg"));

  // Storing again should be a no-op (IsCached returns true)
  EXPECT_NO_THROW(impl.CacheStore("dup.jpg", gpu_data, shape, nullptr));

  CUDA_CALL(cudaFree(gpu_data));
}

TEST_F(CachedDecoderImplTest, ConstructWithCacheBatchCopyFalse) {
  auto spec = MakeCacheSpec(16, 0, "threshold", false, false);
  CachedDecoderImpl impl(spec);
  EXPECT_TRUE(impl.IsCacheEnabled());
}

TEST_F(CachedDecoderImplTest, ConstructWithCacheDebug) {
  auto spec = MakeCacheSpec(16, 0, "threshold", true, true);
  CachedDecoderImpl impl(spec);
  EXPECT_TRUE(impl.IsCacheEnabled());
}

TEST_F(CachedDecoderImplTest, LoadDeferredWithBatchCopyFalse) {
  auto spec = MakeCacheSpec(16, 0, "threshold", false, false);
  CachedDecoderImpl impl(spec);

  const int nbytes = 12;
  uint8_t *gpu_data = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_data, nbytes));
  CUDA_CALL(cudaMemset(gpu_data, 0x55, nbytes));

  ImageCache::ImageShape shape{2, 2, 3};
  impl.CacheStore("batch_false.jpg", gpu_data, shape, nullptr);
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  uint8_t *gpu_out = nullptr;
  CUDA_CALL(cudaMalloc(&gpu_out, nbytes));
  bool deferred = impl.DeferCacheLoad("batch_false.jpg", gpu_out);
  EXPECT_TRUE(deferred);

  // LoadDeferred with Memcpy method (cache_batch_copy=false)
  EXPECT_NO_THROW(impl.LoadDeferred(nullptr));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  CUDA_CALL(cudaFree(gpu_data));
  CUDA_CALL(cudaFree(gpu_out));
}

}  // namespace testing
}  // namespace dali
