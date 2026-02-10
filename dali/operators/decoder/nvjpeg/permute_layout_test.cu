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
#include <cstdint>
#include <vector>
#include "dali/operators/decoder/nvjpeg/permute_layout.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace testing {

class PermuteLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CALL(cudaStreamCreate(&stream_));
  }
  void TearDown() override {
    CUDA_CALL(cudaStreamDestroy(stream_));
  }
  cudaStream_t stream_ = nullptr;
};

// ============================================================================
// PlanarToInterleaved with comp_count == 1 (< 2) → cudaMemcpy path
// covers lines 73-76
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedSingleChannel) {
  const int64_t npixels = 4;
  std::vector<uint8_t> host_input = {10, 20, 30, 40};
  std::vector<uint8_t> host_output(npixels, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels));
  CUDA_CALL(cudaMalloc(&d_output, npixels));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels, cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, 1,
                                        DALI_RGB, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels, cudaMemcpyDeviceToHost));

  EXPECT_EQ(host_output, host_input);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved with comp_count == 2 → VALUE_SWITCH case 2
// covers line 82 case 2
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedTwoChannels) {
  const int64_t npixels = 4;
  const int64_t comp_count = 2;
  // Planar layout: [C0: p0 p1 p2 p3] [C1: p0 p1 p2 p3]
  std::vector<uint8_t> host_input = {10, 20, 30, 40,   // channel 0
                                     50, 60, 70, 80};  // channel 1
  std::vector<uint8_t> host_output(npixels * comp_count, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * comp_count));
  CUDA_CALL(cudaMalloc(&d_output, npixels * comp_count));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * comp_count, cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, comp_count,
                                        DALI_RGB, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels * comp_count, cudaMemcpyDeviceToHost));

  // Expected interleaved: [p0c0, p0c1, p1c0, p1c1, p2c0, p2c1, p3c0, p3c1]
  std::vector<uint8_t> expected = {10, 50, 20, 60, 30, 70, 40, 80};
  EXPECT_EQ(host_output, expected);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved with comp_count == 4 → VALUE_SWITCH case 4
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedFourChannels) {
  const int64_t npixels = 2;
  const int64_t comp_count = 4;
  // Planar: [C0: 1,2] [C1: 3,4] [C2: 5,6] [C3: 7,8]
  std::vector<uint8_t> host_input = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> host_output(npixels * comp_count, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * comp_count));
  CUDA_CALL(cudaMalloc(&d_output, npixels * comp_count));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * comp_count, cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, comp_count,
                                        DALI_ANY_DATA, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels * comp_count, cudaMemcpyDeviceToHost));

  // Expected: [1,3,5,7, 2,4,6,8]
  std::vector<uint8_t> expected = {1, 3, 5, 7, 2, 4, 6, 8};
  EXPECT_EQ(host_output, expected);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved with unsupported comp_count → DALI_FAIL
// covers line 85 default case
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedUnsupportedCompCount) {
  const int64_t npixels = 4;
  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * 5));
  CUDA_CALL(cudaMalloc(&d_output, npixels * 5));

  auto throw_fn = [&]() {
    PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, 5,
                                          DALI_RGB, DALI_UINT8, stream_);
  };
  EXPECT_THROW(throw_fn(), DALIException);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved with small npixels (< 1024) → covers line 79 true branch
// This is implicitly covered by tests above (npixels=4), but explicit here
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedSmallPixels) {
  const int64_t npixels = 8;
  const int64_t comp_count = 3;
  std::vector<uint8_t> host_input(npixels * comp_count);
  for (size_t i = 0; i < host_input.size(); i++) host_input[i] = i;
  std::vector<uint8_t> host_output(npixels * comp_count, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * comp_count));
  CUDA_CALL(cudaMalloc(&d_output, npixels * comp_count));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * comp_count, cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, comp_count,
                                        DALI_RGB, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels * comp_count, cudaMemcpyDeviceToHost));

  // Verify first pixel: planar [R0, R1,...R7, G0,...G7, B0,...B7] → [R0,G0,B0, R1,G1,B1, ...]
  EXPECT_EQ(host_output[0], 0);   // R0
  EXPECT_EQ(host_output[1], 8);   // G0
  EXPECT_EQ(host_output[2], 16);  // B0

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved BGR → covers line 87
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedBGR) {
  const int64_t npixels = 2;
  const int64_t comp_count = 3;
  // Planar RGB: [R:10,20] [G:30,40] [B:50,60]
  std::vector<uint8_t> host_input = {10, 20, 30, 40, 50, 60};
  std::vector<uint8_t> host_output(npixels * comp_count, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * comp_count));
  CUDA_CALL(cudaMalloc(&d_output, npixels * comp_count));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * comp_count, cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, comp_count,
                                        DALI_BGR, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels * comp_count, cudaMemcpyDeviceToHost));

  // BGR: [B0,G0,R0, B1,G1,R1] = [50,30,10, 60,40,20]
  EXPECT_EQ(host_output[0], 50);
  EXPECT_EQ(host_output[1], 30);
  EXPECT_EQ(host_output[2], 10);
  EXPECT_EQ(host_output[3], 60);
  EXPECT_EQ(host_output[4], 40);
  EXPECT_EQ(host_output[5], 20);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved YCbCr → covers line 88-89
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedYCbCr) {
  const int64_t npixels = 2;
  const int64_t comp_count = 3;
  std::vector<uint8_t> host_input = {128, 64, 200, 100, 50, 150};
  std::vector<uint8_t> host_output(npixels * comp_count, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * comp_count));
  CUDA_CALL(cudaMalloc(&d_output, npixels * comp_count));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * comp_count, cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint8_t>(d_output, d_input, npixels, comp_count,
                                        DALI_YCbCr, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels * comp_count, cudaMemcpyDeviceToHost));

  // Just verify it doesn't crash and produces non-zero output
  bool has_nonzero = false;
  for (auto v : host_output) if (v != 0) has_nonzero = true;
  EXPECT_TRUE(has_nonzero);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarToInterleaved with uint16_t input → covers template instantiation
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarToInterleavedUint16Input) {
  const int64_t npixels = 4;
  const int64_t comp_count = 3;
  std::vector<uint16_t> host_input(npixels * comp_count);
  for (size_t i = 0; i < host_input.size(); i++) host_input[i] = i * 256;
  std::vector<uint8_t> host_output(npixels * comp_count, 0);

  uint16_t *d_input = nullptr;
  uint8_t *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * comp_count * sizeof(uint16_t)));
  CUDA_CALL(cudaMalloc(&d_output, npixels * comp_count));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * comp_count * sizeof(uint16_t),
                        cudaMemcpyHostToDevice));

  PlanarToInterleaved<uint8_t, uint16_t>(d_output, d_input, npixels, comp_count,
                                         DALI_RGB, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels * comp_count,
                        cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

// ============================================================================
// PlanarRGBToGray with small npixels → covers line 100 true branch
// ============================================================================

TEST_F(PermuteLayoutTest, PlanarRGBToGraySmall) {
  const int64_t npixels = 4;
  // Planar RGB: [R: 255,0,0,128] [G: 0,255,0,128] [B: 0,0,255,128]
  std::vector<uint8_t> host_input = {255, 0, 0, 128,
                                     0, 255, 0, 128,
                                     0, 0, 255, 128};
  std::vector<uint8_t> host_output(npixels, 0);

  uint8_t *d_input = nullptr, *d_output = nullptr;
  CUDA_CALL(cudaMalloc(&d_input, npixels * 3));
  CUDA_CALL(cudaMalloc(&d_output, npixels));
  CUDA_CALL(cudaMemcpy(d_input, host_input.data(), npixels * 3, cudaMemcpyHostToDevice));

  PlanarRGBToGray<uint8_t, uint8_t>(d_output, d_input, npixels, DALI_UINT8, stream_);
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cudaMemcpy(host_output.data(), d_output, npixels, cudaMemcpyDeviceToHost));

  // Pure red → gray ~76 (0.299*255), pure green ~150 (0.587*255), pure blue ~29 (0.114*255)
  EXPECT_NEAR(host_output[0], 76, 2);
  EXPECT_NEAR(host_output[1], 150, 2);
  EXPECT_NEAR(host_output[2], 29, 2);
  // Gray 128 → ~128
  EXPECT_NEAR(host_output[3], 128, 2);

  CUDA_CALL(cudaFree(d_input));
  CUDA_CALL(cudaFree(d_output));
}

}  // namespace testing
}  // namespace dali
