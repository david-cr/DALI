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
#include <cuda_runtime.h>
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_stream.h"

namespace dali {

TEST(CUDAStream, CreateBlocking) {
  CUDAStream stream = CUDAStream::Create(false, 0);
  ASSERT_TRUE(stream);

  unsigned int flags = cudaStreamNonBlocking;
  CUDA_CALL(cudaStreamGetFlags(stream, &flags));
  EXPECT_EQ(flags, cudaStreamDefault);
  EXPECT_EQ(DeviceFromStream(stream), 0);
  CUDA_CALL(cudaStreamSynchronize(stream));
}

TEST(CUDAStream, CreateWithPriority) {
  int least_priority = 0, greatest_priority = 0;
  CUDA_CALL(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

  CUDAStream high = CUDAStream::CreateWithPriority(true, greatest_priority, 0);
  ASSERT_TRUE(high);
  int priority = 0;
  CUDA_CALL(cudaStreamGetPriority(high, &priority));
  EXPECT_EQ(priority, greatest_priority);
  EXPECT_EQ(DeviceFromStream(high), 0);

  CUDAStream low = CUDAStream::CreateWithPriority(false, least_priority, 0);
  ASSERT_TRUE(low);
  CUDA_CALL(cudaStreamGetPriority(low, &priority));
  EXPECT_EQ(priority, least_priority);

  CUDA_CALL(cudaStreamSynchronize(high));
  CUDA_CALL(cudaStreamSynchronize(low));
}

}  // namespace dali
