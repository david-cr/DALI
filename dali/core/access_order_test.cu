// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>
#include "dali/core/access_order.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_error.h"

namespace dali {

TEST(AccessOrder, ProperStream_MultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }

  CUDAStream s0 = CUDAStream::Create(true, 0);
  CUDAStream s1 = CUDAStream::Create(true, 1);
  AccessOrder o0(s0, 0);
  AccessOrder o1(s1, 1);
  EXPECT_NO_THROW(o0.wait(o1));
  EXPECT_NO_THROW(o1.wait(o0));
}

TEST(AccessOrder, DefaultStream_MultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }

  CUDAStream s0 = CUDAStream::Create(true, 0);
  CUDAStream s1 = CUDAStream::Create(true, 1);
  AccessOrder named_dev0(s0, 0);
  AccessOrder named_dev1(s1, 1);
  AccessOrder default_dev0(0, 0);
  AccessOrder default_dev1(0, 1);
  EXPECT_NO_THROW(named_dev0.wait(default_dev0));
  EXPECT_NO_THROW(named_dev1.wait(default_dev1));
  EXPECT_NO_THROW(named_dev0.wait(default_dev1));
  EXPECT_NO_THROW(named_dev1.wait(default_dev0));

  EXPECT_NO_THROW(default_dev0.wait(named_dev0));
  EXPECT_NO_THROW(default_dev0.wait(named_dev1));
  EXPECT_NO_THROW(default_dev1.wait(named_dev0));
  EXPECT_NO_THROW(default_dev1.wait(named_dev1));

  EXPECT_NO_THROW(default_dev0.wait(default_dev0));
  EXPECT_NO_THROW(default_dev0.wait(default_dev1));
  EXPECT_NO_THROW(default_dev1.wait(default_dev0));
  EXPECT_NO_THROW(default_dev1.wait(default_dev1));
}

TEST(AccessOrder, ErrorWaitForEventOnNullOrder) {
  CUDAEvent event = CUDAEvent::Create(0);
  AccessOrder null_order;
  ASSERT_FALSE(null_order.has_value());
  EXPECT_THROW(null_order.wait(event), std::logic_error);
}

TEST(AccessOrder, WaitForEventOnDefaultStream) {
  CUDAEvent event = CUDAEvent::Create(0);
  CUDAStream stream = CUDAStream::Create(true, 0);
  CUDA_CALL(cudaEventRecord(event, stream));

  // The default stream handle doesn't identify a device on its own, so waiting on it has
  // to make the order's device current first.
  AccessOrder default_dev0(0, 0);
  EXPECT_NO_THROW(default_dev0.wait(event));

  AccessOrder named_dev0(stream, 0);
  EXPECT_NO_THROW(named_dev0.wait(event));
  CUDA_CALL(cudaStreamSynchronize(stream));
}

}  // namespace dali
