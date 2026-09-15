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
#include "dali/core/dynlink_cuda.h"

namespace dali {

TEST(DynlinkCuda, SymbolAvailable) {
  ASSERT_TRUE(cuInitChecked());
  // The first query resolves the symbol, the second one is answered from the cache.
  EXPECT_TRUE(cuIsSymbolAvailable("cuInit"));
  EXPECT_TRUE(cuIsSymbolAvailable("cuInit"));
}

TEST(DynlinkCuda, SymbolNotAvailable) {
  ASSERT_TRUE(cuInitChecked());
  const char kMissing[] = "cuNoSuchSymbolInTheDriver";
  EXPECT_FALSE(cuIsSymbolAvailable(kMissing));
  EXPECT_FALSE(cuIsSymbolAvailable(kMissing));
}

}  // namespace dali
