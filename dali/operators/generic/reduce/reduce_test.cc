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
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

// Declared by DALI_SCHEMA macros in reduce.cc for parent-only schemas
int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_ReduceBase();
int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_ReduceWithOutputType();
int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_ReduceWithMeanInput();

namespace testing {

TEST(ReduceSchemaTest, ReduceBaseSchemaRegistered) {
  EXPECT_EQ(DALI_OPERATOR_SCHEMA_REQUIRED_FOR_ReduceBase(), 42);
  EXPECT_NO_THROW(SchemaRegistry::GetSchema("ReduceBase"));
}

TEST(ReduceSchemaTest, ReduceWithOutputTypeSchemaRegistered) {
  EXPECT_EQ(DALI_OPERATOR_SCHEMA_REQUIRED_FOR_ReduceWithOutputType(), 42);
  EXPECT_NO_THROW(SchemaRegistry::GetSchema("ReduceWithOutputType"));
}

TEST(ReduceSchemaTest, ReduceWithMeanInputSchemaRegistered) {
  EXPECT_EQ(DALI_OPERATOR_SCHEMA_REQUIRED_FOR_ReduceWithMeanInput(), 42);
  EXPECT_NO_THROW(SchemaRegistry::GetSchema("ReduceWithMeanInput"));
}

}  // namespace testing
}  // namespace dali
