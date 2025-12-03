// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "dali/pipeline/dali.pb.h"
#include "dali/pipeline/proto/dali_proto_intern.h"

namespace dali {

class DaliProtoPrivTest : public ::testing::Test {
 protected:
  void SetUp() override {
    arg_ = new dali_proto::Argument();
  }

  void TearDown() override {
    delete arg_;
  }

  dali_proto::Argument *arg_;
};

// ===== Constructor Tests =====

// Test constructor from pointer (covers line 22-23)
TEST_F(DaliProtoPrivTest, ConstructorFromPointer) {
  DaliProtoPriv priv(arg_);
  priv.set_name("test_name");
  EXPECT_EQ(arg_->name(), "test_name");
}

// Test constructor from const pointer (covers line 26-27)
TEST_F(DaliProtoPrivTest, ConstructorFromConstPointer) {
  const dali_proto::Argument *const_arg = arg_;
  DaliProtoPriv priv(const_arg);
  priv.set_name("test_name");
  EXPECT_EQ(arg_->name(), "test_name");
}

// Test copy-like constructor from DaliProtoPriv pointer (covers lines 30-31)
TEST_F(DaliProtoPrivTest, ConstructorFromDaliProtoPrivPointer) {
  DaliProtoPriv priv1(arg_);
  priv1.set_name("original_name");

  // Create another DaliProtoPriv from the first one's pointer
  DaliProtoPriv priv2(&priv1);

  // Both should point to the same underlying argument
  EXPECT_EQ(priv2.name(), "original_name");

  // Modifying through priv2 should affect priv1
  priv2.set_name("modified_name");
  EXPECT_EQ(priv1.name(), "modified_name");
  EXPECT_EQ(arg_->name(), "modified_name");
}

// ===== Setters Tests =====

TEST_F(DaliProtoPrivTest, SetName) {
  DaliProtoPriv priv(arg_);
  priv.set_name("test_arg");
  EXPECT_EQ(priv.name(), "test_arg");
}

TEST_F(DaliProtoPrivTest, SetType) {
  DaliProtoPriv priv(arg_);
  priv.set_type("int32");
  EXPECT_EQ(priv.type(), "int32");
}

TEST_F(DaliProtoPrivTest, SetIsVector) {
  DaliProtoPriv priv(arg_);
  priv.set_is_vector(true);
  EXPECT_TRUE(priv.is_vector());
  priv.set_is_vector(false);
  EXPECT_FALSE(priv.is_vector());
}

// ===== Integer Methods Tests =====

TEST_F(DaliProtoPrivTest, AddAndGetIntsIndexed) {
  DaliProtoPriv priv(arg_);
  priv.add_ints(42);
  priv.add_ints(100);
  priv.add_ints(-5);

  EXPECT_EQ(priv.ints(0), 42);
  EXPECT_EQ(priv.ints(1), 100);
  EXPECT_EQ(priv.ints(2), -5);
}

// Test vector getter for ints (covers lines 78-80)
TEST_F(DaliProtoPrivTest, GetIntsVector) {
  DaliProtoPriv priv(arg_);
  priv.add_ints(10);
  priv.add_ints(20);
  priv.add_ints(30);

  std::vector<int64_t> vec = priv.ints();
  ASSERT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0], 10);
  EXPECT_EQ(vec[1], 20);
  EXPECT_EQ(vec[2], 30);
}

// ===== Float Methods Tests =====

TEST_F(DaliProtoPrivTest, AddAndGetFloatsIndexed) {
  DaliProtoPriv priv(arg_);
  priv.add_floats(3.14f);
  priv.add_floats(2.71f);

  EXPECT_FLOAT_EQ(priv.floats(0), 3.14f);
  EXPECT_FLOAT_EQ(priv.floats(1), 2.71f);
}

// Test vector getter for floats (covers lines 87-89)
TEST_F(DaliProtoPrivTest, GetFloatsVector) {
  DaliProtoPriv priv(arg_);
  priv.add_floats(1.1f);
  priv.add_floats(2.2f);
  priv.add_floats(3.3f);

  std::vector<float> vec = priv.floats();
  ASSERT_EQ(vec.size(), 3);
  EXPECT_FLOAT_EQ(vec[0], 1.1f);
  EXPECT_FLOAT_EQ(vec[1], 2.2f);
  EXPECT_FLOAT_EQ(vec[2], 3.3f);
}

// ===== Bool Methods Tests =====

TEST_F(DaliProtoPrivTest, AddAndGetBoolsIndexed) {
  DaliProtoPriv priv(arg_);
  priv.add_bools(true);
  priv.add_bools(false);
  priv.add_bools(true);

  EXPECT_TRUE(priv.bools(0));
  EXPECT_FALSE(priv.bools(1));
  EXPECT_TRUE(priv.bools(2));
}

// Test vector getter for bools (covers lines 96-98)
TEST_F(DaliProtoPrivTest, GetBoolsVector) {
  DaliProtoPriv priv(arg_);
  priv.add_bools(true);
  priv.add_bools(false);
  priv.add_bools(true);
  priv.add_bools(false);

  std::vector<bool> vec = priv.bools();
  ASSERT_EQ(vec.size(), 4);
  EXPECT_TRUE(vec[0]);
  EXPECT_FALSE(vec[1]);
  EXPECT_TRUE(vec[2]);
  EXPECT_FALSE(vec[3]);
}

// ===== String Methods Tests =====

TEST_F(DaliProtoPrivTest, AddAndGetStringsIndexed) {
  DaliProtoPriv priv(arg_);
  priv.add_strings("first");
  priv.add_strings("second");
  priv.add_strings("third");

  EXPECT_EQ(priv.strings(0), "first");
  EXPECT_EQ(priv.strings(1), "second");
  EXPECT_EQ(priv.strings(2), "third");
}

// Test vector getter for strings (covers lines 105-107)
TEST_F(DaliProtoPrivTest, GetStringsVector) {
  DaliProtoPriv priv(arg_);
  priv.add_strings("alpha");
  priv.add_strings("beta");
  priv.add_strings("gamma");

  std::vector<std::string> vec = priv.strings();
  ASSERT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0], "alpha");
  EXPECT_EQ(vec[1], "beta");
  EXPECT_EQ(vec[2], "gamma");
}

// ===== Extra Args Tests =====

// Test add_extra_args (covers lines 61-63)
TEST_F(DaliProtoPrivTest, AddExtraArgs) {
  DaliProtoPriv priv(arg_);

  // Add an extra argument
  DaliProtoPriv extra1 = priv.add_extra_args();
  extra1.set_name("extra_arg_1");
  extra1.add_ints(42);

  // Add another extra argument
  DaliProtoPriv extra2 = priv.add_extra_args();
  extra2.set_name("extra_arg_2");
  extra2.add_floats(3.14f);

  // Verify through the underlying protobuf
  ASSERT_EQ(arg_->extra_args_size(), 2);
  EXPECT_EQ(arg_->extra_args(0).name(), "extra_arg_1");
  EXPECT_EQ(arg_->extra_args(1).name(), "extra_arg_2");
}

// Test extra_args(int index) (covers lines 122-124)
TEST_F(DaliProtoPrivTest, GetExtraArgsByIndex) {
  DaliProtoPriv priv(arg_);

  // Add extra arguments
  DaliProtoPriv extra1 = priv.add_extra_args();
  extra1.set_name("arg1");
  extra1.add_ints(100);

  DaliProtoPriv extra2 = priv.add_extra_args();
  extra2.set_name("arg2");
  extra2.add_strings("test_string");

  // Retrieve by index
  DaliProtoPriv retrieved1 = priv.extra_args(0);
  EXPECT_EQ(retrieved1.name(), "arg1");
  EXPECT_EQ(retrieved1.ints(0), 100);

  DaliProtoPriv retrieved2 = priv.extra_args(1);
  EXPECT_EQ(retrieved2.name(), "arg2");
  EXPECT_EQ(retrieved2.strings(0), "test_string");
}

// Test extra_args() vector getter (covers lines 114-119)
TEST_F(DaliProtoPrivTest, GetExtraArgsVector) {
  DaliProtoPriv priv(arg_);

  // Add multiple extra arguments
  DaliProtoPriv extra1 = priv.add_extra_args();
  extra1.set_name("first_extra");
  extra1.add_ints(10);

  DaliProtoPriv extra2 = priv.add_extra_args();
  extra2.set_name("second_extra");
  extra2.add_floats(2.5f);

  DaliProtoPriv extra3 = priv.add_extra_args();
  extra3.set_name("third_extra");
  extra3.add_bools(true);

  // Get all extra args as a vector
  std::vector<DaliProtoPriv> extras = priv.extra_args();

  ASSERT_EQ(extras.size(), 3);
  EXPECT_EQ(extras[0].name(), "first_extra");
  EXPECT_EQ(extras[0].ints(0), 10);

  EXPECT_EQ(extras[1].name(), "second_extra");
  EXPECT_FLOAT_EQ(extras[1].floats(0), 2.5f);

  EXPECT_EQ(extras[2].name(), "third_extra");
  EXPECT_TRUE(extras[2].bools(0));
}

// Test empty extra_args vector
TEST_F(DaliProtoPrivTest, GetExtraArgsVectorEmpty) {
  DaliProtoPriv priv(arg_);

  // Get extra args when none have been added
  std::vector<DaliProtoPriv> extras = priv.extra_args();
  EXPECT_EQ(extras.size(), 0);
}

// ===== Comprehensive Round-Trip Tests =====

TEST_F(DaliProtoPrivTest, ComprehensiveRoundTrip) {
  DaliProtoPriv priv(arg_);

  // Set basic properties
  priv.set_name("comprehensive_test");
  priv.set_type("mixed");
  priv.set_is_vector(true);

  // Add various types of data
  priv.add_ints(1);
  priv.add_ints(2);
  priv.add_ints(3);

  priv.add_floats(1.1f);
  priv.add_floats(2.2f);

  priv.add_bools(true);
  priv.add_bools(false);

  priv.add_strings("one");
  priv.add_strings("two");

  // Add extra args
  DaliProtoPriv extra = priv.add_extra_args();
  extra.set_name("nested_arg");
  extra.add_ints(999);

  // Verify all data
  EXPECT_EQ(priv.name(), "comprehensive_test");
  EXPECT_EQ(priv.type(), "mixed");
  EXPECT_TRUE(priv.is_vector());

  auto int_vec = priv.ints();
  EXPECT_EQ(int_vec.size(), 3);
  EXPECT_EQ(int_vec[0], 1);
  EXPECT_EQ(int_vec[1], 2);
  EXPECT_EQ(int_vec[2], 3);

  auto float_vec = priv.floats();
  EXPECT_EQ(float_vec.size(), 2);
  EXPECT_FLOAT_EQ(float_vec[0], 1.1f);
  EXPECT_FLOAT_EQ(float_vec[1], 2.2f);

  auto bool_vec = priv.bools();
  EXPECT_EQ(bool_vec.size(), 2);
  EXPECT_TRUE(bool_vec[0]);
  EXPECT_FALSE(bool_vec[1]);

  auto string_vec = priv.strings();
  EXPECT_EQ(string_vec.size(), 2);
  EXPECT_EQ(string_vec[0], "one");
  EXPECT_EQ(string_vec[1], "two");

  auto extras = priv.extra_args();
  EXPECT_EQ(extras.size(), 1);
  EXPECT_EQ(extras[0].name(), "nested_arg");
  EXPECT_EQ(extras[0].ints(0), 999);
}

}  // namespace dali



