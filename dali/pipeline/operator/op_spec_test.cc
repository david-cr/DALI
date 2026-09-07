// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operator/name_utils.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/test/dali_test.h"

using namespace std::string_literals;  // NOLINT(build/namespaces)

namespace dali {

DALI_SCHEMA(DummyGrandparentForSpecTest)
  .NumInput(0).NumOutput(0)
  .AddOptionalArg("grandparent_replacing_arg", "arg that replaces deprecated arg", 0)
  .DeprecateArgInFavorOf("grandparent_deprecated_arg", "grandparent_replacing_arg", "1.0");

DALI_SCHEMA(DummyParentZeroForSpecTest)
  .NumInput(0).NumOutput(0)
  .AddOptionalArg("parent_zero_replacing_arg", "arg that replaces deprecated arg", 0)
  .DeprecateArgInFavorOf("parent_zero_deprecated_arg", "parent_zero_replacing_arg", "1.0")
  .AddParent("DummyGrandparentForSpecTest");


DALI_SCHEMA(DummyParentOneForSpecTest)
  .NumInput(0).NumOutput(0)
  .AddOptionalArg("parent_one_replacing_arg", "arg that replaces deprecated arg", 0)
  .DeprecateArgInFavorOf("parent_one_deprecated_arg", "parent_one_replacing_arg", "1.0");


DALI_SCHEMA(DummyOpForSpecTest)
  .NumInput(0).NumOutput(0)
  .AddArg("required", "required argument", DALIDataType::DALI_INT32)
  .AddOptionalArg("default", "argument with default", 11)
  .AddOptionalArg<int>("no_default", "argument without default", nullptr)
  .AddArg("required_vec", "required argument", DALIDataType::DALI_INT_VEC)
  .AddOptionalArg("default_vec", "argument with default vec", std::vector<int32_t>{0, 1})
  .AddOptionalArg<std::vector<int>>("no_default_vec", "argument without default", nullptr)
  .AddArg("required_tensor", "required argument", DALIDataType::DALI_INT32, true)
  .AddOptionalArg("default_tensor", "argument with default", 11, true)
  .AddOptionalArg<int>("no_default_tensor", "argument without default", nullptr, true)
  .AddOptionalArg("replacing_arg", "arg that replaces deprecated arg", 0)
  .DeprecateArgInFavorOf("deprecated_arg", "replacing_arg", "1.0")
  .AddOptionalArg("deprecated_ignored_arg",
                  "arg that is deprecated and ignored by the implementation", 0)
  .DeprecateArg("deprecated_ignored_arg", "1.0")
  .AddParent("DummyParentZeroForSpecTest")
  .AddParent("DummyParentOneForSpecTest");

TEST(OpSpecTest, GetArgumentTensorSet) {
  // Check how required and optional arguments handle Argument Inputs
  // Should work only with [Try]GetArgument;
  // [Try]GetRepeatedArgument does not handle Argument Inputs
  for (const auto &arg_name : {"required_tensor"s, "default_tensor"s, "no_default_tensor"s}) {
    ArgumentWorkspace ws0;
    auto tv = std::make_shared<TensorList<CPUBackend>>(2);
    tv->Resize(TensorListShape<0>(2), DALI_INT32);
    for (int i = 0; i < 2; i++) {
      tv->mutable_tensor<int32_t>(i)[0] = 42 + i;
    }
    ws0.AddArgumentInput(arg_name, tv);
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2)
        .AddArgumentInput(arg_name, "<not_used>");
    EXPECT_EQ(spec0.GetArgument<int32_t>(arg_name, &ws0, 0), 42);
    EXPECT_EQ(spec0.GetArgument<int32_t>(arg_name, &ws0, 1), 43);
    int result = 0;
    ASSERT_TRUE(spec0.TryGetArgument<int32_t>(result, arg_name, &ws0, 0));
    EXPECT_EQ(result, 42);
    ASSERT_TRUE(spec0.TryGetArgument<int32_t>(result, arg_name, &ws0, 1));
    EXPECT_EQ(result, 43);
    EXPECT_THROW(spec0.GetArgument<float>(arg_name, &ws0, 0), std::runtime_error);
    float tmp = 0.f;
    EXPECT_FALSE(spec0.TryGetArgument<float>(tmp, arg_name, &ws0, 0));

    ArgumentWorkspace ws1;
    auto spec1 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2);
    // If we have a default optional argument, we will just return its value
    if (arg_name != "default_tensor"s) {
      EXPECT_THROW(spec1.GetArgument<int>(arg_name, &ws1, 0), std::invalid_argument);
      EXPECT_THROW(spec1.GetArgument<int>(arg_name, &ws1, 1), std::invalid_argument);
      int result = 0;
      EXPECT_FALSE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 0));
      EXPECT_FALSE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 1));
    } else {
      EXPECT_EQ(spec1.GetArgument<int>(arg_name, &ws1, 0), 11);
      EXPECT_EQ(spec1.GetArgument<int>(arg_name, &ws1, 1), 11);
      int result = 0;
      EXPECT_TRUE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 0));
      EXPECT_EQ(result, 11);
      result = 0;
      EXPECT_TRUE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 1));
      EXPECT_EQ(result, 11);
    }
  }
}

TEST(OpSpecTest, GetArgumentValue) {
  for (const auto &arg_name : {"required"s, "default"s, "no_default"s,
                               "required_tensor"s, "default_tensor"s, "no_default_tensor"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2)
        .AddArg(arg_name, 42);
    EXPECT_EQ(spec0.GetArgument<int>(arg_name, &ws), 42);
    int result = 0;
    ASSERT_TRUE(spec0.TryGetArgument(result, arg_name, &ws));
    EXPECT_EQ(result, 42);

    EXPECT_THROW(spec0.GetArgument<float>(arg_name, &ws), std::runtime_error);
    float tmp = 0.f;
    EXPECT_FALSE(spec0.TryGetArgument(tmp, arg_name, &ws));
  }

  for (const auto &arg_name : {"required"s, "no_default"s,
                               "required_tensor"s, "no_default_tensor"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2);
    EXPECT_THROW(spec0.GetArgument<int>(arg_name, &ws), std::invalid_argument);
    int result = 0;
    EXPECT_FALSE(spec0.TryGetArgument(result, arg_name, &ws));

    EXPECT_THROW(spec0.GetArgument<float>(arg_name, &ws), std::invalid_argument);
    float tmp = 0.f;
    EXPECT_FALSE(spec0.TryGetArgument(tmp, arg_name, &ws));
  }

  for (const auto &arg_name : {"default"s, "default_tensor"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2);
    EXPECT_EQ(spec0.GetArgument<int>(arg_name, &ws), 11);

    int result = 0;
    ASSERT_TRUE(spec0.TryGetArgument(result, arg_name, &ws));
    EXPECT_EQ(result, 11);

    EXPECT_THROW(spec0.GetArgument<float>(arg_name, &ws), std::invalid_argument);
    float tmp = 0.f;
    EXPECT_FALSE(spec0.TryGetArgument(tmp, arg_name, &ws));
  }
}

TEST(OpSpecTest, GetArgumentVec) {
  for (const auto &arg_name : {"required_vec"s, "default_vec"s, "no_default_vec"s}) {
    ArgumentWorkspace ws;
    auto value = std::vector<int32_t>{42, 43};

    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2)
        .AddArg(arg_name, value);

    EXPECT_EQ(spec0.GetRepeatedArgument<int32_t>(arg_name), value);
    std::vector<int32_t> result;
    ASSERT_TRUE(spec0.TryGetRepeatedArgument(result, arg_name));
    EXPECT_EQ(result, value);
  }

  for (const auto &arg_name : {"required_vec"s, "no_default_vec"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2);

    EXPECT_THROW(spec0.GetRepeatedArgument<int32_t>(arg_name), std::invalid_argument);
    std::vector<int32_t> result_v;
    ASSERT_FALSE(spec0.TryGetRepeatedArgument(result_v, arg_name));
    SmallVector<int32_t, 1> result_sv;
    EXPECT_FALSE(spec0.TryGetRepeatedArgument(result_sv, arg_name));

    EXPECT_THROW(spec0.GetRepeatedArgument<float>(arg_name), std::invalid_argument);
    std::vector<float> tmp_v;
    EXPECT_FALSE(spec0.TryGetRepeatedArgument(tmp_v, arg_name));
    SmallVector<float, 1> tmp_sv;
    EXPECT_FALSE(spec0.TryGetRepeatedArgument(tmp_sv, arg_name));
  }

  {
    auto arg_name = "default_vec"s;
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("max_batch_size", 2);
    auto default_val = std::vector<int32_t>{0, 1};
    EXPECT_EQ(spec0.GetRepeatedArgument<int32_t>(arg_name), default_val);
  }
}


TEST(OpSpecTest, GetArgumentNonExisting) {
  auto spec0 = OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2);
  EXPECT_THROW(spec0.GetArgument<int>("<no_such_argument>"), invalid_key);
  int result = 0;
  EXPECT_FALSE(spec0.TryGetArgument<int>(result, "<no_such_argument>"));


  EXPECT_THROW(spec0.GetRepeatedArgument<int>("<no_such_argument>"), invalid_key);
  std::vector<int> result_vec;
  EXPECT_FALSE(spec0.TryGetRepeatedArgument(result_vec, "<no_such_argument>"));
  SmallVector<int, 1> result_sv;
  EXPECT_FALSE(spec0.TryGetRepeatedArgument(result_sv, "<no_such_argument>"));
}

TEST(OpSpecTest, DeprecatedArgs) {
  auto spec0 = OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("deprecated_arg", 1);
  EXPECT_THROW(spec0.GetArgument<int>("deprecated_arg"), std::invalid_argument);
  EXPECT_EQ(spec0.GetArgument<int>("replacing_arg"), 1);

  int result = 0;
  EXPECT_FALSE(spec0.TryGetArgument<int>(result, "deprecated_arg"));
  ASSERT_TRUE(spec0.TryGetArgument<int>(result, "replacing_arg"));
  EXPECT_EQ(result, 1);

  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("deprecated_arg", 1)
      .AddArg("replacing_arg", 2), DALIException);

  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("replacing_arg", 1)
      .AddArg("deprecated_arg", 2), DALIException);

  auto spec1 = OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("deprecated_ignored_arg", 42);
  // It is marked as to be ingored, but there's no reason we should not be
  // able to query for the argument if it was provided.
  EXPECT_TRUE(spec0.TryGetArgument<int>(result, "deprecated_ignored_arg"));
}

TEST(OpSpecTest, DeprecatedArgsParents) {
  auto spec0 = OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("grandparent_deprecated_arg", 3)
      .AddArg("parent_zero_deprecated_arg", 4)
      .AddArg("parent_one_deprecated_arg", 5);
  EXPECT_THROW(spec0.GetArgument<int>("grandparent_deprecated_arg"), std::invalid_argument);
  EXPECT_THROW(spec0.GetArgument<int>("parent_zero_deprecated_arg"), std::invalid_argument);
  EXPECT_THROW(spec0.GetArgument<int>("parent_one_deprecated_arg"), std::invalid_argument);
  EXPECT_EQ(spec0.GetArgument<int>("grandparent_replacing_arg"), 3);
  EXPECT_EQ(spec0.GetArgument<int>("parent_zero_replacing_arg"), 4);
  EXPECT_EQ(spec0.GetArgument<int>("parent_one_replacing_arg"), 5);


  int result = 0;
  EXPECT_FALSE(spec0.TryGetArgument<int>(result, "grandparent_deprecated_arg"));
  ASSERT_TRUE(spec0.TryGetArgument<int>(result, "grandparent_replacing_arg"));
  EXPECT_EQ(result, 3);

  EXPECT_FALSE(spec0.TryGetArgument<int>(result, "parent_zero_deprecated_arg"));
  ASSERT_TRUE(spec0.TryGetArgument<int>(result, "parent_zero_replacing_arg"));
  EXPECT_EQ(result, 4);

  EXPECT_FALSE(spec0.TryGetArgument<int>(result, "parent_one_deprecated_arg"));
  ASSERT_TRUE(spec0.TryGetArgument<int>(result, "parent_one_replacing_arg"));
  EXPECT_EQ(result, 5);

  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("grandparent_deprecated_arg", 1)
      .AddArg("grandparent_replacing_arg", 2), DALIException);

  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("grandparent_replacing_arg", 1)
      .AddArg("grandparent_deprecated_arg", 2), DALIException);


  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("parent_zero_deprecated_arg", 1)
      .AddArg("parent_zero_replacing_arg", 2), DALIException);

  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("parent_zero_replacing_arg", 1)
      .AddArg("parent_zero_deprecated_arg", 2), DALIException);


  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("parent_one_deprecated_arg", 1)
      .AddArg("parent_one_replacing_arg", 2), DALIException);

  EXPECT_THROW(OpSpec("DummyOpForSpecTest")
      .AddArg("max_batch_size", 2)
      .AddArg("parent_one_replacing_arg", 1)
      .AddArg("parent_one_deprecated_arg", 2), DALIException);
}

class TestArgumentInput_Producer : public Operator<CPUBackend> {
 public:
  explicit TestArgumentInput_Producer(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(3);
    output_desc[0] = {TensorListShape<0>(ws.GetRequestedBatchSize(0)),         DALI_INT32};
    // Non-matching shapes
    output_desc[1] = {uniform_list_shape(ws.GetRequestedBatchSize(1), {1}),    DALI_FLOAT};
    output_desc[2] = {uniform_list_shape(ws.GetRequestedBatchSize(2), {1, 2}), DALI_INT32};
    return true;
  }

  void RunImpl(Workspace &ws) override {
    // Initialize all the data with a 0, 1, 2 .... sequence
    auto &out0 = ws.Output<CPUBackend>(0);
    for (int i = 0; i < out0.shape().num_samples(); i++) {
      *out0.mutable_tensor<int>(i) = i;
    }

    auto &out1 = ws.Output<CPUBackend>(1);
    for (int i = 0; i < out1.shape().num_samples(); i++) {
      *out1.mutable_tensor<float>(i) = i;
    }

    auto &out2 = ws.Output<CPUBackend>(2);
    for (int i = 0; i < out2.shape().num_samples(); i++) {
      for (int j = 0; j < 2; j++) {
        out2.mutable_tensor<int>(i)[j] = i;
      }
    }
  }
};

DALI_REGISTER_OPERATOR(TestArgumentInput_Producer, TestArgumentInput_Producer, CPU);

DALI_SCHEMA(TestArgumentInput_Producer)
    .DocStr("TestArgumentInput_Producer")
    .NumInput(0)
    .NumOutput(3);

class TestArgumentInput_Consumer : public Operator<CPUBackend> {
 public:
  explicit TestArgumentInput_Consumer(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    output_desc[0] = {uniform_list_shape(ws.GetRequestedBatchSize(0), {1}), DALI_INT32};
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto curr_batch_size =
        ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
    for (int i = 0; i < curr_batch_size; i++) {
      EXPECT_EQ(spec_.GetArgument<int>("arg0", &ws, i), i);
    }
    // Non-matching shapes (differnet than 1 scalar value per sample) should not work with
    // OpSpec::GetArgument()
    EXPECT_THROW(auto z = spec_.GetArgument<float>("arg2", &ws, 0), std::runtime_error);

    // They can be accessed as proper ArgumentInputs
    auto &ref_1 = ws.ArgumentInput("arg1");
    ASSERT_EQ(ref_1.shape().num_samples(), curr_batch_size);
    ASSERT_TRUE(is_uniform(ref_1.shape()));
    ASSERT_EQ(ref_1.shape()[0], TensorShape<>(1));
    for (int i = 0; i < ref_1.shape().num_samples(); i++) {
      EXPECT_EQ(ref_1.tensor<float>(i)[0], i);
    }

    auto &ref_2 = ws.ArgumentInput("arg2");
    ASSERT_EQ(ref_2.shape().num_samples(), curr_batch_size);
    ASSERT_TRUE(is_uniform(ref_2.shape()));
    ASSERT_EQ(ref_2.shape()[0], TensorShape<>(1, 2));
    for (int i = 0; i < ref_2.shape().num_samples(); i++) {
      for (int j = 0; j < 2; j++) {
        EXPECT_EQ(ref_2.tensor<int>(i)[j], i);
      }
    }
  }
};

DALI_REGISTER_OPERATOR(TestArgumentInput_Consumer, TestArgumentInput_Consumer, CPU);

DALI_SCHEMA(TestArgumentInput_Consumer)
    .DocStr("TestArgumentInput_Consumer")
    .NumInput(0)
    .NumOutput(1)
    .AddOptionalArg("arg0", "no-doc", 42, true)
    .AddOptionalArg("arg1", "no-doc", 42.f, true)
    .AddOptionalArg("arg2", "no-doc", 42, true)
    .AddOptionalArg("arg3", "no-doc", 42, true);

/*
 * This test is based on test operators implemented specifically for the purpose of testing
 * the access to argument inputs.
 *
 * The EXPECT_* and ASSERT_* macros are actually placed in the RunImpl of operator
 * accessing the data (TestArgumentInput_Consumer), and the different (valid and invalid)
 * arguments inputs are provided by a Operator: TestArgumentInput_Producer.
 */
TEST(ArgumentInputTest, OpSpecAccess) {
  Pipeline pipe(10, 4, 0);
  pipe.AddOperator(OpSpec("TestArgumentInput_Producer")
                       .AddArg("device", "cpu")
                       .AddOutput("support_arg0", StorageDevice::CPU)
                       .AddOutput("support_arg1", StorageDevice::CPU)
                       .AddOutput("support_arg2", StorageDevice::CPU));

  pipe.AddOperator(OpSpec("TestArgumentInput_Consumer")
                       .AddArg("device", "cpu")
                       .AddArgumentInput("arg0", "support_arg0")
                       .AddArgumentInput("arg1", "support_arg1")
                       .AddArgumentInput("arg2", "support_arg2")
                       .AddOutput("I need to specify something", StorageDevice::CPU)
                       .AddArg("preserve", true));

  vector<std::pair<string, string>> outputs = {{"I need to specify something", "cpu"}};
  pipe.Build(outputs);

  pipe.Run();

  Workspace ws;
  pipe.Outputs(&ws);
}

DALI_SCHEMA(Schema_TestOpSpec_Lookup)
    .DocStr("TestOpSpec_Lookup")
    .NumInput(1)
    .NumOutput(1)
    // the names of the arguments are deliberately in a reverse order (lexicographically)
    .AddOptionalArg<int>("zero", "dummy arg that can be an argument input", 0, true)
    .AddOptionalArg<int>("one",  "dummy arg that can be an argument input", 0, true);

TEST(TestOpSpec, Lookup) {
  OpSpec spec("Schema_TestOpSpec_Lookup");
  spec.AddInput("input_0", StorageDevice::GPU);
  spec.AddArgumentInput("one", "input_1");
  spec.AddArgumentInput("zero", "input_2");
  EXPECT_EQ(spec.ArgumentInputIdx("one"), 1);
  EXPECT_EQ(spec.ArgumentInputIdx("zero"), 2);

  EXPECT_EQ(spec.InputName(0), "input_0");
  EXPECT_EQ(spec.InputName(1), "input_1");
  EXPECT_EQ(spec.ArgumentInputName(1), "one");
  EXPECT_EQ(spec.InputName(2), "input_2");
  EXPECT_EQ(spec.ArgumentInputName(2), "zero");
}

TEST(TestOpSpec, EmptySchema) {
  OpSpec spec("nonexistent_schema");
  EXPECT_THROW(spec.GetSchema(), std::runtime_error);
  EXPECT_EQ(spec.GetArgument<std::string>("device"), "cpu");
  EXPECT_EQ(spec.GetArgument<std::string>("_module"), "nvidia.dali.ops");
}

// ===== AddInput Error Path Tests =====

DALI_SCHEMA(DummyInputErrorTests)
  .NumInput(2).NumOutput(1)
  .AddOptionalArg<int>("tensor_arg", "A tensor argument", nullptr, true);

// Test AddInput: regular input added after argument input (covers line 63)
TEST(OpSpecTest, AddInputAfterArgumentInput) {
  auto spec = OpSpec("DummyInputErrorTests")
              .AddArgumentInput("tensor_arg", "arg_input");

  // Try to add regular input after argument input - should throw
  EXPECT_THROW(
    spec.AddInput("regular_input", StorageDevice::CPU),
    std::runtime_error);
}

DALI_SCHEMA(DummyTwoInputs)
  .NumInput(2).NumOutput(1);

// Test AddInput: too many inputs (covers line 69)
TEST(OpSpecTest, AddInputTooMany) {
  auto spec = OpSpec("DummyTwoInputs")
              .AddInput("input1", StorageDevice::CPU)
              .AddInput("input2", StorageDevice::CPU);

  // Try to add third input when schema only allows 2 - should throw
  EXPECT_THROW(
    spec.AddInput("input3", StorageDevice::CPU),
    std::runtime_error);
}

DALI_SCHEMA(DummyInputDeviceCheck)
  .NumInput(1).NumOutput(1)
  .InputDevice(0, InputDevice::CPU);

// Test AddInput: incompatible device (covers line 77)
TEST(OpSpecTest, AddInputIncompatibleDevice) {
  auto spec = OpSpec("DummyInputDeviceCheck")
              .AddArg("device", "cpu");

  // Try to add GPU input when schema requires CPU - should throw
  EXPECT_THROW(
    spec.AddInput("input", StorageDevice::GPU),
    std::runtime_error);
}

// Test that argument inputs are always CPU: AddArgumentInput takes no device,
// so a GPU argument input cannot be expressed, while regular inputs keep theirs.
TEST(OpSpecTest, ArgumentInputIsAlwaysCPU) {
  auto spec = OpSpec("DummyInputErrorTests")
              .AddInput("input", StorageDevice::GPU)
              .AddArgumentInput("tensor_arg", "arg_input");

  ASSERT_EQ(spec.NumRegularInput(), 1);
  ASSERT_EQ(spec.NumArgumentInput(), 1);

  int arg_idx = spec.NumRegularInput();
  EXPECT_TRUE(spec.IsArgumentInput(arg_idx));
  EXPECT_EQ(spec.InputDevice(arg_idx), StorageDevice::CPU);
  EXPECT_EQ(spec.InputDevice(0), StorageDevice::GPU);
}

// ===== AddOutput Error Path Tests =====

// Test AddOutput: duplicate output (covers line 97)
TEST(OpSpecTest, AddOutputDuplicate) {
  auto spec = OpSpec("DummyInputErrorTests")
              .AddOutput("output1", StorageDevice::CPU);

  // Try to add the same output name and device again - should throw
  EXPECT_THROW(
    spec.AddOutput("output1", StorageDevice::CPU),
    std::invalid_argument);
}

// ===== AddArgumentInput Error Path Tests =====

DALI_SCHEMA(DummyArgInputTests)
  .NumInput(0).NumOutput(1)
  .AddOptionalArg("regular_arg", "Not a tensor argument", 42)
  .AddOptionalArg<int>("tensor_arg", "A tensor argument", nullptr, true);

// Test AddArgumentInput: argument already specified (covers line 105)
TEST(OpSpecTest, AddArgumentInputAlreadySpecified) {
  auto spec = OpSpec("DummyArgInputTests")
              .AddArg("regular_arg", 100);

  // Try to add argument input for already-specified argument - should throw
  EXPECT_THROW(
    spec.AddArgumentInput("regular_arg", "input"),
    std::runtime_error);
}

// Test AddArgumentInput: undefined argument (covers line 108)
TEST(OpSpecTest, AddArgumentInputUndefined) {
  auto spec = OpSpec("DummyArgInputTests");

  // Try to add argument input for non-existent argument - should throw
  EXPECT_THROW(
    spec.AddArgumentInput("non_existent", "input"),
    std::runtime_error);
}

// Test AddArgumentInput: not a tensor argument (covers line 111)
TEST(OpSpecTest, AddArgumentInputNotTensor) {
  auto spec = OpSpec("DummyArgInputTests");

  // Try to add argument input for non-tensor argument - should throw
  EXPECT_THROW(
    spec.AddArgumentInput("regular_arg", "input"),
    std::runtime_error);
}

// ===== IsCompatibleDevice Coverage =====

DALI_SCHEMA(DummyDeviceTestCPU)
  .NumInput(1).NumOutput(1)
  .InputDevice(0, InputDevice::CPU);

DALI_SCHEMA(DummyDeviceTestGPU)
  .NumInput(1).NumOutput(1)
  .InputDevice(0, InputDevice::GPU);

// Test IsCompatibleDevice: CPU case (covers lines 23-24)
TEST(OpSpecTest, IsCompatibleDeviceCPU) {
  auto spec = OpSpec("DummyDeviceTestCPU")
              .AddArg("device", "cpu");

  // CPU input for CPU requirement - should succeed
  EXPECT_NO_THROW(spec.AddInput("input", StorageDevice::CPU));

  // GPU input for CPU requirement - should fail
  auto spec2 = OpSpec("DummyDeviceTestCPU")
               .AddArg("device", "cpu");
  EXPECT_THROW(spec2.AddInput("input", StorageDevice::GPU), std::runtime_error);
}

// Test IsCompatibleDevice: GPU case (covers lines 25-26)
TEST(OpSpecTest, IsCompatibleDeviceGPU) {
  auto spec = OpSpec("DummyDeviceTestGPU")
              .AddArg("device", "gpu");

  // GPU input for GPU requirement - should succeed
  EXPECT_NO_THROW(spec.AddInput("input", StorageDevice::GPU));

  // CPU input for GPU requirement - should fail
  auto spec2 = OpSpec("DummyDeviceTestGPU")
               .AddArg("device", "gpu");
  EXPECT_THROW(spec2.AddInput("input", StorageDevice::CPU), std::runtime_error);
}

// ===== ValidDevices Coverage (called in error messages) =====

DALI_SCHEMA(DummyValidDevicesTest)
  .NumInput(6).NumOutput(1)
  .InputDevice(0, InputDevice::CPU)
  .InputDevice(1, InputDevice::GPU)
  .InputDevice(2, InputDevice::MatchBackend)
  .InputDevice(3, InputDevice::MatchBackendOrCPU)
  .InputDevice(4, InputDevice::Any)
  .InputDevice(5, InputDevice::Metadata);

// Test ValidDevices: all cases by triggering device errors
// This covers the ValidDevices function lines 43-56
TEST(OpSpecTest, ValidDevicesCPU) {
  auto spec = OpSpec("DummyValidDevicesTest")
              .AddArg("device", "cpu");

  // Trigger error for InputDevice::CPU to call ValidDevices
  EXPECT_THROW({
    spec.AddInput("in0", StorageDevice::GPU);
  }, std::runtime_error);
}

TEST(OpSpecTest, ValidDevicesGPU) {
  auto spec = OpSpec("DummyValidDevicesTest")
              .AddArg("device", "cpu");

  // Trigger error for InputDevice::GPU to call ValidDevices
  EXPECT_THROW({
    spec.AddInput("dummy", StorageDevice::CPU);  // First input to get to index 1
    spec.AddInput("in1", StorageDevice::CPU);  // This should fail
  }, std::runtime_error);
}

TEST(OpSpecTest, ValidDevicesMatchBackend) {
  auto spec = OpSpec("DummyValidDevicesTest")
              .AddArg("device", "cpu");

  // InputDevice::MatchBackend with CPU op_type expects CPU
  // Trigger error by providing GPU
  EXPECT_THROW({
    spec.AddInput("in0", StorageDevice::CPU);
    spec.AddInput("in1", StorageDevice::GPU);
    spec.AddInput("in2", StorageDevice::GPU);  // This should fail (expects CPU for cpu op)
  }, std::runtime_error);
}

TEST(OpSpecTest, ValidDevicesMatchBackendOrCPU) {
  auto spec = OpSpec("DummyValidDevicesTest")
              .AddArg("device", "cpu");

  // InputDevice::MatchBackendOrCPU with CPU op_type expects CPU
  // For GPU op it would accept GPU or CPU
  // Trigger error by providing GPU with CPU op
  EXPECT_THROW({
    spec.AddInput("in0", StorageDevice::CPU);
    spec.AddInput("in1", StorageDevice::GPU);
    spec.AddInput("in2", StorageDevice::CPU);
    spec.AddInput("in3", StorageDevice::GPU);  // This should fail (CPU op expects CPU)
  }, std::runtime_error);
}

// Test ValidDevices with GPU operator
TEST(OpSpecTest, ValidDevicesGPUOperator) {
  auto spec = OpSpec("DummyValidDevicesTest")
              .AddArg("device", "gpu");

  // For MatchBackend with GPU op, expects GPU
  EXPECT_THROW({
    spec.AddInput("in0", StorageDevice::GPU);
    spec.AddInput("in1", StorageDevice::GPU);
    spec.AddInput("in2", StorageDevice::CPU);  // This should fail (GPU op expects GPU)
  }, std::runtime_error);
}

// ===== name_utils.cc coverage =====

DALI_SCHEMA(DummyNameUtilsNoDox)
  .NumInput(1).NumOutput(1);

DALI_SCHEMA(DummyNameUtilsWithDox)
  .NumInput(1).NumOutput(1)
  .InputDox(0, "the_input", "Tensor", "documented input");

// GetOpModule returns the `_module` argument verbatim.
TEST(NameUtilsTest, GetOpModule) {
  auto spec = OpSpec("DummyNameUtilsNoDox").AddArg("_module", "my.module"s);
  EXPECT_EQ(GetOpModule(spec), "my.module");
}

// GetOpDisplayName without the module path returns just the display name.
TEST(NameUtilsTest, GetOpDisplayNameNoModulePath) {
  auto spec = OpSpec("DummyNameUtilsNoDox").AddArg("_display_name", "MyOp"s);
  EXPECT_EQ(GetOpDisplayName(spec, false), "MyOp");
}

// GetOpDisplayName with the module path and a non-empty module prefixes it.
TEST(NameUtilsTest, GetOpDisplayNameWithModulePath) {
  auto spec = OpSpec("DummyNameUtilsNoDox")
                  .AddArg("_display_name", "MyOp"s)
                  .AddArg("_module", "my.module"s);
  EXPECT_EQ(GetOpDisplayName(spec, true), "my.module.MyOp");
}

// GetOpDisplayName with the module path but an empty module returns just the name.
TEST(NameUtilsTest, GetOpDisplayNameEmptyModule) {
  auto spec = OpSpec("DummyNameUtilsNoDox")
                  .AddArg("_display_name", "MyOp"s)
                  .AddArg("_module", ""s);
  EXPECT_EQ(GetOpDisplayName(spec, true), "MyOp");
}

// FormatInput for a schema without input docs omits the input name.
TEST(NameUtilsTest, FormatInputNoDox) {
  auto spec = OpSpec("DummyNameUtilsNoDox");
  EXPECT_EQ(FormatInput(spec, 0, false), "input `0`");
  EXPECT_EQ(FormatInput(spec, 2, true), "Input `2`");
}

// FormatInput for a schema with input docs includes the documented input name.
TEST(NameUtilsTest, FormatInputWithDox) {
  auto spec = OpSpec("DummyNameUtilsWithDox");
  EXPECT_EQ(FormatInput(spec, 0, false), "input `0` ('__the_input')");
  EXPECT_EQ(FormatInput(spec, 0, true), "Input `0` ('__the_input')");
}

// FormatOutput and FormatArgument formatting, capitalized and not.
TEST(NameUtilsTest, FormatOutputAndArgument) {
  auto spec = OpSpec("DummyNameUtilsNoDox");
  EXPECT_EQ(FormatOutput(spec, 1, false), "output `1`");
  EXPECT_EQ(FormatOutput(spec, 1, true), "Output `1`");
  EXPECT_EQ(FormatArgument(spec, "foo", false), "argument 'foo'");
  EXPECT_EQ(FormatArgument(spec, "foo", true), "Argument 'foo'");
}

}  // namespace dali
