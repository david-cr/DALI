// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/common.h"  // NOLINT
#include "dali/pipeline/operator/error_reporting.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <system_error>
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

DALI_SCHEMA(PipelineCommonTest).AddOptionalArg("size", "size", std::vector<float>{}, true);

TEST(PipelineCommon, GetShapeLikeArgumentScalar) {
  OpSpec spec("PipelineCommonTest");
  ArgumentWorkspace ws;
  spec.AddArg("size", 1.5f);
  vector<float> shape;
  int nsamples, D;
  std::tie(nsamples, D) = GetShapeLikeArgument<float>(shape, spec, "size", ws, 5, 3);
  EXPECT_EQ(D, 3);
  ASSERT_EQ(shape.size(), 15);
  for (size_t i = 0; i < 15; i++) {
    EXPECT_EQ(shape[i], 1.5f);
  }
  shape.clear();
}

TEST(PipelineCommon, GetShapeLikeArgumentVector) {
  OpSpec spec("PipelineCommonTest");
  ArgumentWorkspace ws;
  vector<float> src_shape = {-0.75f, 1, 2.75f, 3.25f};
  spec.SetArg("size", src_shape);
  int max_batch_size = 3;
  spec.SetArg("max_batch_size", max_batch_size);

  vector<float> shape;
  int nsamples, D;
  std::tie(nsamples, D) = GetShapeLikeArgument<float>(shape, spec, "size", ws, max_batch_size);
  EXPECT_EQ(D, 4);
  ASSERT_EQ(shape.size(), 12);
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 4; d++) EXPECT_EQ(shape[i * 4 + d], src_shape[d]);
  }

  vector<int> ref_ishape = { -1, 1, 3, 3 };
  vector<int> ishape;
  spec.SetArg("size", src_shape);
  spec.SetArg("batch_size", 3);
  std::tie(nsamples, D) = GetShapeLikeArgument<float>(ishape, spec, "size", ws, max_batch_size);
  EXPECT_EQ(D, 4);
  ASSERT_EQ(shape.size(), 12);
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 4; d++)
      EXPECT_EQ(ishape[i * 4 + d], ref_ishape[d]) << "@ shape[" << i << "][" << d << "]";
  }
}

TEST(PipelineCommon, GetShapeLikeArgumentInput) {
  OpSpec spec("PipelineCommonTest");
  ArgumentWorkspace ws;
  int D = 5;
  int N = 7;
  auto input = std::make_shared<TensorList<CPUBackend>>();
  input->set_pinned(false);

  // specify the shape as a list of 1D tensors
  input->Resize(uniform_list_shape<1>(N, {D}), DALI_FLOAT);
  for (int sample_idx = 0; sample_idx < N; sample_idx++) {
    float *shape_data = input->mutable_tensor<float>(sample_idx);
    for (int i = 0; i < D; i++) {
      shape_data[i] = (sample_idx * D + i) * 1.1f;
    }
  }

  spec.SetArg("max_batch_size", N);
  spec.AddArgumentInput("size", "size");
  ws.AddArgumentInput("size", input);

  vector<float> shape;
  int nsamples, out_d;
  std::tie(nsamples, out_d) = GetShapeLikeArgument<float>(shape, spec, "size", ws, N);
  EXPECT_EQ(out_d, D) << "Dimensionality should match the size of the tensors in the list.";
  ASSERT_EQ(shape.size(), N * D) << "Total size of the shape should be batch x ndim";
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < D; d++)
      EXPECT_EQ(shape[i * D + d], (i * D + d) * 1.1f);
  }


  // specify the shape as a list of scalars - this will cause the extend to be
  // broadcast to all extents when the extent is known
  ws.Clear();

  input->Resize(TensorListShape<0>(N));
  for (int sample_idx = 0; sample_idx < N; sample_idx++) {
    float *shape_data = input->mutable_tensor<float>(sample_idx);
    shape_data[0] = sample_idx * 1.1f;
  }

  ws.AddArgumentInput("size", input);

  vector<int> ishape;
  std::tie(nsamples, out_d) = GetShapeLikeArgument<float>(ishape, spec, "size", ws, N, D);
  EXPECT_EQ(out_d, D) << "A list of scalars can be broadcast to any number of dimensions.";
  ASSERT_EQ(shape.size(), N * D) << "Total size of the shape should be batch x ndim";
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < D; d++)
      EXPECT_EQ(ishape[i * D + d], std::lround(i * 1.1f));
  }

  shape.clear();
  // if the extent is not know, a list of scalars indicates 1D shapes
  std::tie(nsamples, out_d) = GetShapeLikeArgument<float>(shape, spec, "size", ws, N);
  EXPECT_EQ(out_d, 1) << "A list of scalars should be interpreted as a 1D shape";
  D = 1;
  ASSERT_EQ(shape.size(), N * D) << "Total size of the shape should be batch x ndim";
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < D; d++)
      EXPECT_EQ(shape[i * D + d], i * 1.1f);
  }
}

// Test that covers the DALI_ENFORCE validation in GetOperatorOriginInfo (lines 38-40)
// These tests verify that mismatched stack trace array sizes trigger the appropriate errors
TEST(ErrorReporting, GetOperatorOriginInfoMismatchedArraySizes) {
  // Test case 1: filename and lineno arrays have different sizes
  {
    OpSpec spec("PipelineCommonTest");
    spec.AddArg("_origin_stack_filename", std::vector<std::string>{"file1.py", "file2.py"});
    spec.AddArg("_origin_stack_lineno", std::vector<int>{10, 20, 30}); // Different size

    EXPECT_THROW({
      GetOperatorOriginInfo(spec);
    }, std::exception);

    try {
      GetOperatorOriginInfo(spec);
      FAIL() << "Expected exception for mismatched filename and lineno array sizes";
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      EXPECT_NE(error_msg.find("origin_stack_filename.size() == origin_stack_lineno.size()"), std::string::npos)
          << "Expected error message about array size mismatch, got: " << error_msg;
    }
  }

  // Test case 2: filename and name arrays have different sizes
  {
    OpSpec spec("PipelineCommonTest");
    spec.AddArg("_origin_stack_filename", std::vector<std::string>{"file1.py"});
    spec.AddArg("_origin_stack_lineno", std::vector<int>{10});
    spec.AddArg("_origin_stack_name", std::vector<std::string>{"func1", "func2"}); // Different size

    EXPECT_THROW({
      GetOperatorOriginInfo(spec);
    }, std::exception);

    try {
      GetOperatorOriginInfo(spec);
      FAIL() << "Expected exception for mismatched filename and name array sizes";
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      EXPECT_NE(error_msg.find("origin_stack_filename.size() == origin_stack_name.size()"), std::string::npos)
          << "Expected error message about array size mismatch, got: " << error_msg;
    }
  }

  // Test case 3: filename and line arrays have different sizes
  {
    OpSpec spec("PipelineCommonTest");
    spec.AddArg("_origin_stack_filename", std::vector<std::string>{"file1.py", "file2.py"});
    spec.AddArg("_origin_stack_lineno", std::vector<int>{10, 20});
    spec.AddArg("_origin_stack_name", std::vector<std::string>{"func1", "func2"});
    spec.AddArg("_origin_stack_line", std::vector<std::string>{"line1"}); // Different size

    EXPECT_THROW({
      GetOperatorOriginInfo(spec);
    }, std::exception);

    try {
      GetOperatorOriginInfo(spec);
      FAIL() << "Expected exception for mismatched filename and line array sizes";
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      EXPECT_NE(error_msg.find("origin_stack_filename.size() == origin_stack_line.size()"), std::string::npos)
          << "Expected error message about array size mismatch, got: " << error_msg;
    }
  }
}

// Test that covers the successful case when all arrays have matching sizes
TEST(ErrorReporting, GetOperatorOriginInfoMatchingArraySizes) {
  OpSpec spec("PipelineCommonTest");
  spec.AddArg("_origin_stack_filename", std::vector<std::string>{"file1.py", "file2.py"});
  spec.AddArg("_origin_stack_lineno", std::vector<int>{10, 20});
  spec.AddArg("_origin_stack_name", std::vector<std::string>{"func1", "func2"});
  spec.AddArg("_origin_stack_line", std::vector<std::string>{"line1", "line2"});

  // This should not throw an exception
  EXPECT_NO_THROW({
    auto result = GetOperatorOriginInfo(spec);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].filename, "file1.py");
    EXPECT_EQ(result[0].lineno, 10);
    EXPECT_EQ(result[0].name, "func1");
    EXPECT_EQ(result[0].line, "line1");
    EXPECT_EQ(result[1].filename, "file2.py");
    EXPECT_EQ(result[1].lineno, 20);
    EXPECT_EQ(result[1].name, "func2");
    EXPECT_EQ(result[1].line, "line2");
  });
}

// Test that covers the edge case with empty arrays
TEST(ErrorReporting, GetOperatorOriginInfoEmptyArrays) {
  OpSpec spec("PipelineCommonTest");
  spec.AddArg("_origin_stack_filename", std::vector<std::string>{});
  spec.AddArg("_origin_stack_lineno", std::vector<int>{});
  spec.AddArg("_origin_stack_name", std::vector<std::string>{});
  spec.AddArg("_origin_stack_line", std::vector<std::string>{});

  // Empty arrays should work fine
  EXPECT_NO_THROW({
    auto result = GetOperatorOriginInfo(spec);
    EXPECT_EQ(result.size(), 0);
  });
}

// Test that covers the std::system_error exception handling in PropagateError (lines 82-84)
TEST(ErrorReporting, PropagateErrorSystemError) {
  // Create an ErrorInfo with system_error exception
  ErrorInfo error_info;
  error_info.context_info = "Test context: ";
  error_info.additional_message = " Additional info";

  // Create a system_error exception
  std::system_error original_error(std::make_error_code(std::errc::invalid_argument), "Original system error");
  error_info.exception = std::make_exception_ptr(original_error);

  // Test that PropagateError rethrows system_error with enhanced context
  EXPECT_THROW({
    PropagateError(error_info);
  }, std::system_error);

  try {
    PropagateError(error_info);
    FAIL() << "Expected system_error exception";
  } catch (const std::system_error& e) {
    std::string error_msg = e.what();
    EXPECT_NE(error_msg.find("Test context:"), std::string::npos)
        << "Expected context info in error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Original system error"), std::string::npos)
        << "Expected original error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Additional info"), std::string::npos)
        << "Expected additional message, got: " << error_msg;
    EXPECT_EQ(e.code(), std::make_error_code(std::errc::invalid_argument))
        << "Expected error code to be preserved";
  } catch (...) {
    FAIL() << "Expected system_error exception, got different exception type";
  }
}

// Test that covers the dali::invalid_key exception handling in PropagateError (lines 89-90)
TEST(ErrorReporting, PropagateErrorInvalidKey) {
  // Create an ErrorInfo with invalid_key exception
  ErrorInfo error_info;
  error_info.context_info = "Key error context: ";
  error_info.additional_message = " Key not found";

  // Create an invalid_key exception
  dali::invalid_key original_error("Original key error");
  error_info.exception = std::make_exception_ptr(original_error);

  // Test that PropagateError rethrows invalid_key with enhanced context
  EXPECT_THROW({
    PropagateError(error_info);
  }, dali::invalid_key);

  try {
    PropagateError(error_info);
    FAIL() << "Expected invalid_key exception";
  } catch (const dali::invalid_key& e) {
    std::string error_msg = e.what();
    EXPECT_NE(error_msg.find("Key error context:"), std::string::npos)
        << "Expected context info in error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Original key error"), std::string::npos)
        << "Expected original error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Key not found"), std::string::npos)
        << "Expected additional message, got: " << error_msg;
  } catch (...) {
    FAIL() << "Expected invalid_key exception, got different exception type";
  }
}

// Test that covers the system_error with different error codes
TEST(ErrorReporting, PropagateErrorSystemErrorDifferentCodes) {
  // Test with permission_denied error code
  ErrorInfo error_info;
  error_info.context_info = "Permission error: ";
  error_info.additional_message = " Access denied";

  std::system_error original_error(std::make_error_code(std::errc::permission_denied), "Permission denied");
  error_info.exception = std::make_exception_ptr(original_error);

  try {
    PropagateError(error_info);
    FAIL() << "Expected system_error exception";
  } catch (const std::system_error& e) {
    EXPECT_EQ(e.code(), std::make_error_code(std::errc::permission_denied))
        << "Expected permission_denied error code to be preserved";
    std::string error_msg = e.what();
    EXPECT_NE(error_msg.find("Permission error:"), std::string::npos)
        << "Expected context info in error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Permission denied"), std::string::npos)
        << "Expected original error message, got: " << error_msg;
  }
}

// Test that covers the invalid_key with different error messages
TEST(ErrorReporting, PropagateErrorInvalidKeyDifferentMessages) {
  // Test with different key error message
  ErrorInfo error_info;
  error_info.context_info = "Database error: ";
  error_info.additional_message = " Table not found";

  dali::invalid_key original_error("Database key missing");
  error_info.exception = std::make_exception_ptr(original_error);

  try {
    PropagateError(error_info);
    FAIL() << "Expected invalid_key exception";
  } catch (const dali::invalid_key& e) {
    std::string error_msg = e.what();
    EXPECT_NE(error_msg.find("Database error:"), std::string::npos)
        << "Expected context info in error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Database key missing"), std::string::npos)
        << "Expected original error message, got: " << error_msg;
    EXPECT_NE(error_msg.find("Table not found"), std::string::npos)
        << "Expected additional message, got: " << error_msg;
  }
}

}  // namespace dali
