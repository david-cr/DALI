// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <tuple>
#include <vector>
#include <complex>
#include <cmath>
#include "dali/kernels/signal/window/extract_windows_cpu.h"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace signal {
namespace window {
namespace test {

class ExtractWindowsCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2>, int64_t, int64_t, int64_t, int64_t, Padding>> {
 public:
  ExtractWindowsCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , window_length_(std::get<1>(GetParam()))
    , window_step_(std::get<2>(GetParam()))
    , axis_(std::get<3>(GetParam()))
    , window_center_(std::get<4>(GetParam()))
    , padding_(std::get<5>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~ExtractWindowsCpuTest() override = default;

  template <typename OutputType, typename InputType, int Dims, bool vertical>
  void RunTest();

 protected:
  void SetUp() final {
    SequentialFill(in_view_, 0);
  }
  TensorShape<2> data_shape_;
  int window_length_ = -1, window_step_ = -1, axis_ = -1, window_center_ = -1;
  Padding padding_ = Padding::Zero;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

template <typename T>
void print_data(const OutTensorCPU<T, 2>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      int k = i0 * sh[1] + i1;
      LOG_LINE << " " << data_view.data[k];
    }
    LOG_LINE << "\n";
  }
}

template <typename T>
void print_data(const OutTensorCPU<T, 3>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      for (int i2 = 0; i2 < sh[2]; i2++) {
        int k = i0 * sh[1] * sh[2] + i1 * sh[2] + i2;
        LOG_LINE << " " << data_view.data[k];
      }
      LOG_LINE << "\n";
    }
    LOG_LINE << "\n";
  }
  LOG_LINE << "\n";
}

template <typename OutputType, typename InputType, int Dims, bool vertical>
void ExtractWindowsCpuTest::RunTest() {
  constexpr int InputDims = Dims;
  constexpr int OutputDims = Dims + 1;
  ExtractWindowsCpu<OutputType, InputType, Dims, vertical> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  ExtractWindowsArgs args;
  args.window_length = window_length_;
  args.window_step = window_step_;
  args.axis = axis_;
  args.window_center = window_center_;
  args.padding = padding_;

  // Hamming window
  std::vector<float> window_fn_data(window_length_);
  HammingWindow(make_span(window_fn_data));
  auto window_fn_view = OutTensorCPU<float, 1>(window_fn_data.data(), {1});

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  auto n = in_view_.shape[axis_];
  auto nwindows = padding_ == Padding::None
    ? (n - window_length_) / window_step_ + 1
    : n / window_step_ + 1;
  auto expected_out_shape = vertical ?
    TensorShape<DynamicDimensions>{in_view_.shape[0], window_length_, nwindows} :
    TensorShape<DynamicDimensions>{in_view_.shape[0], nwindows, window_length_};
  ASSERT_EQ(expected_out_shape, out_shape);
  auto expected_out_size = volume(expected_out_shape);
  auto out_size = volume(out_shape);
  ASSERT_EQ(expected_out_size, out_size);

  std::vector<OutputType> expected_out(out_size);
  auto expected_out_view = OutTensorCPU<OutputType, OutputDims>(
    expected_out.data(), out_shape.to_static<OutputDims>());

  TensorShape<> in_shape = in_view_.shape;
  TensorShape<> in_strides = GetStrides(in_shape);

  TensorShape<> flat_out_shape = in_shape;
  flat_out_shape[InputDims-1] = nwindows * window_length_;
  TensorShape<> out_strides = GetStrides(flat_out_shape);

  auto in_stride = in_strides[axis_];
  auto out_stride = out_strides[axis_];

  int window_center_offset = 0;
  if (padding_ != Padding::None)
    window_center_offset = window_center_ < 0 ? window_length_ / 2 : window_center_;
  for (int i = 0; i < in_view_.shape[0]; i++) {
    auto *out_slice = expected_out_view.data + i * out_strides[0];
    auto *in_slice = in_view_.data + i * in_strides[0];
    for (int w = 0; w < nwindows; w++) {
      for (int t = 0; t < window_length_; t++) {
        auto out_k = vertical ? w + t * nwindows : w * window_length_ + t;
        auto in_k = w * window_step_ + t - window_center_offset;
        if (padding_ == Padding::Reflect) {
          while (in_k < 0 || in_k >= n) {
              in_k = (in_k < 0) ? -in_k : 2*n-2-in_k;
          }
        }
        out_slice[out_k] = (in_k >= 0 && in_k < n) ?
          window_fn_data[t] * in_slice[in_k] : 0;
      }
    }
  }


  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "expected out:\n";
  print_data(expected_out_view);

  std::vector<OutputType> out(out_size);
  auto out_view = OutTensorCPU<OutputType, OutputDims>(
    out.data(), out_shape.to_static<OutputDims>());
  kernel.Run(ctx, out_view, in_view_, window_fn_view, args);

  LOG_LINE << "out:\n";
  print_data(out_view);

  for (int idx = 0; idx < volume(out_view.shape); idx++) {
    ASSERT_EQ(expected_out[idx], out_view.data[idx]) <<
      "Output data doesn't match reference (idx=" << idx << ")";
  }
}

TEST_P(ExtractWindowsCpuTest, Vertical) {
  RunTest<float, float, 2, true>();
}

TEST_P(ExtractWindowsCpuTest, Horizontal) {
  RunTest<float, float, 2, false>();
}

INSTANTIATE_TEST_SUITE_P(ExtractWindowsCpuTest, ExtractWindowsCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{1, 12},
                    std::array<int64_t, 2>{2, 12}),
    testing::Values(4),  // window_length
    testing::Values(2),  // step
    testing::Values(1),  // axis
    testing::Values(0, 2, 4),  // window offsets
    testing::Values(Padding::None, Padding::Zero, Padding::Reflect)));  // reflect padding

// ============================================================================
// Additional tests for improved code coverage
// ============================================================================

// Test 1D input with vertical=false (Dims=1, horizontal layout)
TEST(ExtractWindowsCpu1DTest, Horizontal) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(20);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{20};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // Verify output shape
  EXPECT_EQ(out_shape[0], 9);  // nwindows = (20 - 4) / 2 + 1 = 9
  EXPECT_EQ(out_shape[1], 4);  // window_length
}

// Test 1D input with vertical=true (Dims=1, vertical layout)
TEST(ExtractWindowsCpu1DTest, Vertical) {
  ExtractWindowsCpu<float, float, 1, true> kernel;
  KernelContext ctx;

  std::vector<float> data(20);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{20};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // Verify output shape (vertical: window_length first, then nwindows)
  EXPECT_EQ(out_shape[0], 4);  // window_length
  EXPECT_EQ(out_shape[1], 9);  // nwindows
}

// Test with window_length <= 0 (triggers default to 1) - line 48
TEST(ExtractWindowsCpuEdgeCases, WindowLengthZero) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  // Window function with 1 element (matches the default window_length=1)
  std::vector<float> window_fn_data = {1.0f};
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {1});

  ExtractWindowsArgs args;
  args.window_length = 0;  // Should default to 1
  args.window_step = 1;
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // With window_length=1, step=1, input=10: nwindows = 10
  EXPECT_EQ(out_shape[0], 10);  // nwindows
  EXPECT_EQ(out_shape[1], 1);   // window_length (defaulted to 1)
}

// Test with window_step <= 0 (triggers default to 1) - line 49
TEST(ExtractWindowsCpuEdgeCases, WindowStepZero) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 0;  // Should default to 1
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // With window_length=4, step=1 (default), input=10: nwindows = (10 - 4) / 1 + 1 = 7
  EXPECT_EQ(out_shape[0], 7);  // nwindows
  EXPECT_EQ(out_shape[1], 4);  // window_length
}

// Test with axis < 0 (triggers default to last axis) - line 65
TEST(ExtractWindowsCpuEdgeCases, NegativeAxis) {
  ExtractWindowsCpu<float, float, 2, false> kernel;
  KernelContext ctx;

  std::vector<float> data(24);  // 2 x 12
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<2> in_shape{2, 12};
  OutTensorCPU<float, 2> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = -1;  // Should default to InputDims - 1 = 1
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 3> out_view(out.data(), out_shape.to_static<3>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // Axis defaults to 1 (last axis)
  // nwindows = (12 - 4) / 2 + 1 = 5
  EXPECT_EQ(out_shape[0], 2);  // first dim unchanged
  EXPECT_EQ(out_shape[1], 5);  // nwindows
  EXPECT_EQ(out_shape[2], 4);  // window_length
}

// Test with window_center < 0 and padding != None (auto-center) - line 52
TEST(ExtractWindowsCpuEdgeCases, AutoCenterWithPadding) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(20);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{20};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;
  args.window_center = -1;  // Auto-center: window_length / 2 = 2
  args.padding = Padding::Zero;  // Padding enabled

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // With padding, nwindows = n / step + 1 = 20 / 2 + 1 = 11
  EXPECT_EQ(out_shape[0], 11);  // nwindows
  EXPECT_EQ(out_shape[1], 4);   // window_length
}

// Test with explicit window_center and padding - line 52 (false branch)
TEST(ExtractWindowsCpuEdgeCases, ExplicitCenterWithPadding) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(20);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{20};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;
  args.window_center = 1;  // Explicit center offset
  args.padding = Padding::Reflect;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // With padding, nwindows = n / step + 1 = 20 / 2 + 1 = 11
  EXPECT_EQ(out_shape[0], 11);
  EXPECT_EQ(out_shape[1], 4);
}

// ============================================================================
// Error path tests (DALI_ENFORCE)
// ============================================================================

// Test error: empty window function (line 60)
TEST(ExtractWindowsCpuErrors, EmptyWindowFunction) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  // Empty window function
  std::vector<float> window_fn_data;
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {0});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  EXPECT_THROW(kernel.Setup(ctx, in_view, window_fn_view, args), DALIException);
}

// Test error: window function larger than window length (line 61)
TEST(ExtractWindowsCpuErrors, WindowFunctionTooLarge) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  // Window function larger than window_length
  std::vector<float> window_fn_data(8);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {8});

  ExtractWindowsArgs args;
  args.window_length = 4;  // Smaller than window function
  args.window_step = 2;
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  EXPECT_THROW(kernel.Setup(ctx, in_view, window_fn_view, args), DALIException);
}

// Test error: window center offset out of range (line 56)
TEST(ExtractWindowsCpuErrors, WindowCenterOutOfRange) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;
  args.window_center = 10;  // > window_length, invalid
  args.padding = Padding::Zero;  // Padding needed to use window_center

  EXPECT_THROW(kernel.Setup(ctx, in_view, window_fn_view, args), DALIException);
}

// Test error: invalid axis (line 66)
TEST(ExtractWindowsCpuErrors, InvalidAxis) {
  ExtractWindowsCpu<float, float, 2, false> kernel;
  KernelContext ctx;

  std::vector<float> data(24);
  TensorShape<2> in_shape{2, 12};
  OutTensorCPU<float, 2> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 5;  // Invalid: >= InputDims (2)
  args.window_center = -1;
  args.padding = Padding::None;

  EXPECT_THROW(kernel.Setup(ctx, in_view, window_fn_view, args), DALIException);
}

// Test with axis = 0 (first dimension, not last)
TEST(ExtractWindowsCpuEdgeCases, AxisZero) {
  ExtractWindowsCpu<float, float, 2, false> kernel;
  KernelContext ctx;

  std::vector<float> data(24);  // 12 x 2
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<2> in_shape{12, 2};
  OutTensorCPU<float, 2> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = 2;
  args.axis = 0;  // Extract windows along first axis
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 3> out_view(out.data(), out_shape.to_static<3>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // Axis 0: nwindows = (12 - 4) / 2 + 1 = 5
  EXPECT_EQ(out_shape[0], 5);  // nwindows (horizontal layout)
  EXPECT_EQ(out_shape[1], 4);  // window_length
  EXPECT_EQ(out_shape[2], 2);  // second dim unchanged
}

// Test with negative window_length (triggers default)
TEST(ExtractWindowsCpuEdgeCases, NegativeWindowLength) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data = {1.0f};
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {1});

  ExtractWindowsArgs args;
  args.window_length = -5;  // Negative, should default to 1
  args.window_step = 1;
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  EXPECT_EQ(out_shape[0], 10);  // nwindows
  EXPECT_EQ(out_shape[1], 1);   // window_length defaulted to 1
}

// Test with negative window_step (triggers default)
TEST(ExtractWindowsCpuEdgeCases, NegativeWindowStep) {
  ExtractWindowsCpu<float, float, 1, false> kernel;
  KernelContext ctx;

  std::vector<float> data(10);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<float>(i);
  }
  TensorShape<1> in_shape{10};
  OutTensorCPU<float, 1> in_view(data.data(), in_shape);

  std::vector<float> window_fn_data(4);
  HammingWindow(make_span(window_fn_data));
  OutTensorCPU<float, 1> window_fn_view(window_fn_data.data(), {4});

  ExtractWindowsArgs args;
  args.window_length = 4;
  args.window_step = -3;  // Negative, should default to 1
  args.axis = 0;
  args.window_center = -1;
  args.padding = Padding::None;

  auto reqs = kernel.Setup(ctx, in_view, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  std::vector<float> out(volume(out_shape));
  OutTensorCPU<float, 2> out_view(out.data(), out_shape.to_static<2>());

  kernel.Run(ctx, out_view, in_view, window_fn_view, args);

  // step defaults to 1: nwindows = (10 - 4) / 1 + 1 = 7
  EXPECT_EQ(out_shape[0], 7);
  EXPECT_EQ(out_shape[1], 4);
}

}  // namespace test
}  // namespace window
}  // namespace signal
}  // namespace kernels
}  // namespace dali
