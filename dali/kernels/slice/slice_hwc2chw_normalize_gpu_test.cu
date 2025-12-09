// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/slice/slice_hwc2chw_normalize_gpu.h"
#include "dali/kernels/slice/slice_kernel_test.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace slice_flip_normalize {
namespace test {

template <typename Out>
class SliceHwc2ChwNormalizeGPUTest : public ::testing::Test {
 public:
  using Kernel = SliceHwc2HwcChwNormalizeGPU<Out>;
  using SampleArgs = typename Kernel::SampleArgs;
  static constexpr int ndim = 3;

  void SetupContext() {
    ctx_.gpu.stream = 0;
    ctx_.scratchpad = &scratchpad_;
  }

  void PrepareInput(const TensorListShape<ndim> &shapes) {
    input_.reshape(shapes);
    auto cpu_view = input_.cpu();
    // Fill with sequential values for testing
    for (int s = 0; s < shapes.num_samples(); s++) {
      auto *ptr = cpu_view.tensor_data(s);
      int64_t size = shapes.tensor_size(s);
      for (int64_t i = 0; i < size; i++) {
        ptr[i] = static_cast<uint8_t>((i + s * 17) % 256);
      }
    }
  }

  TensorListShape<ndim> MakeShapes(std::vector<TensorShape<ndim>> shapes) {
    return TensorListShape<ndim>(shapes);
  }

  // Helper to create ROI from HWC shape - ROI uses {x, y} = {W, H} ordering
  static Roi<2> MakeFullRoi(int H, int W) {
    return Roi<2>{{0, 0}, {W, H}};
  }

  static Roi<2> MakeCropRoi(int y_start, int x_start, int y_end, int x_end) {
    return Roi<2>{{x_start, y_start}, {x_end, y_end}};
  }

  KernelContext ctx_;
  DynamicScratchpad scratchpad_;
  Kernel kernel_;
  TestTensorList<uint8_t, ndim> input_;
};

using SliceHwc2ChwNormalizeGPUTest_float = SliceHwc2ChwNormalizeGPUTest<float>;
using SliceHwc2ChwNormalizeGPUTest_float16 = SliceHwc2ChwNormalizeGPUTest<float16>;

// Basic test with HWC output layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, BasicHWC) {
  SetupContext();

  // Create 3-channel HWC input (required by this kernel)
  // Shape: {H, W, C} = {10, 20, 3} and {15, 25, 3}
  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(2);
  // Full image crop - ROI uses {x, y} = {W, H} ordering
  args[0].roi = MakeFullRoi(10, 20);  // H=10, W=20
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  args[1].roi = MakeFullRoi(15, 25);  // H=15, W=25
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Verify output shape matches expected HWC layout
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
  EXPECT_EQ(req.output_shapes[0][1], TensorShape<>(15, 25, 3));
}

// Basic test with CHW output layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, BasicCHW) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(2);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  args[1].roi = MakeFullRoi(15, 25);
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Verify output shape matches expected CHW layout (channels first)
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(3, 10, 20));
  EXPECT_EQ(req.output_shapes[0][1], TensorShape<>(3, 15, 25));
}

// Test with x-flip enabled
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, WithFlipX) {
  SetupContext();

  // Input: {H=10, W=20, C=3}
  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  // Crop from (y=2, x=3) to (y=8, x=15) -> output {H=6, W=12, C=3}
  args[0].roi = MakeCropRoi(2, 3, 8, 15);
  args[0].mean = {10.0f, 20.0f, 30.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = true;  // Enable flip

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(6, 12, 3));
}

// Test with channel padding (fill_values has 4 channels, input has 3)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, WithPadding) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  // 4 fill values causes padding to 4 channels
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Output should have 4 channels due to padding
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 4));
}

// Test with channel padding in CHW layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, WithPaddingCHW) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  // 4 fill values causes padding to 4 channels
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Output should have 4 channels in CHW layout
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(4, 10, 20));
}

// Test with x-crop (crop in x dimension triggers slice path)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, WithXCrop) {
  SetupContext();

  // Input: {H=10, W=20, C=3}
  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  // Crop that doesn't use the full width - triggers need_crop_x path
  // Crop from (y=0, x=5) to (y=10, x=15) -> output {H=10, W=10, C=3}
  args[0].roi = MakeCropRoi(0, 5, 10, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(3, 10, 10));
}

// Test with flip and padding combined
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, FlipAndPadding) {
  SetupContext();

  // Input: {H=10, W=20, C=3}
  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  // Crop from (y=2, x=3) to (y=8, x=18) -> output {H=6, W=15, C=4}
  args[0].roi = MakeCropRoi(2, 3, 8, 18);
  args[0].mean = {10.0f, 20.0f, 30.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 0.5f};
  args[0].flip_x = true;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(6, 15, 4));
}

// Test with multiple samples having different flip settings
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, MixedFlipSettings) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}, {8, 16, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(3);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = true;

  args[1].roi = MakeFullRoi(15, 25);
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = false;

  args[2].roi = MakeFullRoi(8, 16);
  args[2].mean = {0.0f, 0.0f, 0.0f};
  args[2].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[2].fill_values = {0.0f, 0.0f, 0.0f};
  args[2].flip_x = true;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// Test with float16 output type
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, BasicFloat16HWC) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
}

// Test float16 with padding (triggers specialized Hwc2HwcNormalizePadFp16 kernel)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16WithPadding) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};  // Padding to 4 channels
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 4));
}

// Test float16 with padding and flip
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16WithPaddingAndFlip) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = true;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 4));
}

// Test float16 CHW output
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, BasicFloat16CHW) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(3, 10, 20));
}

// Test with large batch to ensure proper handling
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, LargeBatch) {
  SetupContext();

  int num_samples = 32;
  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < num_samples; i++) {
    shapes.push_back({10 + i % 5, 20 + i % 7, 3});
  }
  TensorListShape<3> in_shapes(shapes);
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(num_samples);
  for (int i = 0; i < num_samples; i++) {
    int h = 10 + i % 5;
    int w = 20 + i % 7;
    args[i].roi = MakeFullRoi(h, w);
    args[i].mean = {0.0f, 0.0f, 0.0f};
    args[i].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    args[i].fill_values = {0.0f, 0.0f, 0.0f};
    args[i].flip_x = (i % 2 == 0);
  }

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// Test x-crop with HWC layout (triggers SliceHwc2HwcNormalize kernel)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, XCropHWC) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  // Crop from (y=0, x=5) to (y=10, x=15) -> output {H=10, W=10, C=3}
  args[0].roi = MakeCropRoi(0, 5, 10, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 10, 3));
}

// Test x-crop with padding in HWC layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, XCropWithPaddingHWC) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(0, 5, 10, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};  // Padding to 4 channels
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 10, 4));
}

// Test x-crop with flip in CHW layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, XCropWithFlipCHW) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(2, 5, 8, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = true;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(3, 6, 10));
}

// Test x-crop with padding in CHW layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, XCropWithPaddingCHW) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(0, 5, 10, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(4, 10, 10));
}

// ============================================================================
// Edge case tests for improved code coverage
// ============================================================================

// Test with shorter mean/stddev arrays than number of channels
// This triggers the loop at lines 794-796 that fills remaining norm values
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, ShorterMeanStddev) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  // Provide only 2 mean/stddev values for 3 channels
  // The kernel should use default values (0.0, 1.0) for the 3rd channel
  args[0].mean = {10.0f, 20.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 128.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
}

// Test with single mean/stddev value (broadcast to all channels)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, SingleMeanStddev) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  // Provide only 1 mean/stddev value for 3 channels
  args[0].mean = {10.0f};
  args[0].inv_stddev = {1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
}

// Test with empty mean/stddev (all defaults)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, EmptyMeanStddev) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  // Empty mean/stddev - kernel should use defaults (0.0, 1.0) for all channels
  args[0].mean = {};
  args[0].inv_stddev = {};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
}

// Test with empty fill_values (kernel fills with zeros)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, EmptyFillValues) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  // Empty fill_values
  args[0].fill_values = {};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
}

// Test error: invalid output layout
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, InvalidOutputLayout) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  // Invalid layout should throw
  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "NCHW"), DALIException);
}

// Test error: mismatched args count
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, MismatchedArgsCount) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}});
  PrepareInput(in_shapes);

  // Only 1 arg for 2 samples
  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test error: ROI requests padding instead of cropping
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, RoiRequestsPadding) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  // ROI larger than input - requests padding which is not allowed
  args[0].roi = Roi<2>{{0, 0}, {30, 15}};  // W=30 > input W=20
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test error: inconsistent fill_values sizes across samples
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, InconsistentFillValues) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(2);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};  // 4 fill values
  args[0].flip_x = false;

  args[1].roi = MakeFullRoi(15, 25);
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};  // 3 fill values - mismatch!
  args[1].flip_x = false;

  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test float16 with x-crop (triggers SliceHwc2HwcNormalize for fp16)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16XCrop) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(0, 5, 10, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 10, 3));
}

// Test float16 with x-crop and padding
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16XCropWithPadding) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(0, 5, 10, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 10, 4));
}

// Test float16 CHW with padding
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16CHWWithPadding) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(4, 10, 20));
}

// Test float16 CHW with x-crop
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16CHWXCrop) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(2, 5, 8, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(3, 6, 10));
}

// Test float16 CHW with x-crop and padding
TEST_F(SliceHwc2ChwNormalizeGPUTest_float16, Float16CHWXCropWithPadding) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeCropRoi(2, 5, 8, 15);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float16, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(4, 6, 10));
}

// Test single sample batch (boundary case)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, SingleSample) {
  SetupContext();

  auto in_shapes = MakeShapes({{8, 16, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(8, 16);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(8, 16, 3));
}

// Test with very small images
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, SmallImages) {
  SetupContext();

  auto in_shapes = MakeShapes({{1, 1, 3}, {2, 2, 3}, {1, 3, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(3);
  args[0].roi = MakeFullRoi(1, 1);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  args[1].roi = MakeFullRoi(2, 2);
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = true;

  args[2].roi = MakeFullRoi(1, 3);
  args[2].mean = {0.0f, 0.0f, 0.0f};
  args[2].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[2].fill_values = {0.0f, 0.0f, 0.0f};
  args[2].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(3, 1, 1));
  EXPECT_EQ(req.output_shapes[0][1], TensorShape<>(3, 2, 2));
  EXPECT_EQ(req.output_shapes[0][2], TensorShape<>(3, 1, 3));
}

// Test with partial fill_values (less than output channels but non-empty)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, PartialFillValues) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  // 2 fill values - remaining should be filled with 0
  args[0].fill_values = {0.5f, 0.5f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Output should have 3 channels (fill_values < input channels doesn't pad)
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 20, 3));
}

// ============================================================================
// Additional error path tests for DALI_ENFORCE coverage
// ============================================================================

// Test error: non-3-channel input (line 855 - nchannels_ != kStaticChannels)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, NonThreeChannelInput) {
  SetupContext();

  // Input with 4 channels instead of 3 - kernel only supports 3 channels
  auto in_shapes = MakeShapes({{10, 20, 4}});
  input_.reshape(in_shapes);
  auto cpu_view = input_.cpu();
  auto *ptr = cpu_view.tensor_data(0);
  int64_t size = in_shapes.tensor_size(0);
  for (int64_t i = 0; i < size; i++) {
    ptr[i] = static_cast<uint8_t>(i % 256);
  }

  std::vector<SampleArgs> args(1);
  args[0].roi = Roi<2>{{0, 0}, {20, 10}};
  args[0].mean = {0.0f, 0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  // Should throw because kernel only supports 3 input channels
  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test error: single-channel input
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, SingleChannelInput) {
  SetupContext();

  // Input with 1 channel instead of 3
  auto in_shapes = MakeShapes({{10, 20, 1}});
  input_.reshape(in_shapes);
  auto cpu_view = input_.cpu();
  auto *ptr = cpu_view.tensor_data(0);
  int64_t size = in_shapes.tensor_size(0);
  for (int64_t i = 0; i < size; i++) {
    ptr[i] = static_cast<uint8_t>(i % 256);
  }

  std::vector<SampleArgs> args(1);
  args[0].roi = Roi<2>{{0, 0}, {20, 10}};
  args[0].mean = {0.0f};
  args[0].inv_stddev = {1.0f};
  args[0].fill_values = {0.0f};
  args[0].flip_x = false;

  // Should throw because kernel only supports 3 input channels
  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test error: HWC output with more than 4 channels (line 859)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, HWCTooManyOutputChannels) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  // 5 fill values - would require 5 output channels, but HWC only supports 3 or 4
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f};
  args[0].flip_x = false;

  // Should throw because HWC layout only supports 3 or 4 output channels
  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test error: inconsistent channel counts across samples (line 842)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, InconsistentChannelCounts) {
  SetupContext();

  // Create inputs with different channel counts - this should fail validation
  // Sample 0: 3 channels, Sample 1: 4 channels
  std::vector<TensorShape<3>> shapes = {{10, 20, 3}, {15, 25, 4}};
  TensorListShape<3> in_shapes(shapes);
  input_.reshape(in_shapes);
  auto cpu_view = input_.cpu();
  for (int s = 0; s < in_shapes.num_samples(); s++) {
    auto *ptr = cpu_view.tensor_data(s);
    int64_t size = in_shapes.tensor_size(s);
    for (int64_t i = 0; i < size; i++) {
      ptr[i] = static_cast<uint8_t>(i % 256);
    }
  }

  std::vector<SampleArgs> args(2);
  args[0].roi = Roi<2>{{0, 0}, {20, 10}};
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  args[1].roi = Roi<2>{{0, 0}, {25, 15}};
  args[1].mean = {0.0f, 0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f, 1.0f, 1.0f, 1.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = false;

  // Should throw because samples have different channel counts
  EXPECT_THROW(kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC"), DALIException);
}

// Test with zero-size ROI (triggers sample_size == 0 path, lines 897-898)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, ZeroSizeRoi) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(2);
  // First sample: zero-height ROI (empty)
  args[0].roi = Roi<2>{{0, 5}, {10, 5}};  // y_start == y_end = 5, so height = 0
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  // Second sample: normal ROI
  args[1].roi = MakeFullRoi(15, 25);
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // First sample has 0-height output
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(0, 10, 3));
  EXPECT_EQ(req.output_shapes[0][1], TensorShape<>(15, 25, 3));
}

// Test with zero-width ROI
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, ZeroWidthRoi) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  // Zero-width ROI
  args[0].roi = Roi<2>{{5, 0}, {5, 10}};  // x_start == x_end = 5, so width = 0
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Output has 0-width
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 0, 3));
}

// Test with all samples having zero-size ROI (triggers nonempty_samples == 0, line 936-937)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, AllEmptySamples) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}, {15, 25, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(2);
  // Both samples have empty ROIs
  args[0].roi = Roi<2>{{0, 0}, {0, 10}};  // width = 0
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[0].fill_values = {0.0f, 0.0f, 0.0f};
  args[0].flip_x = false;

  args[1].roi = Roi<2>{{0, 5}, {25, 5}};  // height = 0
  args[1].mean = {0.0f, 0.0f, 0.0f};
  args[1].inv_stddev = {1.0f, 1.0f, 1.0f};
  args[1].fill_values = {0.0f, 0.0f, 0.0f};
  args[1].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "HWC");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  // This should trigger early return at line 936-937
  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(10, 0, 3));
  EXPECT_EQ(req.output_shapes[0][1], TensorShape<>(0, 25, 3));
}

// Test CHW layout with many output channels (allowed in CHW, unlike HWC)
TEST_F(SliceHwc2ChwNormalizeGPUTest_float, CHWManyOutputChannels) {
  SetupContext();

  auto in_shapes = MakeShapes({{10, 20, 3}});
  PrepareInput(in_shapes);

  std::vector<SampleArgs> args(1);
  args[0].roi = MakeFullRoi(10, 20);
  args[0].mean = {0.0f, 0.0f, 0.0f};
  args[0].inv_stddev = {1.0f, 1.0f, 1.0f};
  // 6 fill values - CHW layout should support this (unlike HWC)
  args[0].fill_values = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  args[0].flip_x = false;

  auto req = kernel_.Setup(ctx_, in_shapes, make_cspan(args), "CHW");

  TestTensorList<float, 3> output;
  output.reshape(req.output_shapes[0].to_static<3>());

  kernel_.Run(ctx_, output.gpu(), input_.gpu(), make_cspan(args));
  CUDA_CALL(cudaStreamSynchronize(0));

  // CHW allows more output channels
  EXPECT_EQ(req.output_shapes[0][0], TensorShape<>(6, 10, 20));
}

}  // namespace test
}  // namespace slice_flip_normalize
}  // namespace kernels
}  // namespace dali
