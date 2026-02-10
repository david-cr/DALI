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
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "dali/operators/bbox/bbox_rotate.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/common.h"

namespace dali {
namespace testing {

using BBox = std::array<float, 4>;

class BBoxRotateTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  // Create a basic OpSpec with common defaults
  OpSpec MakeOpSpec(float angle, const std::vector<int> &input_shape,
                    const std::string &mode = "expand",
                    const std::string &bbox_layout = "xyXY",
                    bool bbox_normalized = true,
                    bool keep_size = false,
                    float remove_threshold = 0.1f,
                    const std::string &shape_layout = "HW") {
    return OpSpec("BBoxRotate")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("angle", angle)
        .AddArg("input_shape", input_shape)
        .AddArg("mode", mode)
        .AddArg("bbox_layout", bbox_layout)
        .AddArg("bbox_normalized", bbox_normalized)
        .AddArg("keep_size", keep_size)
        .AddArg("remove_threshold", remove_threshold)
        .AddArg("shape_layout", shape_layout)
        .AddInput("bboxes", StorageDevice::CPU)
        .AddOutput("out_bboxes", StorageDevice::CPU);
  }

  // Create OpSpec that also accepts labels as second input
  OpSpec MakeOpSpecWithLabels(float angle, const std::vector<int> &input_shape,
                              const std::string &mode = "expand",
                              const std::string &bbox_layout = "xyXY",
                              bool bbox_normalized = true,
                              bool keep_size = false,
                              float remove_threshold = 0.1f) {
    return OpSpec("BBoxRotate")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("angle", angle)
        .AddArg("input_shape", input_shape)
        .AddArg("mode", mode)
        .AddArg("bbox_layout", bbox_layout)
        .AddArg("bbox_normalized", bbox_normalized)
        .AddArg("keep_size", keep_size)
        .AddArg("remove_threshold", remove_threshold)
        .AddInput("bboxes", StorageDevice::CPU)
        .AddInput("labels", StorageDevice::CPU)
        .AddOutput("out_bboxes", StorageDevice::CPU)
        .AddOutput("out_labels", StorageDevice::CPU);
  }

  // Create box input TensorList from vector of boxes (each sample is Nx4)
  std::shared_ptr<TensorList<CPUBackend>> MakeBoxes(
      const std::vector<std::vector<BBox>> &batched_boxes) {
    auto tl = std::make_shared<TensorList<CPUBackend>>();
    int num_samples = batched_boxes.size();
    TensorListShape<2> shape(num_samples);
    for (int i = 0; i < num_samples; i++) {
      shape.set_tensor_shape(i, {static_cast<int64_t>(batched_boxes[i].size()), 4});
    }
    tl->Resize(shape, DALI_FLOAT);
    for (int i = 0; i < num_samples; i++) {
      auto *data = tl->mutable_tensor<float>(i);
      for (size_t j = 0; j < batched_boxes[i].size(); j++) {
        for (int k = 0; k < 4; k++) {
          data[4 * j + k] = batched_boxes[i][j][k];
        }
      }
    }
    return tl;
  }

  // Create 1D label input TensorList
  std::shared_ptr<TensorList<CPUBackend>> MakeLabels1D(
      const std::vector<std::vector<int>> &batched_labels) {
    auto tl = std::make_shared<TensorList<CPUBackend>>();
    int num_samples = batched_labels.size();
    TensorListShape<1> shape(num_samples);
    for (int i = 0; i < num_samples; i++) {
      shape.set_tensor_shape(i, {static_cast<int64_t>(batched_labels[i].size())});
    }
    tl->Resize(shape, DALI_INT32);
    for (int i = 0; i < num_samples; i++) {
      auto *data = tl->mutable_tensor<int>(i);
      for (size_t j = 0; j < batched_labels[i].size(); j++) {
        data[j] = batched_labels[i][j];
      }
    }
    return tl;
  }

  // Create 2D label input TensorList (Nx1 shape)
  std::shared_ptr<TensorList<CPUBackend>> MakeLabels2D(
      const std::vector<std::vector<int>> &batched_labels) {
    auto tl = std::make_shared<TensorList<CPUBackend>>();
    int num_samples = batched_labels.size();
    TensorListShape<2> shape(num_samples);
    for (int i = 0; i < num_samples; i++) {
      shape.set_tensor_shape(i,
          {static_cast<int64_t>(batched_labels[i].size()), 1});
    }
    tl->Resize(shape, DALI_INT32);
    for (int i = 0; i < num_samples; i++) {
      auto *data = tl->mutable_tensor<int>(i);
      for (size_t j = 0; j < batched_labels[i].size(); j++) {
        data[j] = batched_labels[i][j];
      }
    }
    return tl;
  }

  void SetupWorkspace(Workspace &ws,
                      std::shared_ptr<TensorList<CPUBackend>> boxes,
                      ThreadPool &tp) {
    ws.AddInput(boxes);
    auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
    ws.AddOutput(out_boxes);
    ws.SetBatchSizes(boxes->num_samples());
    ws.SetThreadPool(&tp);
  }

  void SetupWorkspaceWithLabels(Workspace &ws,
                                std::shared_ptr<TensorList<CPUBackend>> boxes,
                                std::shared_ptr<TensorList<CPUBackend>> labels,
                                ThreadPool &tp) {
    ws.AddInput(boxes);
    ws.AddInput(labels);
    auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
    auto out_labels = std::make_shared<TensorList<CPUBackend>>();
    ws.AddOutput(out_boxes);
    ws.AddOutput(out_labels);
    ws.SetBatchSizes(boxes->num_samples());
    ws.SetThreadPool(&tp);
  }
};

// ============================================================================
// Basic rotation with xyXY (ltrb) format
// ============================================================================

TEST_F(BBoxRotateTest, BasicLTRB_ZeroAngle) {
  // Zero rotation should produce boxes that are essentially the same
  auto boxes = MakeBoxes({{{0.1f, 0.2f, 0.8f, 0.9f}}});
  auto spec = MakeOpSpec(0.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out.num_samples(), 1);
  EXPECT_EQ(out.tensor_shape(0)[0], 1);  // 1 box preserved
}

TEST_F(BBoxRotateTest, BasicLTRB_90DegRotation) {
  auto boxes = MakeBoxes({{{0.2f, 0.3f, 0.8f, 0.7f}}});
  auto spec = MakeOpSpec(90.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, BasicLTRB_45DegRotation) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(45.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// xyWH format tests (non-ltrb path)
// ============================================================================

TEST_F(BBoxRotateTest, BasicXYWH_ZeroAngle) {
  // xyWH format: x, y, width, height
  auto boxes = MakeBoxes({{{0.1f, 0.2f, 0.7f, 0.7f}}});
  auto spec = MakeOpSpec(0.0f, {480, 640}, "expand", "xyWH");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, BasicXYWH_45DegRotation) {
  auto boxes = MakeBoxes({{{0.1f, 0.1f, 0.5f, 0.5f}}});
  auto spec = MakeOpSpec(45.0f, {480, 640}, "expand", "xyWH");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Non-normalized bbox tests (bbox_normalized=false)
// ============================================================================

TEST_F(BBoxRotateTest, NonNormalizedBboxes) {
  // Absolute pixel coordinates
  auto boxes = MakeBoxes({{{100.f, 50.f, 500.f, 400.f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "expand", "xyXY", false);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, NonNormalizedXYWH) {
  // Non-normalized + xyWH: covers the output conversion path (!ltrb || bbox_norm)
  auto boxes = MakeBoxes({{{100.f, 50.f, 400.f, 350.f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "expand", "xyWH", false);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Mode tests: expand, halfway, fixed
// ============================================================================

TEST_F(BBoxRotateTest, ModeExpand) {
  auto boxes = MakeBoxes({{{0.1f, 0.1f, 0.9f, 0.9f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "expand");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, ModeHalfway) {
  auto boxes = MakeBoxes({{{0.1f, 0.1f, 0.9f, 0.9f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "halfway");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, ModeFixed) {
  auto boxes = MakeBoxes({{{0.1f, 0.1f, 0.9f, 0.9f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "fixed");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// Fixed mode with aspect ratio flip (angle ~60-70 degrees on non-square box)
TEST_F(BBoxRotateTest, ModeFixedAspectRatioFlip) {
  // Wide box: w > h. At 60 degrees, the rotated bounding box may have w < h
  auto boxes = MakeBoxes({{{0.1f, 0.3f, 0.9f, 0.5f}}});
  auto spec = MakeOpSpec(60.0f, {480, 640}, "fixed");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, ModeHalfwayAspectRatioFlip) {
  auto boxes = MakeBoxes({{{0.1f, 0.3f, 0.9f, 0.5f}}});
  auto spec = MakeOpSpec(60.0f, {480, 640}, "halfway");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// keep_size tests
// ============================================================================

TEST_F(BBoxRotateTest, KeepSizeTrue) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(45.0f, {480, 640}, "expand", "xyXY", true, true);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Explicit size argument (non-tensor)
// ============================================================================

TEST_F(BBoxRotateTest, ExplicitSizeArg) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddArg("size", std::vector<float>{500.0f, 700.0f})
      .AddInput("bboxes", StorageDevice::CPU)
      .AddOutput("out_bboxes", StorageDevice::CPU);

  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Auto-computed canvas (no keep_size, no explicit size)
// ============================================================================

TEST_F(BBoxRotateTest, AutoComputedCanvas) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(45.0f, {480, 640}, "expand", "xyXY", true, false);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Label handling: 1D labels
// ============================================================================

TEST_F(BBoxRotateTest, WithLabels1D_NoRemoval) {
  // Boxes centered well within image, no removal expected
  auto boxes = MakeBoxes({{{0.3f, 0.3f, 0.7f, 0.7f}}});
  auto labels = MakeLabels1D({{42}});
  auto spec = MakeOpSpecWithLabels(10.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out_labels = ws.Output<CPUBackend>(1);
  ASSERT_EQ(out_labels.num_samples(), 1);
  // Box should be kept
  EXPECT_EQ(out_labels.tensor_shape(0)[0], 1);
  EXPECT_EQ(out_labels.tensor<int>(0)[0], 42);
}

TEST_F(BBoxRotateTest, WithLabels1D_WithRemoval) {
  // Box at corner of image, large rotation and strict threshold -> likely removed
  auto boxes = MakeBoxes({{{0.0f, 0.0f, 0.05f, 0.05f},
                            {0.3f, 0.3f, 0.7f, 0.7f}}});
  auto labels = MakeLabels1D({{10, 20}});
  auto spec = MakeOpSpecWithLabels(90.0f, {480, 640}, "expand", "xyXY",
                                   true, true, 0.5f);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Label handling: 2D labels (Nx1)
// ============================================================================

TEST_F(BBoxRotateTest, WithLabels2D_NoRemoval) {
  auto boxes = MakeBoxes({{{0.3f, 0.3f, 0.7f, 0.7f}}});
  auto labels = MakeLabels2D({{42}});
  auto spec = MakeOpSpecWithLabels(10.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out_labels = ws.Output<CPUBackend>(1);
  ASSERT_EQ(out_labels.num_samples(), 1);
}

TEST_F(BBoxRotateTest, WithLabels2D_WithRemoval) {
  // Edge box that may be removed + center box that stays
  auto boxes = MakeBoxes({{{0.0f, 0.0f, 0.02f, 0.02f},
                            {0.4f, 0.4f, 0.6f, 0.6f}}});
  auto labels = MakeLabels2D({{10, 20}});
  auto spec = MakeOpSpecWithLabels(90.0f, {480, 640}, "expand", "xyXY",
                                   true, true, 0.5f);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Batch processing - multiple samples
// ============================================================================

TEST_F(BBoxRotateTest, BatchMultipleSamples) {
  auto boxes = MakeBoxes({
      {{0.1f, 0.1f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.9f, 0.9f}},
      {{0.2f, 0.3f, 0.8f, 0.7f}},
      {{0.0f, 0.0f, 1.0f, 1.0f}}
  });
  auto spec = MakeOpSpec(30.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(2, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out.num_samples(), 3);
}

// ============================================================================
// Batch processing with labels
// ============================================================================

TEST_F(BBoxRotateTest, BatchWithLabels) {
  auto boxes = MakeBoxes({
      {{0.1f, 0.1f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.9f, 0.9f}},
      {{0.2f, 0.3f, 0.8f, 0.7f}}
  });
  auto labels = MakeLabels1D({{1, 2}, {3}});
  auto spec = MakeOpSpecWithLabels(10.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out = ws.Output<CPUBackend>(0);
  ASSERT_EQ(out.num_samples(), 2);
}

// ============================================================================
// Tensor argument: angle
// ============================================================================

TEST_F(BBoxRotateTest, AngleTensorArgument) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddInput("bboxes", StorageDevice::CPU)
      .AddArgumentInput("angle", "angle_input")
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  // Create angle tensor argument
  auto angle_tl = std::make_shared<TensorList<CPUBackend>>(1);
  angle_tl->Resize(TensorListShape<0>(1), DALI_FLOAT);
  angle_tl->mutable_tensor<float>(0)[0] = 45.0f;

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  ws.AddInput(boxes);
  auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
  ws.AddOutput(out_boxes);
  ws.SetBatchSizes(1);
  ws.SetThreadPool(&tp);
  ws.AddArgumentInput("angle", angle_tl);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Tensor argument: input_shape
// ============================================================================

TEST_F(BBoxRotateTest, InputShapeTensorArgument) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddInput("bboxes", StorageDevice::CPU)
      .AddArgumentInput("input_shape", "shape_input")
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  // Create input_shape tensor argument (int64)
  auto shape_tl = std::make_shared<TensorList<CPUBackend>>(1);
  TensorListShape<1> shape_tl_shape(1);
  shape_tl_shape.set_tensor_shape(0, {2});
  shape_tl->Resize(shape_tl_shape, DALI_INT64);
  auto *shape_data = shape_tl->mutable_tensor<int64_t>(0);
  shape_data[0] = 480;  // H
  shape_data[1] = 640;  // W

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  ws.AddInput(boxes);
  auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
  ws.AddOutput(out_boxes);
  ws.SetBatchSizes(1);
  ws.SetThreadPool(&tp);
  ws.AddArgumentInput("input_shape", shape_tl);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Tensor argument: size (various types to cover TYPE_SWITCH)
// ============================================================================

TEST_F(BBoxRotateTest, SizeTensorArgInt32) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddInput("bboxes", StorageDevice::CPU)
      .AddArgumentInput("size", "size_input")
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto size_tl = std::make_shared<TensorList<CPUBackend>>(1);
  TensorListShape<1> size_shape_i32(1);
  size_shape_i32.set_tensor_shape(0, {2});
  size_tl->Resize(size_shape_i32, DALI_INT32);
  auto *data = size_tl->mutable_tensor<int32_t>(0);
  data[0] = 500;  // H
  data[1] = 700;  // W

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  ws.AddInput(boxes);
  auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
  ws.AddOutput(out_boxes);
  ws.SetBatchSizes(1);
  ws.SetThreadPool(&tp);
  ws.AddArgumentInput("size", size_tl);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, SizeTensorArgInt64) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddInput("bboxes", StorageDevice::CPU)
      .AddArgumentInput("size", "size_input")
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto size_tl = std::make_shared<TensorList<CPUBackend>>(1);
  TensorListShape<1> size_shape_i64(1);
  size_shape_i64.set_tensor_shape(0, {2});
  size_tl->Resize(size_shape_i64, DALI_INT64);
  auto *data = size_tl->mutable_tensor<int64_t>(0);
  data[0] = 500;
  data[1] = 700;

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  ws.AddInput(boxes);
  auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
  ws.AddOutput(out_boxes);
  ws.SetBatchSizes(1);
  ws.SetThreadPool(&tp);
  ws.AddArgumentInput("size", size_tl);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, SizeTensorArgUInt32) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddInput("bboxes", StorageDevice::CPU)
      .AddArgumentInput("size", "size_input")
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto size_tl = std::make_shared<TensorList<CPUBackend>>(1);
  TensorListShape<1> size_shape_u32(1);
  size_shape_u32.set_tensor_shape(0, {2});
  size_tl->Resize(size_shape_u32, DALI_UINT32);
  auto *data = size_tl->mutable_tensor<uint32_t>(0);
  data[0] = 500;
  data[1] = 700;

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  ws.AddInput(boxes);
  auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
  ws.AddOutput(out_boxes);
  ws.SetBatchSizes(1);
  ws.SetThreadPool(&tp);
  ws.AddArgumentInput("size", size_tl);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, SizeTensorArgFloat) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddInput("bboxes", StorageDevice::CPU)
      .AddArgumentInput("size", "size_input")
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  auto size_tl = std::make_shared<TensorList<CPUBackend>>(1);
  TensorListShape<1> size_shape_f(1);
  size_shape_f.set_tensor_shape(0, {2});
  size_tl->Resize(size_shape_f, DALI_FLOAT);
  auto *data = size_tl->mutable_tensor<float>(0);
  data[0] = 500.5f;
  data[1] = 700.5f;

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  ws.AddInput(boxes);
  auto out_boxes = std::make_shared<TensorList<CPUBackend>>();
  ws.AddOutput(out_boxes);
  ws.SetBatchSizes(1);
  ws.SetThreadPool(&tp);
  ws.AddArgumentInput("size", size_tl);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// shape_layout tests
// ============================================================================

TEST_F(BBoxRotateTest, ShapeLayoutHWC) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640, 3}, "expand", "xyXY", true,
                         false, 0.1f, "HWC");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, ShapeLayoutWH) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(30.0f, {640, 480}, "expand", "xyXY", true,
                         false, 0.1f, "WH");
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Error paths - constructor
// ============================================================================

TEST_F(BBoxRotateTest, InvalidMode) {
  EXPECT_THROW({
    auto spec = MakeOpSpec(30.0f, {480, 640}, "invalid_mode");
    InstantiateOperator(spec);
  }, DALIException);
}

TEST_F(BBoxRotateTest, InvalidBBoxLayout) {
  EXPECT_THROW({
    auto spec = OpSpec("BBoxRotate")
        .AddArg("device", "cpu")
        .AddArg("num_threads", 1)
        .AddArg("max_batch_size", 32)
        .AddArg("angle", 30.0f)
        .AddArg("input_shape", std::vector<int>{480, 640})
        .AddArg("bbox_layout", TensorLayout("ABCD"))
        .AddArg("bbox_normalized", true)
        .AddInput("bboxes", StorageDevice::CPU)
        .AddOutput("out_bboxes", StorageDevice::CPU);
    InstantiateOperator(spec);
  }, DALIException);
}

TEST_F(BBoxRotateTest, InvalidShapeLayout) {
  EXPECT_THROW({
    auto spec = MakeOpSpec(30.0f, {480, 640}, "expand", "xyXY", true, false, 0.1f, "AB");
    InstantiateOperator(spec);
  }, DALIException);
}

TEST_F(BBoxRotateTest, InvalidRemoveThreshold) {
  EXPECT_THROW({
    auto spec = MakeOpSpec(30.0f, {480, 640}, "expand", "xyXY", true, false, 1.5f);
    InstantiateOperator(spec);
  }, DALIException);
}

// ============================================================================
// Error paths - SetupImpl
// ============================================================================

TEST_F(BBoxRotateTest, InvalidBoxShape1D) {
  // Create boxes with 1D shape instead of 2D - this should fail validation
  auto tl = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<1> shape(1);
  shape.set_tensor_shape(0, {8});
  tl->Resize(shape, DALI_FLOAT);

  auto spec = MakeOpSpec(30.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, tl, tp);

  std::vector<OutputDesc> output_desc;
  // DALI_ERROR is used in SetupImpl, which logs but may not throw in release.
  // We just exercise the code path here.
  op->Setup(output_desc, ws);
}

TEST_F(BBoxRotateTest, InvalidLabelShape) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  // Create labels with shape [N, 2] instead of [N, 1]
  auto labels = std::make_shared<TensorList<CPUBackend>>();
  TensorListShape<2> lshape(1);
  lshape.set_tensor_shape(0, {1, 2});
  labels->Resize(lshape, DALI_INT32);

  auto spec = MakeOpSpecWithLabels(30.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  EXPECT_THROW(op->Setup(output_desc, ws), DALIException);
}

TEST_F(BBoxRotateTest, LabelCountMismatch) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}, {0.3f, 0.3f, 0.7f, 0.7f}}});
  auto labels = MakeLabels1D({{1}});  // 1 label for 2 boxes

  auto spec = MakeOpSpecWithLabels(30.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspaceWithLabels(ws, boxes, labels, tp);

  std::vector<OutputDesc> output_desc;
  EXPECT_THROW(op->Setup(output_desc, ws), DALIException);
}

// ============================================================================
// Error paths - RunImpl
// ============================================================================

TEST_F(BBoxRotateTest, KeepSizeWithSizeMutuallyExclusive) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", true)
      .AddArg("remove_threshold", 0.1f)
      .AddArg("size", std::vector<float>{500.0f, 700.0f})
      .AddInput("bboxes", StorageDevice::CPU)
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

TEST_F(BBoxRotateTest, SizeListWrongLength) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});

  auto spec = OpSpec("BBoxRotate")
      .AddArg("device", "cpu")
      .AddArg("num_threads", 1)
      .AddArg("max_batch_size", 32)
      .AddArg("angle", 30.0f)
      .AddArg("input_shape", std::vector<int>{480, 640})
      .AddArg("mode", std::string("expand"))
      .AddArg("bbox_layout", TensorLayout("xyXY"))
      .AddArg("bbox_normalized", true)
      .AddArg("keep_size", false)
      .AddArg("remove_threshold", 0.1f)
      .AddArg("size", std::vector<float>{500.0f})
      .AddInput("bboxes", StorageDevice::CPU)
      .AddOutput("out_bboxes", StorageDevice::CPU);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_THROW(op->Run(ws), DALIException);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(BBoxRotateTest, EmptyBoxes) {
  // No boxes
  auto boxes = MakeBoxes({{}});
  auto spec = MakeOpSpec(30.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out = ws.Output<CPUBackend>(0);
  EXPECT_EQ(out.tensor_shape(0)[0], 0);
}

TEST_F(BBoxRotateTest, ManyBoxes) {
  // Many boxes in a single sample
  std::vector<BBox> many;
  for (int i = 0; i < 50; i++) {
    float x = static_cast<float>(i) / 100.0f;
    many.push_back({x, x, x + 0.05f, x + 0.05f});
  }
  auto boxes = MakeBoxes({many});
  auto spec = MakeOpSpec(15.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, FullRotation360) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(360.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, NegativeAngle) {
  auto boxes = MakeBoxes({{{0.2f, 0.2f, 0.8f, 0.8f}}});
  auto spec = MakeOpSpec(-45.0f, {480, 640});
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// remove_threshold = 0 means keep all boxes
TEST_F(BBoxRotateTest, RemoveThresholdZero) {
  auto boxes = MakeBoxes({{{0.0f, 0.0f, 0.01f, 0.01f}}});
  auto spec = MakeOpSpec(90.0f, {480, 640}, "expand", "xyXY", true, true, 0.0f);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));

  auto &out = ws.Output<CPUBackend>(0);
  EXPECT_GE(out.tensor_shape(0)[0], 1);  // should keep all with threshold=0
}

// remove_threshold near 1 means remove if almost any part is outside
TEST_F(BBoxRotateTest, RemoveThresholdHigh) {
  auto boxes = MakeBoxes({{{0.0f, 0.0f, 0.1f, 0.1f}}});
  auto spec = MakeOpSpec(45.0f, {480, 640}, "expand", "xyXY", true, true, 0.99f);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

// ============================================================================
// Combined coverage: normalized xywh + modes
// ============================================================================

TEST_F(BBoxRotateTest, NormalizedXYWH_ModeFixed) {
  auto boxes = MakeBoxes({{{0.1f, 0.1f, 0.5f, 0.5f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "fixed", "xyWH", true);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

TEST_F(BBoxRotateTest, NormalizedXYWH_ModeHalfway) {
  auto boxes = MakeBoxes({{{0.1f, 0.1f, 0.5f, 0.5f}}});
  auto spec = MakeOpSpec(30.0f, {480, 640}, "halfway", "xyWH", true);
  auto op = InstantiateOperator(spec);

  Workspace ws;
  ThreadPool tp(1, 0, false, "TestPool");
  SetupWorkspace(ws, boxes, tp);

  std::vector<OutputDesc> output_desc;
  ASSERT_FALSE(op->Setup(output_desc, ws));
  EXPECT_NO_THROW(op->Run(ws));
}

}  // namespace testing
}  // namespace dali
