// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/util/copy_with_stride.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include "dali/core/dev_buffer.h"
#include "dali/pipeline/data/dltensor.h"

namespace dali {

TEST(CopyWithStrideTest, OneDim) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T data[] = {1, 2, 3, 4, 5, 6};
  TensorShape<1> stride{2};
  TensorShape<1> shape{3};
  constexpr int vol = 3;
  ASSERT_EQ(vol, volume(shape));
  std::array<T, vol> out;
  auto dl_tensor = MakeDLTensor(data, dtype, false, false, -1, shape, stride);
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, vol>{1, 3, 5}));
}

TEST(CopyWithStrideTest, TwoDims)  {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T data[] = {11, 12, 13, 14,
              21, 22, 23, 24,
              31, 32, 33, 34,
              41, 42, 43, 44};
  TensorShape<2> stride{8, 1};
  TensorShape<2> shape{2, 4};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  std::array<T, vol> out;
  auto dl_tensor = MakeDLTensor(data, dtype, false, false, -1, shape, stride);
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, vol>{11, 12, 13, 14,
                                         31, 32, 33, 34}));
}

TEST(CopyWithStrideTest, SimpleCopy) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T data[] = {1, 2,
              3, 4,

              5, 6,
              7, 8};
  TensorShape<3>  stride{4, 2, 1};
  TensorShape<3> shape{2, 2, 2};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  std::array<T, vol> out;
  auto dl_tensor = MakeDLTensor(data, dtype, false, false, -1, shape, stride);
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, vol>{1, 2,
                                         3, 4,

                                         5, 6,
                                         7, 8}));
}

// CPU tests to cover CopyVecStatic and CopyVec functions

// Test item_size=1 (covers VALUE_SWITCH case 1)
TEST(CopyWithStrideTest, CPUCopy1Byte) {
  using T = uint8_t;
  T data[] = {1, 0, 2, 0, 3, 0, 4, 0};
  TensorShape<1> stride{2};  // Every other byte
  TensorShape<1> shape{4};
  std::array<T, 4> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 4>{1, 2, 3, 4}));
}

// Test item_size=2 (covers VALUE_SWITCH case 2)
TEST(CopyWithStrideTest, CPUCopy2Bytes) {
  using T = uint16_t;
  T data[] = {1, 999, 2, 999, 3, 999};
  TensorShape<1> stride{2};
  TensorShape<1> shape{3};
  std::array<T, 3> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT16, false, false, -1, shape, stride);
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 3>{1, 2, 3}));
}

// Test item_size=3 (covers VALUE_SWITCH case 3) - RGB triplet
TEST(CopyWithStrideTest, CPUCopy3Bytes) {
  using T = uint8_t;
  T data[] = {
    1, 2, 3,     // RGB pixel 0
    0, 0, 0,     // Padding
    4, 5, 6,     // RGB pixel 1
    0, 0, 0,     // Padding
    7, 8, 9      // RGB pixel 2
  };
  TensorShape<1> stride{2};  // 2 elements * 3 bytes = 6 bytes
  TensorShape<1> shape{3};
  std::array<T, 9> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 24;  // 3 bytes
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Test item_size=5 (covers VALUE_SWITCH case 5)
TEST(CopyWithStrideTest, CPUCopy5Bytes) {
  using T = uint8_t;
  T data[] = {
    1, 2, 3, 4, 5,
    0, 0, 0, 0, 0,
    6, 7, 8, 9, 10
  };
  TensorShape<1> stride{2};  // 2 * 5 = 10 bytes
  TensorShape<1> shape{2};
  std::array<T, 10> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 40;  // 5 bytes
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 10>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

// Test item_size=6 (covers VALUE_SWITCH case 6)
TEST(CopyWithStrideTest, CPUCopy6Bytes) {
  using T = uint8_t;
  T data[] = {
    1, 2, 3, 4, 5, 6,
    0, 0, 0, 0, 0, 0,
    7, 8, 9, 10, 11, 12
  };
  TensorShape<1> stride{2};
  TensorShape<1> shape{2};
  std::array<T, 12> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 48;  // 6 bytes
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

// Test item_size=7 (covers VALUE_SWITCH case 7)
TEST(CopyWithStrideTest, CPUCopy7Bytes) {
  using T = uint8_t;
  T data[] = {
    1, 2, 3, 4, 5, 6, 7,
    0, 0, 0, 0, 0, 0, 0,
    8, 9, 10, 11, 12, 13, 14
  };
  TensorShape<1> stride{2};
  TensorShape<1> shape{2};
  std::array<T, 14> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 56;  // 7 bytes
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 14>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}));
}

// Test item_size=8 (covers VALUE_SWITCH case 8)
TEST(CopyWithStrideTest, CPUCopy8Bytes) {
  using T = uint64_t;
  T data[] = {1, 999, 2, 999, 3};
  TensorShape<1> stride{2};
  TensorShape<1> shape{2};
  std::array<T, 2> out;
  auto dl_tensor = MakeDLTensor(data, DALI_INT64, false, false, -1, shape, stride);
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 2>{1, 2}));
}

// Test item_size=12 (covers VALUE_SWITCH case 12) - 3x float32
TEST(CopyWithStrideTest, CPUCopy12Bytes) {
  using T = uint8_t;
  T data[36];
  for (int i = 0; i < 12; i++) data[i] = i + 1;
  for (int i = 12; i < 24; i++) data[i] = 0;  // Padding
  for (int i = 24; i < 36; i++) data[i] = i - 23;  // Second element: 1-12
  TensorShape<1> stride{2};  // 2 * 12 = 24 bytes
  TensorShape<1> shape{2};
  std::array<T, 24> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 96;  // 12 bytes
  CopyDlTensorCpu(out.data(), dl_tensor);
  for (int i = 0; i < 12; i++) ASSERT_EQ(out[i], i + 1);
  for (int i = 12; i < 24; i++) ASSERT_EQ(out[i], i - 11);  // Should be 1-12
}

// Test item_size=16 (covers VALUE_SWITCH case 16) - 2x int64 or 4x float32
TEST(CopyWithStrideTest, CPUCopy16Bytes) {
  using T = uint8_t;
  T data[48];
  for (int i = 0; i < 16; i++) data[i] = i + 1;
  for (int i = 16; i < 32; i++) data[i] = 0;  // Padding
  for (int i = 32; i < 48; i++) data[i] = i - 31;  // Second element: 1-16
  TensorShape<1> stride{2};  // 2 * 16 = 32 bytes
  TensorShape<1> shape{2};
  std::array<T, 32> out;
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 128;  // 16 bytes
  CopyDlTensorCpu(out.data(), dl_tensor);
  for (int i = 0; i < 16; i++) ASSERT_EQ(out[i], i + 1);
  for (int i = 16; i < 32; i++) ASSERT_EQ(out[i], i - 15);  // Should be 1-16
}

// Test CopyVec with unusual bit width (not in VALUE_SWITCH: 1,2,3,4,5,6,7,8,12,16)
// This covers lines 23-30 in copy_with_stride.cc.vcast.bak and the default case
TEST(CopyWithStrideTest, UnusualBitWidth72Bits) {
  // Create data with 72 bits (9 bytes) per element - forces CopyVec path
  using T = uint8_t;
  constexpr int item_size = 9;  // 72 bits, not in VALUE_SWITCH
  // Data: 3 elements, each 9 bytes, with stride of 18 bytes (skip every other)
  T data[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9,        // Element 0
    0, 0, 0, 0, 0, 0, 0, 0, 0,        // Padding (skipped)
    10, 11, 12, 13, 14, 15, 16, 17, 18,  // Element 1
    0, 0, 0, 0, 0, 0, 0, 0, 0,        // Padding (skipped)
    19, 20, 21, 22, 23, 24, 25, 26, 27   // Element 2
  };

  TensorShape<1> stride{2};  // Stride in elements (2 * 9 bytes = 18 bytes)
  TensorShape<1> shape{3};   // 3 elements
  constexpr int vol = 3 * item_size;
  std::array<T, vol> out;

  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  // Manually set unusual bit width
  dl_tensor->dl_tensor.dtype.bits = 72;  // 9 bytes, not in VALUE_SWITCH

  CopyDlTensorCpu(out.data(), dl_tensor);

  // Verify output
  std::array<T, vol> expected = {
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27
  };
  ASSERT_TRUE(out == expected);
}

// Test CopyVec with 10 bytes (80 bits) - another unusual size
TEST(CopyWithStrideTest, UnusualBitWidth80Bits) {
  using T = uint8_t;
  constexpr int item_size = 10;  // 80 bits, not in VALUE_SWITCH
  T data[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20
  };

  TensorShape<1> stride{2};  // 2 * 10 bytes = 20 bytes
  TensorShape<1> shape{2};
  constexpr int vol = 2 * item_size;
  std::array<T, vol> out;

  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 80;  // 10 bytes

  CopyDlTensorCpu(out.data(), dl_tensor);

  std::array<T, vol> expected = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20
  };
  ASSERT_TRUE(out == expected);
}

// Test CopyVec with 24 bytes (192 bits) - larger than VALUE_SWITCH max (16)
TEST(CopyWithStrideTest, UnusualBitWidth192Bits) {
  using T = uint8_t;
  constexpr int item_size = 24;  // 192 bits, larger than VALUE_SWITCH max
  T data[item_size * 2];
  for (int i = 0; i < item_size * 2; i++) {
    data[i] = i + 1;
  }

  TensorShape<1> stride{1};
  TensorShape<1> shape{2};
  constexpr int vol = 2 * item_size;
  std::array<T, vol> out;

  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 192;  // 24 bytes

  CopyDlTensorCpu(out.data(), dl_tensor);

  // Verify all bytes copied correctly
  for (int i = 0; i < vol; i++) {
    ASSERT_EQ(out[i], static_cast<T>(i + 1));
  }
}

// Test nullptr strides path in CopyWithStrideCpu (lines 90-92 in .cc.vcast.bak)
TEST(CopyWithStrideTest, CPUCopyNullStrides) {
  using T = float;
  T data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TensorShape<2> shape{2, 3};
  std::array<T, 6> out;

  // Create DLTensor without strides (compact tensor)
  auto dl_tensor = MakeDLTensor(data, DALI_FLOAT, false, false, -1, shape);
  // Verify strides is nullptr
  ASSERT_EQ(dl_tensor->dl_tensor.strides, nullptr);

  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 6>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Test another nullptr strides case with different data type
TEST(CopyWithStrideTest, CPUCopyNullStridesInt64) {
  using T = int64_t;
  T data[] = {10, 20, 30, 40};
  TensorShape<1> shape{4};
  std::array<T, 4> out;

  auto dl_tensor = MakeDLTensor(data, DALI_INT64, false, false, -1, shape);
  ASSERT_EQ(dl_tensor->dl_tensor.strides, nullptr);

  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, 4>{10, 20, 30, 40}));
}

// Test nullptr strides with uint8
TEST(CopyWithStrideTest, CPUCopyNullStridesUint8) {
  using T = uint8_t;
  T data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TensorShape<3> shape{2, 2, 3};
  std::array<T, 12> out;

  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape);
  ASSERT_EQ(dl_tensor->dl_tensor.strides, nullptr);

  CopyDlTensorCpu(out.data(), dl_tensor);
  for (int i = 0; i < 12; i++) {
    ASSERT_EQ(out[i], data[i]);
  }
}

// Test CopyVec with another unusual bit width (13 bytes = 104 bits)
TEST(CopyWithStrideTest, UnusualBitWidth104Bits) {
  using T = uint8_t;
  constexpr int item_size = 13;  // 104 bits, not in VALUE_SWITCH
  // Stride of 2 elements * 13 bytes = 26 bytes between elements
  T data[] = {
    // Element 0: bytes 0-12
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Padding to byte 26
    // Element 1: bytes 26-38
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Padding to byte 52
    // Element 2: bytes 52-64
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
  };

  TensorShape<1> stride{2};  // 2 * 13 bytes = 26 bytes
  TensorShape<1> shape{3};
  constexpr int vol = 3 * item_size;
  std::array<T, vol> out;

  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 104;  // 13 bytes

  CopyDlTensorCpu(out.data(), dl_tensor);

  std::array<T, vol> expected = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
  };
  ASSERT_TRUE(out == expected);
}

// Test CopyVec with 20 bytes (160 bits) - covers lines 23-30 with larger size
TEST(CopyWithStrideTest, UnusualBitWidth160Bits) {
  using T = uint8_t;
  constexpr int item_size = 20;  // 160 bits, not in VALUE_SWITCH
  T data[item_size * 4];
  for (int i = 0; i < item_size * 4; i++) {
    data[i] = static_cast<T>(i % 256);
  }

  TensorShape<1> stride{1};
  TensorShape<1> shape{4};
  constexpr int vol = 4 * item_size;
  std::array<T, vol> out;

  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 160;  // 20 bytes

  CopyDlTensorCpu(out.data(), dl_tensor);

  // Verify all bytes copied correctly
  for (int i = 0; i < vol; i++) {
    ASSERT_EQ(out[i], static_cast<T>(i % 256));
  }
}

DLMTensorPtr AsDlTensor(void* data, DALIDataType dtype, TensorShape<> shape, TensorShape<> stride) {
  return MakeDLTensor(data, dtype, false, false, -1, shape, stride);
}

std::vector<DLMTensorPtr> DlTensorSingletonBatch(DLMTensorPtr dl_tensor) {
  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(std::move(dl_tensor));
  return dl_tensors;
}

TensorList<GPUBackend> SingletonTL(TensorShape<> shape, DALIDataType dtype) {
  TensorList<GPUBackend> output;
  TensorListShape tls(1, shape.sample_dim());
  tls.set_tensor_shape(0, shape);
  output.Resize(tls, dtype);
  return output;
}

TEST(CopyWithStrideTest, OneDimGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1, 2, 3, 4, 5, 6};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<1> stride{2};
  TensorShape<1> shape{3};
  constexpr int vol = 3;
  ASSERT_EQ(vol, volume(shape));
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5}));
}

TEST(CopyWithStrideTest, TwoDimsGPU) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[] = {11, 12, 13, 14,
                21, 22, 23, 24,
                31, 32, 33, 34,
                41, 42, 43, 44};
  TensorShape<2> stride{8, 1};
  TensorShape<2> shape{2, 4};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<int64_t> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{11, 12, 13, 14, 31, 32, 33, 34}));
}

TEST(CopyWithStrideTest, TwoDimsGPUOdd) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[] = {1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
                26, 27, 28, 29, 30};
  TensorShape<2> stride{15, 1};
  TensorShape<2> shape{2, 4};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 2, 3, 4, 16, 17, 18, 19}));
}

TEST(CopyWithStrideTest, TwoDimsInnerStride) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[] = {1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
                26, 27, 28, 29, 30};
  TensorShape<2> stride{15, 5};
  TensorShape<2> shape{2, 3};
  constexpr int vol = 6;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 6, 11, 16, 21, 26}));
}

TEST(CopyWithStrideTest, TwoDimsTransposed) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[] = {1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
                26, 27, 28, 29, 30};
  TensorShape<2> stride{1, 5};
  TensorShape<2> shape{5, 6};
  constexpr int vol = 30;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  std::array<T, vol> ref = {
    1,  6, 11, 16, 21, 26,
    2,  7, 12, 17, 22, 27,
    3,  8, 13, 18, 23, 28,
    4,  9, 14, 19, 24, 29,
    5, 10, 15, 20, 25, 30};
  ASSERT_TRUE(h_out == ref);
}

TEST(CopyWithStrideTest, SimpleCopyGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1,  2,  3,
                4,  5,  6,

                7,  8,  9,
                10, 11, 12};
  TensorShape<3> stride{6, 3, 1};
  TensorShape<3> shape{2, 2, 3};
  constexpr int vol = 12;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1,  2,  3,
                                           4,  5,  6,

                                           7,  8,  9,
                                           10, 11, 12}));
}

// Test empty batch (covers line 349-350)
TEST(CopyWithStrideTest, EmptyBatchGPU) {
  std::vector<DLMTensorPtr> empty_tensors;
  TensorList<GPUBackend> output;
  // Should return early without error
  CopyDlTensorBatchGpu(output, empty_tensors, 0);
}

// Test non-strided tensor (covers lines 368-370)
TEST(CopyWithStrideTest, NonStridedTensorGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1, 2, 3, 4, 5, 6};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<1> shape{6};
  // Create DLTensor without strides (nullptr for compact/dense tensor)
  auto dl_tensor = MakeDLTensor(data, dtype, false, false, -1, shape);
  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(std::move(dl_tensor));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, 6> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), 6 * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, 6>{1, 2, 3, 4, 5, 6}));
}

// Test empty tensor (covers lines 377-378)
TEST(CopyWithStrideTest, EmptyTensorGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  DeviceBuffer<T> data;
  TensorShape<1> stride{1};
  TensorShape<1> shape{0};  // Empty tensor
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  // Should handle empty tensor gracefully
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
}

// Test zero-dimensional tensor (covers line 384 false branch)
TEST(CopyWithStrideTest, ZeroDimTensorGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {42.0f};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<0> stride{};
  TensorShape<0> shape{};  // Scalar (0-D tensor)
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  T h_out;
  CUDA_CALL(cudaMemcpy(&h_out, output_tl.raw_mutable_tensor(0), sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out, 42.0f);
}

// Test 3D strided tensor with uint8 (covers NDim=3 case for ElementSize=1)
TEST(CopyWithStrideTest, ThreeDimsGPU) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<3> stride{8, 4, 2};  // Strided in all dimensions
  TensorShape<3> shape{2, 2, 2};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5, 7, 9, 11, 13, 15}));
}

// Test 4D strided tensor (covers NDim=4 case)
TEST(CopyWithStrideTest, FourDimsGPU) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[32];
  for (int i = 0; i < 32; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<4> stride{16, 8, 4, 2};  // Strided in all dimensions
  TensorShape<4> shape{2, 2, 2, 2};
  constexpr int vol = 16;
  ASSERT_EQ(vol, volume(shape));
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}));
}

// Test 5D strided tensor (covers NDim=5 case)
TEST(CopyWithStrideTest, FiveDimsGPU) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[64];
  for (int i = 0; i < 64; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<5> stride{32, 16, 8, 4, 2};  // Strided in all dimensions
  TensorShape<5> shape{2, 2, 2, 2, 2};
  constexpr int vol = 32;
  ASSERT_EQ(vol, volume(shape));
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  // Verify first few elements
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
  ASSERT_EQ(h_out[2], 5);
}

// Test 6D strided tensor (covers default case with MismatchedNdim<-1> and line 111)
TEST(CopyWithStrideTest, SixDimsGPU) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[128];
  for (int i = 0; i < 128; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<6> stride{64, 32, 16, 8, 4, 2};  // Strided in all dimensions
  TensorShape<6> shape{2, 2, 2, 2, 2, 2};
  constexpr int vol = 64;
  ASSERT_EQ(vol, volume(shape));
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  // Verify first few elements
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
  ASSERT_EQ(h_out[2], 5);
}

// Test 3D tensor with uint16 (covers NDim=3 for ElementSize=2)
TEST(CopyWithStrideTest, ThreeDimsUint16GPU) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[16];
  for (int i = 0; i < 16; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<3> stride{8, 4, 2};
  TensorShape<3> shape{2, 2, 2};
  constexpr int vol = 8;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5, 7, 9, 11, 13, 15}));
}

// Test 2D tensor with float (covers NDim=2 for ElementSize=4)
TEST(CopyWithStrideTest, TwoDimsFloat32GPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<2> stride{4, 2};
  TensorShape<2> shape{2, 2};
  constexpr int vol = 4;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1.0f, 3.0f, 5.0f, 7.0f}));
}

// Test 2D tensor with int64 (covers NDim=2 for ElementSize=8)
TEST(CopyWithStrideTest, TwoDimsInt64GPU) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[] = {10, 20, 30, 40, 50, 60, 70, 80};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<2> stride{4, 2};
  TensorShape<2> shape{2, 2};
  constexpr int vol = 4;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{10, 30, 50, 70}));
}

// Error validation tests

// Test invalid lanes (covers lines 302-304)
TEST(CopyWithStrideTest, InvalidLanesError) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1.0f, 2.0f};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<1> shape{2};
  TensorShape<1> stride{1};

  // Manually create DLTensor with invalid lanes
  auto dl_tensor = MakeDLTensor(data, dtype, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.lanes = 2;  // Invalid: DALI requires lanes == 1

  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(std::move(dl_tensor));
  auto output_tl = SingletonTL(shape, dtype);

  ASSERT_THROW(CopyDlTensorBatchGpu(output_tl, dl_tensors, 0), DALIException);
}

// Test type mismatch in batch (covers lines 309-310)
TEST(CopyWithStrideTest, TypeMismatchError) {
  using T1 = float;
  using T2 = int64_t;
  T1 h_data1[] = {1.0f, 2.0f};
  T2 h_data2[] = {3, 4};
  DeviceBuffer<T1> data1;
  DeviceBuffer<T2> data2;
  data1.from_host(h_data1);
  data2.from_host(h_data2);

  TensorShape<1> shape{2};
  TensorShape<1> stride{1};

  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(MakeDLTensor(data1, DALI_FLOAT, false, false, -1, shape, stride));
  dl_tensors.push_back(MakeDLTensor(data2, DALI_INT64, false, false, -1, shape, stride));

  TensorList<GPUBackend> output;
  TensorListShape tls(2, 1);
  tls.set_tensor_shape(0, shape);
  tls.set_tensor_shape(1, shape);
  output.Resize(tls, DALI_FLOAT);

  ASSERT_THROW(CopyDlTensorBatchGpu(output, dl_tensors, 0), DALIException);
}

// Test ndim mismatch in batch (covers lines 312-316)
TEST(CopyWithStrideTest, NdimMismatchError) {
  using T = float;
  T h_data1[] = {1.0f, 2.0f};
  T h_data2[] = {3.0f, 4.0f, 5.0f, 6.0f};
  DeviceBuffer<T> data1;
  DeviceBuffer<T> data2;
  data1.from_host(h_data1);
  data2.from_host(h_data2);

  TensorShape<1> shape1{2};
  TensorShape<2> shape2{2, 2};
  TensorShape<1> stride1{1};
  TensorShape<2> stride2{2, 1};

  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(MakeDLTensor(data1, DALI_FLOAT, false, false, -1, shape1, stride1));
  dl_tensors.push_back(MakeDLTensor(data2, DALI_FLOAT, false, false, -1, shape2, stride2));

  TensorList<GPUBackend> output;
  TensorListShape tls(2, 1);
  tls.set_tensor_shape(0, shape1);
  tls.set_tensor_shape(1, shape2);
  output.Resize(tls, DALI_FLOAT);

  ASSERT_THROW(CopyDlTensorBatchGpu(output, dl_tensors, 0), DALIException);
}

// Test unsupported bit width (covers lines 320-322)
TEST(CopyWithStrideTest, UnsupportedBitWidthError) {
  using T = uint8_t;
  T h_data[] = {1, 2};
  DeviceBuffer<T> data;
  data.from_host(h_data);

  TensorShape<1> shape{2};
  TensorShape<1> stride{1};

  // Manually create DLTensor with unsupported bit width
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.dtype.bits = 7;  // Invalid: only 8, 16, 32, 64 supported

  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(std::move(dl_tensor));
  auto output_tl = SingletonTL(shape, DALI_UINT8);

  ASSERT_THROW(CopyDlTensorBatchGpu(output_tl, dl_tensors, 0), DALIException);
}

// Test ndim out of range (covers lines 323-325)
TEST(CopyWithStrideTest, NdimOutOfRangeError) {
  using T = uint8_t;
  T h_data[] = {1, 2};
  DeviceBuffer<T> data;
  data.from_host(h_data);

  TensorShape<1> shape{2};
  TensorShape<1> stride{1};

  // Manually create DLTensor with invalid ndim
  auto dl_tensor = MakeDLTensor(data, DALI_UINT8, false, false, -1, shape, stride);
  dl_tensor->dl_tensor.ndim = 16;  // Invalid: MAX_DIMS is 15

  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(std::move(dl_tensor));
  auto output_tl = SingletonTL(shape, DALI_UINT8);

  ASSERT_THROW(CopyDlTensorBatchGpu(output_tl, dl_tensors, 0), DALIException);
}

// Rare stride pattern combinations to hit specific NDim cases in VALUE_SWITCH

// ElementSize=2 (uint16), NDim=1: stride pattern with 1 mismatch
TEST(CopyWithStrideTest, Uint16OneDimStrided) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[] = {1, 10, 2, 20, 3, 30, 4, 40};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<1> stride{2};  // Every other element
  TensorShape<1> shape{4};
  constexpr int vol = 4;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 2, 3, 4}));
}

// ElementSize=2 (uint16), NDim=4: all dimensions mismatched
TEST(CopyWithStrideTest, Uint16FourDimsStrided) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[32];
  for (int i = 0; i < 32; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<4> stride{16, 8, 4, 2};  // All dimensions have stride > compact
  TensorShape<4> shape{2, 2, 2, 2};
  constexpr int vol = 16;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}));
}

// ElementSize=2 (uint16), NDim=5: all dimensions mismatched
TEST(CopyWithStrideTest, Uint16FiveDimsStrided) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[64];
  for (int i = 0; i < 64; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<5> stride{32, 16, 8, 4, 2};  // All dimensions have stride > compact
  TensorShape<5> shape{2, 2, 2, 2, 2};
  constexpr int vol = 32;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  // Verify first few elements
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
  ASSERT_EQ(h_out[2], 5);
}

// ElementSize=2 (uint16), default case (NDim>=6): 6D tensor
TEST(CopyWithStrideTest, Uint16SixDimsStrided) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[128];
  for (int i = 0; i < 128; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<6> stride{64, 32, 16, 8, 4, 2};  // All dimensions strided
  TensorShape<6> shape{2, 2, 2, 2, 2, 2};
  constexpr int vol = 64;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
}

// ElementSize=4 (float), NDim=3: all dimensions mismatched
TEST(CopyWithStrideTest, Float32ThreeDimsStrided) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[16];
  for (int i = 0; i < 16; i++) h_data[i] = static_cast<T>(i + 1);
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<3> stride{8, 4, 2};  // All dimensions strided
  TensorShape<3> shape{2, 2, 2};
  constexpr int vol = 8;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f}));
}

// ElementSize=4 (float), NDim=4: all dimensions mismatched
TEST(CopyWithStrideTest, Float32FourDimsStrided) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[32];
  for (int i = 0; i < 32; i++) h_data[i] = static_cast<T>(i + 1);
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<4> stride{16, 8, 4, 2};  // All dimensions strided
  TensorShape<4> shape{2, 2, 2, 2};
  constexpr int vol = 16;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1.0f);
  ASSERT_EQ(h_out[1], 3.0f);
}

// ElementSize=4 (float), NDim=5: all dimensions mismatched
TEST(CopyWithStrideTest, Float32FiveDimsStrided) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[64];
  for (int i = 0; i < 64; i++) h_data[i] = static_cast<T>(i + 1);
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<5> stride{32, 16, 8, 4, 2};  // All dimensions strided
  TensorShape<5> shape{2, 2, 2, 2, 2};
  constexpr int vol = 32;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1.0f);
  ASSERT_EQ(h_out[1], 3.0f);
}

// ElementSize=4 (float), default case (NDim>=6): 7D tensor to trigger default
TEST(CopyWithStrideTest, Float32SevenDimsStrided) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[256];
  for (int i = 0; i < 256; i++) h_data[i] = static_cast<T>(i + 1);
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<7> stride{128, 64, 32, 16, 8, 4, 2};  // All dimensions strided
  TensorShape<7> shape{2, 2, 2, 2, 2, 2, 2};
  constexpr int vol = 128;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1.0f);
  ASSERT_EQ(h_out[1], 3.0f);
}

// ElementSize=8 (int64), NDim=3: all dimensions mismatched
TEST(CopyWithStrideTest, Int64ThreeDimsStrided) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[16];
  for (int i = 0; i < 16; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<3> stride{8, 4, 2};  // All dimensions strided
  TensorShape<3> shape{2, 2, 2};
  constexpr int vol = 8;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5, 7, 9, 11, 13, 15}));
}

// ElementSize=8 (int64), NDim=4: all dimensions mismatched
TEST(CopyWithStrideTest, Int64FourDimsStrided) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[32];
  for (int i = 0; i < 32; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<4> stride{16, 8, 4, 2};  // All dimensions strided
  TensorShape<4> shape{2, 2, 2, 2};
  constexpr int vol = 16;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
}

// ElementSize=8 (int64), NDim=5: all dimensions mismatched
TEST(CopyWithStrideTest, Int64FiveDimsStrided) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[64];
  for (int i = 0; i < 64; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<5> stride{32, 16, 8, 4, 2};  // All dimensions strided
  TensorShape<5> shape{2, 2, 2, 2, 2};
  constexpr int vol = 32;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
}

// ElementSize=8 (int64), default case (NDim>=6): 8D tensor to trigger default
TEST(CopyWithStrideTest, Int64EightDimsStrided) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[512];
  for (int i = 0; i < 512; i++) h_data[i] = i + 1;
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<8> stride{256, 128, 64, 32, 16, 8, 4, 2};  // All dimensions strided
  TensorShape<8> shape{2, 2, 2, 2, 2, 2, 2, 2};
  constexpr int vol = 256;
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 3);
}

}  // namespace dali
