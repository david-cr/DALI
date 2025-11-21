// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <thread>
#include <utility>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/dltensor.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

TEST(DLPackTest, DLType) {
  DLDataType dl;
  for (DALIDataType dali : {
      DALI_BOOL,
      DALI_FLOAT16, DALI_FLOAT, DALI_FLOAT64,
      DALI_INT8, DALI_UINT8,
      DALI_INT16, DALI_UINT16,
      DALI_INT32, DALI_UINT32,
      DALI_INT64, DALI_UINT64 }) {
    dl = ToDLType(dali);
    TypeInfo info = TypeTable::GetTypeInfo(dali);
    EXPECT_EQ(dl.lanes, 1);
    EXPECT_EQ(dl.bits, info.size() * 8);
    if (info.name().find("uint") == 0) {
      EXPECT_EQ(dl.code, kDLUInt);
    } else if (info.name().find("int") == 0) {
      EXPECT_EQ(dl.code, kDLInt);
    } else if (info.name().find("float") == 0) {
      EXPECT_EQ(dl.code, kDLFloat);
    } else if (info.name().find("bool") == 0) {
      EXPECT_EQ(dl.code, kDLBool);
    }

    EXPECT_EQ(ToDALIType(dl), dali) << "Conversion back to DALI type yielded a different type.";
  }
}

TEST(DLPackTest, DLTypeToString) {
  EXPECT_EQ(to_string(DLDataType{ kDLBool, 8, 1 }), "b8");
  EXPECT_EQ(to_string(DLDataType{ kDLBfloat, 16, 1 }), "bf16");
  EXPECT_EQ(to_string(DLDataType{ kDLFloat, 32, 4 }), "f32x4");
  EXPECT_EQ(to_string(DLDataType{ kDLUInt, 16, 2 }), "u16x2");
  EXPECT_EQ(to_string(DLDataType{ kDLInt, 64, 1 }), "i64");
  EXPECT_EQ(to_string(DLDataType{ 123, 8, 16 }), "<unknown:123>8x16");
}

TEST(DLPackTest, ToDALITypeInvalidLanes) {
  DLDataType dl_type{kDLFloat, 32, 2};
  EXPECT_THROW(ToDALIType(dl_type), DALIException);
}

TEST(DLPackTest, ToDALITypeUnsupportedCode) {
  DLDataType dl_type{123, 32, 1};
  EXPECT_THROW(ToDALIType(dl_type), DALIException);
}

TEST(DLPackTest, ToDALITypeUnsupportedBits) {
  DLDataType dl_type{kDLFloat, 128, 1};
  EXPECT_THROW(ToDALIType(dl_type), DALIException);

  dl_type = {kDLInt, 128, 1};
  EXPECT_THROW(ToDALIType(dl_type), DALIException);

  dl_type = {kDLUInt, 128, 1};
  EXPECT_THROW(ToDALIType(dl_type), DALIException);
}

namespace {

void TestSampleViewCPU(bool pinned) {
  Tensor<CPUBackend> tensor;
  tensor.set_pinned(pinned);
  tensor.set_device_id(0);
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  SampleView<CPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
  DLMTensorPtr dlm_tensor = GetDLTensorView(sv, tensor.is_pinned(), tensor.device_id());
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.data, sv.raw_data());
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, pinned ? kDLCUDAHost : kDLCPU);
  EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

}  // namespace

TEST(DLMTensorPtr, ViewCPU) {
  TestSampleViewCPU(false);
}

TEST(DLMTensorPtr, ViewPinnedCPU) {
  TestSampleViewCPU(true);
}

TEST(DLMTensorPtr, CPUShared) {
  Tensor<CPUBackend> tensor;
  tensor.set_pinned(false);
  tensor.set_device_id(0);
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_EQ(tensor.get_data_ptr().use_count(), 2) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCPU);
    EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
  }
  EXPECT_EQ(tensor.get_data_ptr().use_count(), 1) << "Reference leaked.";
}

TEST(DLMTensorPtr, ViewGPU) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({100, 50, 1}, DALI_INT32);
  SampleView<GPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
  DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, tensor.device_id());
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensor->dl_tensor.data, sv.raw_data());
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int) * 8);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_id, tensor.device_id());
  EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
}

TEST(DLMTensorPtr, CPUList) {
  TensorList<CPUBackend> tlist;
  tlist.set_pinned(false);
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_FLOAT64);
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);

  EXPECT_EQ(tlist.tensor_shape(1).size(), 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
}


TEST(DLMTensorPtr, CPUSharedList) {
  TensorList<CPUBackend> tlist;
  tlist.set_pinned(false);
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_FLOAT64);
  const auto &ptr = unsafe_owner(tlist);
  EXPECT_EQ(ptr.use_count(), 3);
  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_EQ(ptr.use_count(), 5);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);

  EXPECT_EQ(tlist.tensor_shape(1).size(), 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(double) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
  dlm_tensors.clear();
  EXPECT_EQ(ptr.use_count(), 3);
}

TEST(DLMTensorPtr, GPUList) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_UINT8);
  std::vector<DLMTensorPtr> dlm_tensors = GetDLTensorListView(tlist);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLUInt);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_id, tlist.device_id());

  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLUInt);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_id, tlist.device_id());
}

struct TestDLPayload {
  explicit TestDLPayload(bool &destroyed)
  : destroyed(destroyed) {}

  ~TestDLPayload() {
    destroyed = true;
  }

  bool &destroyed;
};


TEST(DLMTensorPtr, Cleanup) {
  bool deleter_called = false;
  {
    auto rsrc = DLTensorResource<TestDLPayload>::Create(deleter_called);
    auto dlm_tensor = ToDLMTensor(std::move(rsrc));
    EXPECT_EQ(rsrc, nullptr);
  }
  EXPECT_TRUE(deleter_called);
}

TEST(DLMTensorPtr, DeleterWithNullptr) {
  DLMTensorPtrDeleter(nullptr);
}

TEST(DLMTensorPtr, GPUShared) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({100, 50, 3}, DALI_FLOAT);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 100);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 50);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(float) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_id, tensor.device_id());
    EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST(DLMTensorPtr, GPUSharedList) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{100, 50, 1}, {50, 30, 3}}, DALI_UINT8);
  const auto &ptr = unsafe_owner(tlist);
  int initial_count = ptr.use_count();
  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_GT(ptr.use_count(), initial_count) << "Reference count not increased";
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[2], 1);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.data, tlist.raw_tensor(0));
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLUInt);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.device.device_id, tlist.device_id());
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.byte_offset, 0);

  EXPECT_EQ(dlm_tensors[1]->dl_tensor.ndim, 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 30);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.data, tlist.raw_tensor(1));
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLUInt);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(uint8_t) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.device.device_id, tlist.device_id());
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.byte_offset, 0);
  dlm_tensors.clear();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST(DLTensorGraveyard, EnqueueForDeletion) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({10, 10}, DALI_FLOAT);
  auto data_ptr = tensor.get_data_ptr();
  int initial_count = data_ptr.use_count();
  EnqueueForDeletion(data_ptr, tensor.device_id());
  EXPECT_GT(data_ptr.use_count(), initial_count) << "Reference count should increase";
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test all supported data types for ToDLType
TEST(DLPackTest, ToDLTypeAllTypes) {
  // Test unsigned integer types
  auto dl_uint8 = ToDLType(DALI_UINT8);
  EXPECT_EQ(dl_uint8.code, kDLUInt);
  EXPECT_EQ(dl_uint8.bits, 8);
  EXPECT_EQ(dl_uint8.lanes, 1);

  auto dl_uint16 = ToDLType(DALI_UINT16);
  EXPECT_EQ(dl_uint16.code, kDLUInt);
  EXPECT_EQ(dl_uint16.bits, 16);

  auto dl_uint32 = ToDLType(DALI_UINT32);
  EXPECT_EQ(dl_uint32.code, kDLUInt);
  EXPECT_EQ(dl_uint32.bits, 32);

  auto dl_uint64 = ToDLType(DALI_UINT64);
  EXPECT_EQ(dl_uint64.code, kDLUInt);
  EXPECT_EQ(dl_uint64.bits, 64);

  // Test signed integer types
  auto dl_int8 = ToDLType(DALI_INT8);
  EXPECT_EQ(dl_int8.code, kDLInt);
  EXPECT_EQ(dl_int8.bits, 8);

  auto dl_int16 = ToDLType(DALI_INT16);
  EXPECT_EQ(dl_int16.code, kDLInt);
  EXPECT_EQ(dl_int16.bits, 16);

  auto dl_int32 = ToDLType(DALI_INT32);
  EXPECT_EQ(dl_int32.code, kDLInt);
  EXPECT_EQ(dl_int32.bits, 32);

  auto dl_int64 = ToDLType(DALI_INT64);
  EXPECT_EQ(dl_int64.code, kDLInt);
  EXPECT_EQ(dl_int64.bits, 64);

  // Test floating point types
  auto dl_float16 = ToDLType(DALI_FLOAT16);
  EXPECT_EQ(dl_float16.code, kDLFloat);
  EXPECT_EQ(dl_float16.bits, 16);

  auto dl_float = ToDLType(DALI_FLOAT);
  EXPECT_EQ(dl_float.code, kDLFloat);
  EXPECT_EQ(dl_float.bits, 32);

  auto dl_float64 = ToDLType(DALI_FLOAT64);
  EXPECT_EQ(dl_float64.code, kDLFloat);
  EXPECT_EQ(dl_float64.bits, 64);

  // Test bool type
  auto dl_bool = ToDLType(DALI_BOOL);
  EXPECT_EQ(dl_bool.code, kDLBool);
  EXPECT_EQ(dl_bool.bits, 8);
}

// Test pinned CPU memory tensors
TEST(DLMTensorPtr, PinnedCPUShared) {
  Tensor<CPUBackend> tensor;
  tensor.set_pinned(true);
  tensor.set_device_id(0);
  tensor.Resize({50, 25, 4}, DALI_INT16);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 50);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 25);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 4);
    EXPECT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int16_t) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDAHost);
    EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
  }
  // Pinned memory goes through EnqueueForDeletion, so we need to sync and wait
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor with different data types
TEST(DLMTensorPtr, GPUSharedInt64) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({20, 30}, DALI_INT64);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 2);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 20);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 30);
    EXPECT_EQ(dlm_tensor->dl_tensor.data, tensor.raw_data());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int64_t) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_id, tensor.device_id());
    EXPECT_EQ(dlm_tensor->dl_tensor.byte_offset, 0);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor with uint64
TEST(DLMTensorPtr, GPUSharedUInt64) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({15, 25}, DALI_UINT64);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLUInt);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(uint64_t) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor with float16
TEST(DLMTensorPtr, GPUSharedFloat16) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({10, 20, 5}, DALI_FLOAT16);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, 16);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor with double
TEST(DLMTensorPtr, GPUSharedFloat64) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({8, 12}, DALI_FLOAT64);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(double) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor with bool
TEST(DLMTensorPtr, GPUSharedBool) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({32, 32}, DALI_BOOL);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count) << "Reference count not increased";
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLBool);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(bool) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor list with various data types
TEST(DLMTensorPtr, GPUSharedListInt16) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{64, 64}, {32, 128}}, DALI_INT16);
  const auto &ptr = unsafe_owner(tlist);
  int initial_count = ptr.use_count();
  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_GT(ptr.use_count(), initial_count) << "Reference count not increased";
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.dtype.bits, sizeof(int16_t) * 8);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.dtype.bits, sizeof(int16_t) * 8);
  dlm_tensors.clear();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor list with uint32
TEST(DLMTensorPtr, GPUSharedListUInt32) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{40, 30}, {50, 50}, {20, 60}}, DALI_UINT32);
  const auto &ptr = unsafe_owner(tlist);
  int initial_count = ptr.use_count();
  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_GT(ptr.use_count(), initial_count) << "Reference count not increased";
  EXPECT_EQ(dlm_tensors.size(), 3);
  for (const auto &tensor : dlm_tensors) {
    EXPECT_EQ(tensor->dl_tensor.dtype.code, kDLUInt);
    EXPECT_EQ(tensor->dl_tensor.dtype.bits, sizeof(uint32_t) * 8);
    EXPECT_EQ(tensor->dl_tensor.device.device_type, kDLCUDA);
  }
  dlm_tensors.clear();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test 1D tensor (vector)
TEST(DLMTensorPtr, GPU1DTensor) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({1000}, DALI_FLOAT);
  DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 1);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 1000);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  dlm_tensor.reset();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test 4D tensor
TEST(DLMTensorPtr, GPU4DTensor) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({2, 3, 4, 5}, DALI_INT32);
  DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 4);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 2);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 4);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[3], 5);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  dlm_tensor.reset();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test scalar tensor
TEST(DLMTensorPtr, GPUScalarTensor) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({1}, DALI_INT32);
  DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 1);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 1);
  EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  dlm_tensor.reset();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test CPU tensor list with multiple samples
TEST(DLMTensorPtr, CPUSharedListMultipleSamples) {
  TensorList<CPUBackend> tlist;
  tlist.set_pinned(false);
  tlist.Resize({{10, 10}, {20, 20}, {15, 15}, {25, 25}, {30, 30}}, DALI_INT32);
  const auto &ptr = unsafe_owner(tlist);
  int initial_count = ptr.use_count();
  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_GT(ptr.use_count(), initial_count) << "Reference count not increased";
  EXPECT_EQ(dlm_tensors.size(), 5);

  // Verify each sample
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 10);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 20);
  EXPECT_EQ(dlm_tensors[2]->dl_tensor.shape[0], 15);
  EXPECT_EQ(dlm_tensors[3]->dl_tensor.shape[0], 25);
  EXPECT_EQ(dlm_tensors[4]->dl_tensor.shape[0], 30);

  for (const auto &tensor : dlm_tensors) {
    EXPECT_EQ(tensor->dl_tensor.dtype.code, kDLInt);
    EXPECT_EQ(tensor->dl_tensor.dtype.bits, sizeof(int32_t) * 8);
    EXPECT_EQ(tensor->dl_tensor.device.device_type, kDLCPU);
  }

  dlm_tensors.clear();
  EXPECT_EQ(ptr.use_count(), initial_count);
}

// Test ToDALIType with all valid types
TEST(DLPackTest, ToDALITypeAllValidTypes) {
  // Test all unsigned integer types
  EXPECT_EQ(ToDALIType(DLDataType{kDLUInt, 8, 1}), DALI_UINT8);
  EXPECT_EQ(ToDALIType(DLDataType{kDLUInt, 16, 1}), DALI_UINT16);
  EXPECT_EQ(ToDALIType(DLDataType{kDLUInt, 32, 1}), DALI_UINT32);
  EXPECT_EQ(ToDALIType(DLDataType{kDLUInt, 64, 1}), DALI_UINT64);

  // Test all signed integer types
  EXPECT_EQ(ToDALIType(DLDataType{kDLInt, 8, 1}), DALI_INT8);
  EXPECT_EQ(ToDALIType(DLDataType{kDLInt, 16, 1}), DALI_INT16);
  EXPECT_EQ(ToDALIType(DLDataType{kDLInt, 32, 1}), DALI_INT32);
  EXPECT_EQ(ToDALIType(DLDataType{kDLInt, 64, 1}), DALI_INT64);

  // Test all floating point types
  EXPECT_EQ(ToDALIType(DLDataType{kDLFloat, 16, 1}), DALI_FLOAT16);
  EXPECT_EQ(ToDALIType(DLDataType{kDLFloat, 32, 1}), DALI_FLOAT);
  EXPECT_EQ(ToDALIType(DLDataType{kDLFloat, 64, 1}), DALI_FLOAT64);

  // Test bool type
  EXPECT_EQ(ToDALIType(DLDataType{kDLBool, 8, 1}), DALI_BOOL);
}

// Test view tensors with different data types
TEST(DLMTensorPtr, ViewCPUDifferentTypes) {
  // Test with uint16
  {
    Tensor<CPUBackend> tensor;
    tensor.set_pinned(false);
    tensor.Resize({50, 60}, DALI_UINT16);
    SampleView<CPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
    DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, 0);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLUInt);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(uint16_t) * 8);
  }

  // Test with int64
  {
    Tensor<CPUBackend> tensor;
    tensor.set_pinned(false);
    tensor.Resize({40, 40}, DALI_INT64);
    SampleView<CPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
    DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, 0);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(int64_t) * 8);
  }

  // Test with float16
  {
    Tensor<CPUBackend> tensor;
    tensor.set_pinned(false);
    tensor.Resize({30, 30}, DALI_FLOAT16);
    SampleView<CPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
    DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, 0);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, 16);
  }
}

// Test GPU view tensors with different data types
TEST(DLMTensorPtr, ViewGPUDifferentTypes) {
  // Test with uint64
  {
    Tensor<GPUBackend> tensor;
    tensor.Resize({25, 35}, DALI_UINT64);
    SampleView<GPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
    DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, tensor.device_id());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLUInt);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(uint64_t) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }

  // Test with float64
  {
    Tensor<GPUBackend> tensor;
    tensor.Resize({45, 55}, DALI_FLOAT64);
    SampleView<GPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
    DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, tensor.device_id());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(double) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }

  // Test with bool
  {
    Tensor<GPUBackend> tensor;
    tensor.Resize({16, 16}, DALI_BOOL);
    SampleView<GPUBackend> sv{tensor.raw_mutable_data(), tensor.shape(), tensor.type()};
    DLMTensorPtr dlm_tensor = GetDLTensorView(sv, false, tensor.device_id());
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLBool);
    EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, sizeof(bool) * 8);
    EXPECT_EQ(dlm_tensor->dl_tensor.device.device_type, kDLCUDA);
  }
}

// Test multiple GPU tensors with rapid allocation and deallocation
TEST(DLTensorGraveyard, MultipleEnqueues) {
  std::vector<Tensor<GPUBackend>> tensors;
  tensors.resize(10);

  // Create multiple tensors and enqueue them for deletion
  std::vector<std::shared_ptr<void>> ptrs;
  for (int i = 0; i < 10; i++) {
    tensors[i].Resize({100, 100}, DALI_FLOAT);
    auto ptr = tensors[i].get_data_ptr();
    ptrs.push_back(ptr);
    EnqueueForDeletion(ptr, tensors[i].device_id());
  }

  // Verify all pointers have increased reference counts
  for (int i = 0; i < 10; i++) {
    EXPECT_GT(ptrs[i].use_count(), 1) << "Reference count not increased for tensor " << i;
  }

  // Sync and wait for cleanup
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

// Test rapid creation and destruction of GPU shared tensors
TEST(DLMTensorPtr, RapidGPUSharedCreationDestruction) {
  for (int i = 0; i < 20; i++) {
    Tensor<GPUBackend> tensor;
    tensor.Resize({50 + i, 50 + i}, DALI_INT32);
    int initial_count = tensor.get_data_ptr().use_count();
    {
      DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
      EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count);
    }
    // Don't sync after every iteration to test accumulation
    if (i % 5 == 4) {
      CUDA_CALL(cudaDeviceSynchronize());
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  // Final sync
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test multiple GPU tensor lists being cleaned up
TEST(DLMTensorPtr, MultipleGPUSharedLists) {
  std::vector<TensorList<GPUBackend>> tlists(5);
  std::vector<std::vector<DLMTensorPtr>> all_tensors;

  for (int i = 0; i < 5; i++) {
    tlists[i].Resize({{32, 32}, {64, 64}}, DALI_FLOAT);
    auto dlm_tensors = GetSharedDLTensorList(tlists[i]);
    all_tensors.push_back(std::move(dlm_tensors));
  }

  // Verify all tensor lists have proper shared ownership
  for (int i = 0; i < 5; i++) {
    const auto &ptr = unsafe_owner(tlists[i]);
    EXPECT_GT(ptr.use_count(), 1) << "Tensor list " << i << " doesn't have increased ref count";
  }

  // Clear all DL tensors
  all_tensors.clear();

  // Sync and wait for cleanup
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
}

// Test mixed CPU and GPU tensor operations
TEST(DLMTensorPtr, MixedCPUGPUOperations) {
  // Create CPU tensor
  Tensor<CPUBackend> cpu_tensor;
  cpu_tensor.set_pinned(false);
  cpu_tensor.Resize({100, 100}, DALI_FLOAT);

  // Create GPU tensors
  Tensor<GPUBackend> gpu_tensor1;
  gpu_tensor1.Resize({50, 50}, DALI_INT32);

  Tensor<GPUBackend> gpu_tensor2;
  gpu_tensor2.Resize({75, 75}, DALI_UINT8);

  // Get shared DL tensors
  {
    DLMTensorPtr cpu_dlm = GetSharedDLTensor(cpu_tensor);
    DLMTensorPtr gpu_dlm1 = GetSharedDLTensor(gpu_tensor1);
    DLMTensorPtr gpu_dlm2 = GetSharedDLTensor(gpu_tensor2);

    EXPECT_EQ(cpu_dlm->dl_tensor.device.device_type, kDLCPU);
    EXPECT_EQ(gpu_dlm1->dl_tensor.device.device_type, kDLCUDA);
    EXPECT_EQ(gpu_dlm2->dl_tensor.device.device_type, kDLCUDA);
  }

  // Sync for GPU cleanup
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test large GPU tensor allocation and cleanup
TEST(DLMTensorPtr, LargeGPUTensor) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({1024, 1024, 4}, DALI_FLOAT);
  int initial_count = tensor.get_data_ptr().use_count();
  {
    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    EXPECT_GT(tensor.get_data_ptr().use_count(), initial_count);
    EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 3);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 1024);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 1024);
    EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 4);
  }
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test GPU tensor list with varying shapes
TEST(DLMTensorPtr, GPUListVaryingShapes) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({{10, 20, 3}, {50, 50, 3}, {100, 25, 3}, {25, 100, 3}, {75, 75, 3}}, DALI_FLOAT16);

  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_EQ(dlm_tensors.size(), 5);

  // Verify each tensor has the correct shape
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[0], 10);
  EXPECT_EQ(dlm_tensors[0]->dl_tensor.shape[1], 20);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[0], 50);
  EXPECT_EQ(dlm_tensors[1]->dl_tensor.shape[1], 50);
  EXPECT_EQ(dlm_tensors[2]->dl_tensor.shape[0], 100);
  EXPECT_EQ(dlm_tensors[2]->dl_tensor.shape[1], 25);
  EXPECT_EQ(dlm_tensors[3]->dl_tensor.shape[0], 25);
  EXPECT_EQ(dlm_tensors[3]->dl_tensor.shape[1], 100);
  EXPECT_EQ(dlm_tensors[4]->dl_tensor.shape[0], 75);
  EXPECT_EQ(dlm_tensors[4]->dl_tensor.shape[1], 75);

  // All should be float16
  for (const auto &tensor : dlm_tensors) {
    EXPECT_EQ(tensor->dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(tensor->dl_tensor.dtype.bits, 16);
  }

  dlm_tensors.clear();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test pinned CPU tensor list
TEST(DLMTensorPtr, PinnedCPUList) {
  TensorList<CPUBackend> tlist;
  tlist.set_pinned(true);
  tlist.set_device_id(0);
  tlist.Resize({{128, 128}, {256, 128}, {128, 256}}, DALI_UINT16);

  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_EQ(dlm_tensors.size(), 3);

  // All should be kDLCUDAHost for pinned memory
  for (const auto &tensor : dlm_tensors) {
    EXPECT_EQ(tensor->dl_tensor.device.device_type, kDLCUDAHost);
    EXPECT_EQ(tensor->dl_tensor.dtype.code, kDLUInt);
    EXPECT_EQ(tensor->dl_tensor.dtype.bits, 16);
  }

  dlm_tensors.clear();
  // Pinned memory also goes through graveyard
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test sequential GPU tensor operations to exercise graveyard queue
TEST(DLTensorGraveyard, SequentialOperations) {
  for (int iteration = 0; iteration < 30; iteration++) {
    Tensor<GPUBackend> tensor;
    tensor.Resize({64, 64}, DALI_FLOAT);
    auto data_ptr = tensor.get_data_ptr();

    // Enqueue for deletion
    EnqueueForDeletion(data_ptr, tensor.device_id());

    // Every 10 iterations, sync to allow graveyard to process
    if (iteration % 10 == 9) {
      CUDA_CALL(cudaDeviceSynchronize());
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  // Final sync
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test empty GPU tensor list
TEST(DLMTensorPtr, EmptyGPUTensorList) {
  TensorList<GPUBackend> tlist;
  tlist.Resize({}, DALI_FLOAT);  // Empty list

  std::vector<DLMTensorPtr> dlm_tensors = GetSharedDLTensorList(tlist);
  EXPECT_EQ(dlm_tensors.size(), 0);
}

// Test 5D GPU tensor (higher dimensional)
TEST(DLMTensorPtr, GPU5DTensor) {
  Tensor<GPUBackend> tensor;
  tensor.Resize({2, 3, 4, 5, 6}, DALI_INT8);

  DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
  EXPECT_EQ(dlm_tensor->dl_tensor.ndim, 5);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[0], 2);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[1], 3);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[2], 4);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[3], 5);
  EXPECT_EQ(dlm_tensor->dl_tensor.shape[4], 6);
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlm_tensor->dl_tensor.dtype.bits, 8);

  dlm_tensor.reset();
  CUDA_CALL(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

}  // namespace dali
