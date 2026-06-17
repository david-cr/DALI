// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/checkpointing/snapshot_serializer.h"

#include <gtest/gtest.h>

#include "dali/test/dali_test.h"

namespace dali {

class SnapshotSerializerTest : public DALITest {};

TEST_F(SnapshotSerializerTest, VectorMt19937) {
  std::vector<std::mt19937> snapshot;
  for (int i = 123; i <= 321; i++)
    snapshot.emplace_back(i);

  std::string serialized = SnapshotSerializer().Serialize(snapshot);
  auto deserialized = SnapshotSerializer().Deserialize<std::vector<std::mt19937>>(serialized);

  ASSERT_EQ(snapshot.size(), deserialized.size());
  for (size_t i = 0; i < snapshot.size(); i++)
    EXPECT_EQ(snapshot[i], deserialized[i]);
}

TEST_F(SnapshotSerializerTest, VectorMt19937_64) {
  std::vector<std::mt19937_64> snapshot;
  for (int i = 123; i <= 321; i++)
    snapshot.emplace_back(i);

  std::string serialized = SnapshotSerializer().Serialize(snapshot);
  auto deserialized = SnapshotSerializer().Deserialize<std::vector<std::mt19937_64>>(serialized);

  ASSERT_EQ(snapshot.size(), deserialized.size());
  for (size_t i = 0; i < snapshot.size(); i++)
    EXPECT_EQ(snapshot[i], deserialized[i]);
}

TEST_F(SnapshotSerializerTest, LoaderStateSnapshot) {
  LoaderStateSnapshot snapshot = {
    std::default_random_engine(123),
    321,
    567
  };

  std::string serialized = SnapshotSerializer().Serialize(snapshot);
  auto deserialized = SnapshotSerializer().Deserialize<LoaderStateSnapshot>(serialized);

  EXPECT_EQ(snapshot.rng, deserialized.rng);
  EXPECT_EQ(snapshot.current_epoch, deserialized.current_epoch);
  EXPECT_EQ(snapshot.age, deserialized.age);
}

// Test GPU curandState serialization/deserialization
// Covers lines 89-93
TEST_F(SnapshotSerializerTest, VectorCurandState) {
  int device_count = 0;
  CUDA_CALL(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    GTEST_SKIP() << "At least 1 GPU required";
  }

  // Create a vector of curandState
  std::vector<curandState> snapshot(5);

  // Initialize curandStates with different seeds
  for (size_t i = 0; i < snapshot.size(); i++) {
    curand_init(123 + i, 0, 0, &snapshot[i]);
  }

  // Serialize
  std::string serialized = SnapshotSerializer().Serialize(snapshot);

  // Deserialize
  auto deserialized = SnapshotSerializer().Deserialize<std::vector<curandState>>(serialized);

  // Verify size
  ASSERT_EQ(snapshot.size(), deserialized.size());

  // Verify state is preserved by comparing binary data
  for (size_t i = 0; i < snapshot.size(); i++) {
    EXPECT_EQ(memcmp(&snapshot[i], &deserialized[i], sizeof(curandState)), 0);
  }
}

// Test GPU curandState serialization with empty vector
// Covers lines 89-93 with edge case
TEST_F(SnapshotSerializerTest, VectorCurandStateEmpty) {
  int device_count = 0;
  CUDA_CALL(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    GTEST_SKIP() << "At least 1 GPU required";
  }

  // Create empty vector
  std::vector<curandState> snapshot;

  // Serialize empty vector
  std::string serialized = SnapshotSerializer().Serialize(snapshot);

  // Deserialize
  auto deserialized = SnapshotSerializer().Deserialize<std::vector<curandState>>(serialized);

  // Verify empty
  EXPECT_EQ(deserialized.size(), 0);
}

// Test GPU curandState round-trip with generated values
// Covers lines 97-103 (deserialize path)
TEST_F(SnapshotSerializerTest, VectorCurandStateRoundTrip) {
  int device_count = 0;
  CUDA_CALL(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    GTEST_SKIP() << "At least 1 GPU required";
  }

  // Create and initialize curandStates
  std::vector<curandState> original(10);
  for (size_t i = 0; i < original.size(); i++) {
    curand_init(456 + i * 7, i, 0, &original[i]);
  }

  // Serialize
  std::string serialized = SnapshotSerializer().Serialize(original);

  // Verify serialized data is not empty and reasonable size
  EXPECT_GT(serialized.size(), 0);
  // Protobuf adds overhead, so size will be >= raw data size
  EXPECT_GE(serialized.size(), original.size() * sizeof(curandState));

  // Deserialize
  auto restored = SnapshotSerializer().Deserialize<std::vector<curandState>>(serialized);

  // Verify exact match
  ASSERT_EQ(original.size(), restored.size());
  for (size_t i = 0; i < original.size(); i++) {
    EXPECT_EQ(memcmp(&original[i], &restored[i], sizeof(curandState)), 0)
        << "Mismatch at index " << i;
  }
}

// Malformed protobuf wire data: a length-delimited field (tag 0x0a) that claims
// 16 payload bytes but provides none. ParseFromString must fail, exercising the
// DALI_ENFORCE failure paths in the Deserialize overloads. These do not require
// a GPU since the parse failure happens before any device work.
namespace {
const std::string kInvalidProtobuf("\x0a\x10", 2);
}  // namespace

TEST_F(SnapshotSerializerTest, DeserializeMt19937InvalidData) {
  EXPECT_THROW(SnapshotSerializer().Deserialize<std::vector<std::mt19937>>(kInvalidProtobuf),
               DALIException);
}

TEST_F(SnapshotSerializerTest, DeserializeMt19937_64InvalidData) {
  EXPECT_THROW(SnapshotSerializer().Deserialize<std::vector<std::mt19937_64>>(kInvalidProtobuf),
               DALIException);
}

TEST_F(SnapshotSerializerTest, DeserializeCurandStateInvalidData) {
  EXPECT_THROW(SnapshotSerializer().Deserialize<std::vector<curandState>>(kInvalidProtobuf),
               DALIException);
}

TEST_F(SnapshotSerializerTest, DeserializeLoaderStateInvalidData) {
  EXPECT_THROW(SnapshotSerializer().Deserialize<LoaderStateSnapshot>(kInvalidProtobuf),
               DALIException);
}

}  // namespace dali
