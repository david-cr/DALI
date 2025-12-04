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
#include <cstdlib>
#include <string>
#include "dali/pipeline/init.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

// Forward declaration of internal function for testing
// This function is defined in init.cc but not exposed in init.h
extern void InitializeBufferPolicies();

class InitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Save original environment variables
    SaveEnvVar("DALI_HOST_BUFFER_SHRINK_THRESHOLD", saved_shrink_threshold_);
    SaveEnvVar("DALI_BUFFER_GROWTH_FACTOR", saved_growth_factor_);
    SaveEnvVar("DALI_HOST_BUFFER_GROWTH_FACTOR", saved_host_growth_factor_);
    SaveEnvVar("DALI_DEVICE_BUFFER_GROWTH_FACTOR", saved_device_growth_factor_);

    // Save original buffer settings
    original_cpu_growth_ = Buffer<CPUBackend>::GetGrowthFactor();
    original_gpu_growth_ = Buffer<GPUBackend>::GetGrowthFactor();
    original_cpu_shrink_ = Buffer<CPUBackend>::GetShrinkThreshold();
  }

  void TearDown() override {
    // Restore original environment variables
    RestoreEnvVar("DALI_HOST_BUFFER_SHRINK_THRESHOLD", saved_shrink_threshold_);
    RestoreEnvVar("DALI_BUFFER_GROWTH_FACTOR", saved_growth_factor_);
    RestoreEnvVar("DALI_HOST_BUFFER_GROWTH_FACTOR", saved_host_growth_factor_);
    RestoreEnvVar("DALI_DEVICE_BUFFER_GROWTH_FACTOR", saved_device_growth_factor_);

    // Restore original buffer settings
    Buffer<CPUBackend>::SetGrowthFactor(original_cpu_growth_);
    Buffer<GPUBackend>::SetGrowthFactor(original_gpu_growth_);
    Buffer<CPUBackend>::SetShrinkThreshold(original_cpu_shrink_);
  }

 private:
  void SaveEnvVar(const char *name, std::pair<bool, std::string> &storage) {
    const char *value = std::getenv(name);
    if (value) {
      storage = {true, value};
    } else {
      storage = {false, ""};
    }
  }

  void RestoreEnvVar(const char *name, const std::pair<bool, std::string> &storage) {
    if (storage.first) {
      setenv(name, storage.second.c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  std::pair<bool, std::string> saved_shrink_threshold_;
  std::pair<bool, std::string> saved_growth_factor_;
  std::pair<bool, std::string> saved_host_growth_factor_;
  std::pair<bool, std::string> saved_device_growth_factor_;

  double original_cpu_growth_;
  double original_gpu_growth_;
  double original_cpu_shrink_;
};

// ====================================================================================
// Tests for InitializeBufferPolicies() - Lines 46-64
// ====================================================================================

// Test DALI_HOST_BUFFER_SHRINK_THRESHOLD environment variable
// Covers lines 47-49
TEST_F(InitTest, HostBufferShrinkThreshold) {
  // Test with valid threshold value
  setenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD", "0.5", 1);
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetShrinkThreshold(), 0.5);
}

// Test DALI_HOST_BUFFER_SHRINK_THRESHOLD with clamping to [0.0, 1.0]
// Covers lines 47-49 with clamp behavior
TEST_F(InitTest, HostBufferShrinkThresholdClamping) {
  // Test clamping to minimum (0.0)
  setenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD", "-0.5", 1);
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetShrinkThreshold(), 0.0);

  // Test clamping to maximum (1.0)
  setenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD", "2.0", 1);
  InitializeBufferPolicies();
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetShrinkThreshold(), 1.0);
}

// Test DALI_BUFFER_GROWTH_FACTOR environment variable
// Covers lines 50-55 (sets both CPU and GPU growth factors)
TEST_F(InitTest, BufferGrowthFactor) {
  setenv("DALI_BUFFER_GROWTH_FACTOR", "1.5", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  // Should set both CPU and GPU backends
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.5);
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 1.5);
}

// Test DALI_BUFFER_GROWTH_FACTOR with clamping
// Covers lines 50-55 with clamp behavior
TEST_F(InitTest, BufferGrowthFactorClamping) {
  // Test clamping to minimum (1.0)
  setenv("DALI_BUFFER_GROWTH_FACTOR", "0.5", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.0);
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 1.0);

  // Test clamping to maximum (kMaxGrowthFactor)
  setenv("DALI_BUFFER_GROWTH_FACTOR", "1000.0", 1);
  InitializeBufferPolicies();
  EXPECT_LE(Buffer<CPUBackend>::GetGrowthFactor(), Buffer<CPUBackend>::kMaxGrowthFactor);
  EXPECT_LE(Buffer<GPUBackend>::GetGrowthFactor(), Buffer<GPUBackend>::kMaxGrowthFactor);
}

// Test DALI_HOST_BUFFER_GROWTH_FACTOR environment variable
// Covers lines 56-59 (sets only CPU growth factor)
TEST_F(InitTest, HostBufferGrowthFactor) {
  // Save GPU growth factor to verify it doesn't change
  double gpu_before = Buffer<GPUBackend>::GetGrowthFactor();

  setenv("DALI_HOST_BUFFER_GROWTH_FACTOR", "1.8", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  // Should only set CPU backend
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.8);
  // GPU backend should remain unchanged
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), gpu_before);
}

// Test DALI_HOST_BUFFER_GROWTH_FACTOR with clamping
// Covers lines 56-59 with clamp behavior
TEST_F(InitTest, HostBufferGrowthFactorClamping) {
  // Test clamping to minimum (1.0)
  setenv("DALI_HOST_BUFFER_GROWTH_FACTOR", "0.8", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.0);

  // Test clamping to maximum (kMaxGrowthFactor)
  setenv("DALI_HOST_BUFFER_GROWTH_FACTOR", "1000.0", 1);
  InitializeBufferPolicies();
  EXPECT_LE(Buffer<CPUBackend>::GetGrowthFactor(), Buffer<CPUBackend>::kMaxGrowthFactor);
}

// Test DALI_DEVICE_BUFFER_GROWTH_FACTOR environment variable
// Covers lines 60-63 (sets only GPU growth factor)
TEST_F(InitTest, DeviceBufferGrowthFactor) {
  // Save CPU growth factor to verify it doesn't change
  double cpu_before = Buffer<CPUBackend>::GetGrowthFactor();

  setenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR", "2.0", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  // Should only set GPU backend
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 2.0);
  // CPU backend should remain unchanged
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), cpu_before);
}

// Test DALI_DEVICE_BUFFER_GROWTH_FACTOR with clamping
// Covers lines 60-63 with clamp behavior
TEST_F(InitTest, DeviceBufferGrowthFactorClamping) {
  // Test clamping to minimum (1.0)
  setenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR", "0.3", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 1.0);

  // Test clamping to maximum (kMaxGrowthFactor)
  setenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR", "1000.0", 1);
  InitializeBufferPolicies();
  EXPECT_LE(Buffer<GPUBackend>::GetGrowthFactor(), Buffer<GPUBackend>::kMaxGrowthFactor);
}

// Test with all environment variables set
// Covers all lines 47-63
TEST_F(InitTest, AllEnvironmentVariablesSet) {
  setenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD", "0.3", 1);
  setenv("DALI_BUFFER_GROWTH_FACTOR", "1.6", 1);
  setenv("DALI_HOST_BUFFER_GROWTH_FACTOR", "1.7", 1);
  setenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR", "1.9", 1);

  InitializeBufferPolicies();

  // DALI_HOST_BUFFER_GROWTH_FACTOR should override DALI_BUFFER_GROWTH_FACTOR for CPU
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.7);
  // DALI_DEVICE_BUFFER_GROWTH_FACTOR should override DALI_BUFFER_GROWTH_FACTOR for GPU
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 1.9);
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetShrinkThreshold(), 0.3);
}

// Test with no environment variables set
// Covers lines 46-64 with all if conditions being false
TEST_F(InitTest, NoEnvironmentVariablesSet) {
  // Save original values
  double cpu_growth_before = Buffer<CPUBackend>::GetGrowthFactor();
  double gpu_growth_before = Buffer<GPUBackend>::GetGrowthFactor();
  double cpu_shrink_before = Buffer<CPUBackend>::GetShrinkThreshold();

  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  // All values should remain unchanged
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), cpu_growth_before);
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), gpu_growth_before);
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetShrinkThreshold(), cpu_shrink_before);
}

// Test priority: specific environment variables override general ones
// Covers lines 50-63 interaction
TEST_F(InitTest, EnvironmentVariablePriority) {
  // Set DALI_BUFFER_GROWTH_FACTOR to 1.4
  setenv("DALI_BUFFER_GROWTH_FACTOR", "1.4", 1);
  unsetenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD");
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  // Both should be 1.4
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.4);
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 1.4);

  // Now set specific overrides
  setenv("DALI_BUFFER_GROWTH_FACTOR", "1.4", 1);
  setenv("DALI_HOST_BUFFER_GROWTH_FACTOR", "1.3", 1);
  setenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR", "2.1", 1);

  InitializeBufferPolicies();

  // Specific values should override the general one
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.3);
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 2.1);
}

// Test with invalid/unparseable values (atof behavior)
// Covers lines 48, 52, 54, 58, 62 with atof() edge cases
TEST_F(InitTest, InvalidEnvironmentVariableValues) {
  // atof() returns 0.0 for unparseable strings, which gets clamped to valid ranges
  setenv("DALI_HOST_BUFFER_SHRINK_THRESHOLD", "invalid", 1);
  setenv("DALI_BUFFER_GROWTH_FACTOR", "not_a_number", 1);
  unsetenv("DALI_HOST_BUFFER_GROWTH_FACTOR");
  unsetenv("DALI_DEVICE_BUFFER_GROWTH_FACTOR");

  InitializeBufferPolicies();

  // atof("invalid") returns 0.0, clamped to [0.0, 1.0] for shrink threshold
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetShrinkThreshold(), 0.0);
  // atof("not_a_number") returns 0.0, clamped to [1.0, max] for growth factor
  EXPECT_DOUBLE_EQ(Buffer<CPUBackend>::GetGrowthFactor(), 1.0);
  EXPECT_DOUBLE_EQ(Buffer<GPUBackend>::GetGrowthFactor(), 1.0);
}

}  // namespace dali

