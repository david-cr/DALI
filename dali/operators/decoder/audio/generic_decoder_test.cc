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
#include <fstream>
#include <string>
#include <cstring>
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/span.h"

namespace dali {
namespace {

std::string audio_data_root = make_string(testing::dali_extra_path(), "/db/audio/wav/");  // NOLINT

/**
 * Reads file and saves it to vector. Saves the file as plain numbers.
 */
template<typename T>
std::vector<T> ReadTxt(const std::string &filepath) {
  std::ifstream file(filepath.c_str());
  std::istream_iterator<T> begin(file);
  std::istream_iterator<T> end;
  return {begin, end};
}

/**
 * Reads file as a byte stream and saves it to vector.
 */
std::vector<char> ReadBytes(const std::string &filepath) {
  std::vector<char> ret;
  std::ifstream infile(filepath);
  infile.seekg(0, std::ios::end);
  size_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  ret.resize(length);
  infile.read(ret.data(), length);
  return ret;
}


template<typename T>
bool CheckBuffers(const T *buf1, const T *buf2, int size) {
  return !std::memcmp(buf1, buf2, sizeof(T) * size);
}

}  // namespace

TEST(AudioDecoderTest, WavDecoderTest) {
  using DataType = short;  // NOLINT
  auto decoder = make_generic_audio_decoder();

  // Contains wav file to be decoded
  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  // Contains raw PCM data decoded offline
  std::string decoded_path = make_string(audio_data_root, "dziendobry.txt");

  constexpr int expected_frequency = 44100;
  constexpr int expected_nchannels = 2;
  std::vector<DataType> vec;
  std::vector<char> bytes;
  try {
    vec = ReadTxt<DataType>(decoded_path);
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data hasn't been provided: Expected `" << wav_path << "` and `" << decoded_path
           << "` to exist";
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    EXPECT_EQ(meta.channels, expected_nchannels);
    EXPECT_EQ(meta.channels_interleaved, true);
    EXPECT_EQ(meta.sample_rate, expected_frequency);
    decoder->Close();
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    std::vector<DataType> output(meta.length * meta.channels);
    decoder->Decode(make_span(output));
    EXPECT_PRED3(CheckBuffers<DataType>, output.data(), vec.data(), vec.size());
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    std::vector<DataType> output(meta.length * meta.channels);
    decoder->DecodeFrames(output.data(), meta.length);
    EXPECT_PRED3(CheckBuffers<DataType>, output.data(), vec.data(), vec.size());
  }

  {
    auto meta = decoder->Open(make_cspan(bytes));
    int64_t offset = meta.length / 2;
    int64_t length = meta.length - offset;
    // allocating a bigger buffer in purpose
    std::vector<DataType> output(meta.length * meta.channels, 0xBE);
    decoder->SeekFrames(offset, SEEK_CUR);
    decoder->DecodeFrames(output.data(), length);
    EXPECT_PRED3(CheckBuffers<DataType>, output.data(), vec.data() + offset * meta.channels,
                 length * meta.channels);
    // Verifying that we didn't read more than we should
    for (size_t i = length * meta.channels; i < output.size(); i++) {
      ASSERT_EQ(0xBE, output[i]);
    }
  }
}

// ============================================================================
// Test Decode with int32_t output type (covers DecodeImpl(span<int32_t>))
// ============================================================================

TEST(AudioDecoderTest, DecodeInt32) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  auto meta = decoder->Open(make_cspan(bytes));
  EXPECT_GT(meta.length, 0);
  EXPECT_GT(meta.channels, 0);

  std::vector<int32_t> output(meta.length * meta.channels);
  auto samples_read = decoder->Decode(make_span(output));
  EXPECT_GT(samples_read, 0);
}

// ============================================================================
// Test Decode with float output type (covers DecodeImpl(span<float>))
// ============================================================================

TEST(AudioDecoderTest, DecodeFloat) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  auto meta = decoder->Open(make_cspan(bytes));
  EXPECT_GT(meta.length, 0);
  EXPECT_GT(meta.channels, 0);

  std::vector<float> output(meta.length * meta.channels);
  auto samples_read = decoder->Decode(make_span(output));
  EXPECT_GT(samples_read, 0);

  // Verify float values are in reasonable range [-1, 1] for normalized audio
  for (int i = 0; i < std::min<int>(100, output.size()); i++) {
    EXPECT_GE(output[i], -1.1f);
    EXPECT_LE(output[i], 1.1f);
  }
}

// ============================================================================
// Test OpenFromFile with valid path (covers OpenFromFileImpl happy path)
// ============================================================================

TEST(AudioDecoderTest, OpenFromFile) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  auto meta = decoder->OpenFromFile(wav_path);
  EXPECT_GT(meta.length, 0);
  EXPECT_GT(meta.channels, 0);
  EXPECT_GT(meta.sample_rate, 0);
  EXPECT_TRUE(meta.channels_interleaved);

  // Also decode after OpenFromFile to verify it works end-to-end
  std::vector<int16_t> output(meta.length * meta.channels);
  auto frames_read = decoder->DecodeFrames(output.data(), meta.length);
  EXPECT_EQ(frames_read, meta.length);
}

// ============================================================================
// Test OpenFromFile with empty path (covers DALI_ENFORCE error path)
// ============================================================================

TEST(AudioDecoderTest, OpenFromFileEmptyPathThrows) {
  auto decoder = make_generic_audio_decoder();
  EXPECT_THROW(decoder->OpenFromFile(""), DALIException);
}

// ============================================================================
// Test OpenFromFile with non-existent path (covers DALI_FAIL error path)
// ============================================================================

TEST(AudioDecoderTest, OpenFromFileNonExistentThrows) {
  auto decoder = make_generic_audio_decoder();
  EXPECT_THROW(decoder->OpenFromFile("/nonexistent/path/to/file.wav"), DALIException);
}

// ============================================================================
// Test Open with invalid/corrupt encoded data (covers DALI_FAIL error path)
// ============================================================================

TEST(AudioDecoderTest, OpenInvalidDataThrows) {
  auto decoder = make_generic_audio_decoder();

  // Random garbage data that isn't a valid audio format
  std::vector<char> garbage(256, 'X');
  auto garbage_span = make_cspan(garbage);
  EXPECT_THROW(decoder->Open(garbage_span), DALIException);
}

// ============================================================================
// Test SeekFrames with SEEK_SET (already covered, but explicit)
// ============================================================================

TEST(AudioDecoderTest, SeekFramesSEEK_SET) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  auto meta = decoder->Open(make_cspan(bytes));

  // Seek to a specific frame from the start
  auto pos = decoder->SeekFrames(10, SEEK_SET);
  EXPECT_EQ(pos, 10);

  // Seek to beginning
  pos = decoder->SeekFrames(0, SEEK_SET);
  EXPECT_EQ(pos, 0);
}

// ============================================================================
// Test SeekFrames with SEEK_END
// ============================================================================

TEST(AudioDecoderTest, SeekFramesSEEK_END) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  auto meta = decoder->Open(make_cspan(bytes));

  // Seek to 10 frames before the end
  auto pos = decoder->SeekFrames(-10, SEEK_END);
  EXPECT_EQ(pos, meta.length - 10);
}

// ============================================================================
// Test DecodeFrames with int32_t type
// ============================================================================

TEST(AudioDecoderTest, DecodeFramesInt32) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  auto meta = decoder->Open(make_cspan(bytes));
  std::vector<int32_t> output(meta.length * meta.channels);
  auto frames_read = decoder->DecodeFrames(output.data(), meta.length);
  EXPECT_EQ(frames_read, meta.length);
}

// ============================================================================
// Test DecodeFrames with float type
// ============================================================================

TEST(AudioDecoderTest, DecodeFramesFloat) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  auto meta = decoder->Open(make_cspan(bytes));
  std::vector<float> output(meta.length * meta.channels);
  auto frames_read = decoder->DecodeFrames(output.data(), meta.length);
  EXPECT_EQ(frames_read, meta.length);
}

// ============================================================================
// Test Close and reopen (covers CloseImpl)
// ============================================================================

TEST(AudioDecoderTest, CloseAndReopen) {
  auto decoder = make_generic_audio_decoder();

  std::string wav_path = make_string(audio_data_root, "dziendobry.wav");
  std::vector<char> bytes;
  try {
    bytes = ReadBytes(wav_path);
  } catch (const std::bad_alloc &e) {
    FAIL() << "Test data not found: " << wav_path;
  }

  // Open, close, reopen
  auto meta1 = decoder->Open(make_cspan(bytes));
  decoder->Close();

  auto meta2 = decoder->Open(make_cspan(bytes));
  EXPECT_EQ(meta1.length, meta2.length);
  EXPECT_EQ(meta1.channels, meta2.channels);
  EXPECT_EQ(meta1.sample_rate, meta2.sample_rate);

  // Can still decode after reopen
  std::vector<int16_t> output(meta2.length * meta2.channels);
  auto samples = decoder->Decode(make_span(output));
  EXPECT_GT(samples, 0);
}

}  // namespace dali
