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

#include "dali/util/uri.h"
#include <gtest/gtest.h>

namespace dali {

TEST(URI, Parse_1) {
  auto uri = URI::Parse(
    "https://john.doe@www.example.com:123/forum/questions/?tag=networking&order=newest#top");
  EXPECT_EQ("https", uri.scheme());
  EXPECT_EQ("john.doe@www.example.com:123", uri.authority());
  EXPECT_EQ("/forum/questions/", uri.path());
  EXPECT_EQ("tag=networking&order=newest", uri.query());
  EXPECT_EQ("/forum/questions/?tag=networking&order=newest", uri.path_and_query());
  EXPECT_EQ("top", uri.fragment());
}

TEST(URI, Parse_2) {
  auto uri = URI::Parse(
    "ldap://[2001:db8::7]/c=GB?objectClass?one");
  EXPECT_EQ("ldap", uri.scheme());
  EXPECT_EQ("[2001:db8::7]", uri.authority());
  EXPECT_EQ("/c=GB", uri.path());
  EXPECT_EQ("objectClass?one", uri.query());
  EXPECT_EQ("/c=GB?objectClass?one", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_3) {
  auto uri = URI::Parse(
    "mailto:John.Doe@example.com");
  EXPECT_EQ("mailto", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("John.Doe@example.com", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("John.Doe@example.com", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_4) {
  auto uri = URI::Parse(
    "news:comp.infosystems.www.servers.unix");
  EXPECT_EQ("news", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("comp.infosystems.www.servers.unix", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("comp.infosystems.www.servers.unix", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_5) {
  auto uri = URI::Parse(
    "tel:+1-816-555-1212");
  EXPECT_EQ("tel", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("+1-816-555-1212", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("+1-816-555-1212", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_6) {
  auto uri = URI::Parse(
    "telnet://192.0.2.16:80/");
  EXPECT_EQ("telnet", uri.scheme());
  EXPECT_EQ("192.0.2.16:80", uri.authority());
  EXPECT_EQ("/", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("/", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_7) {
  auto uri = URI::Parse(
    "urn:oasis:names:specification:docbook:dtd:xml:4.1.2");
  EXPECT_EQ("urn", uri.scheme());
  EXPECT_EQ("", uri.authority());
  EXPECT_EQ("oasis:names:specification:docbook:dtd:xml:4.1.2", uri.path());
  EXPECT_EQ("", uri.query());
  EXPECT_EQ("oasis:names:specification:docbook:dtd:xml:4.1.2", uri.path_and_query());
  EXPECT_EQ("", uri.fragment());
}

TEST(URI, Parse_Error1) {
  auto uri = URI::Parse(
    "telnet://192.  0.2.16:80/");
  EXPECT_FALSE(uri.valid());
}

TEST(URI, Parse_Error2) {
  auto uri = URI::Parse(
    "telnet://192.\n0.2.16:80/");
  EXPECT_FALSE(uri.valid());
}

TEST(URI, Parse_Error3) {
  auto uri = URI::Parse("noscheme");
  EXPECT_FALSE(uri.valid());
}

TEST(URI, Parse_Error4) {
  auto uri = URI::Parse(
    "http://example.com/path\a");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\a' character in the path
}

TEST(URI, Parse_Error5) {
  auto uri = URI::Parse(
    "http://example.com?query\a");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\a' character in the query
}

TEST(URI, Parse_Error6) {
  auto uri = URI::Parse(
    "http://example.com#fragment\a");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\a' character in the fragment
}

TEST(URI, Parse_Error7) {
  auto uri = URI::Parse(
    "http://example\a.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\a' character in the authority
}

TEST(URI, Parse_Error8) {
  auto uri = URI::Parse(
    "http\a://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\a' character in the scheme
}

TEST(URI, Parse_Error9) {
  auto uri = URI::Parse(
    "http://example.com/path\b");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\b' character in the path
}

TEST(URI, Parse_Error10) {
  auto uri = URI::Parse(
    "http://example.com?query\b");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\b' character in the query
}

TEST(URI, Parse_Error11) {
  auto uri = URI::Parse(
    "http://example.com#fragment\b");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\b' character in the fragment
}

TEST(URI, Parse_Error12) {
  auto uri = URI::Parse(
    "http://example\b.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\b' character in the authority
}

TEST(URI, Parse_Error13) {
  auto uri = URI::Parse(
    "http\b://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\b' character in the scheme
}

TEST(URI, Parse_Error14) {
  auto uri = URI::Parse(
    "http://example.com/path\t");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\t' character in the path
}

TEST(URI, Parse_Error15) {
  auto uri = URI::Parse(
    "http://example.com?query\t");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\t' character in the query
}

TEST(URI, Parse_Error16) {
  auto uri = URI::Parse(
    "http://example.com#fragment\t");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\t' character in the fragment
}

TEST(URI, Parse_Error17) {
  auto uri = URI::Parse(
    "http://example\t.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\t' character in the authority
}

TEST(URI, Parse_Error18) {
  auto uri = URI::Parse(
    "http\t://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\t' character in the scheme
}

TEST(URI, Parse_Error19) {
  auto uri = URI::Parse(
    "http://example.com/path\n");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\n' character in the path
}

TEST(URI, Parse_Error20) {
  auto uri = URI::Parse(
    "http://example.com?query\n");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\n' character in the query
}

TEST(URI, Parse_Error21) {
  auto uri = URI::Parse(
    "http://example.com#fragment\n");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\n' character in the fragment
}

TEST(URI, Parse_Error22) {
  auto uri = URI::Parse(
    "http://example\n.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\n' character in the authority
}

TEST(URI, Parse_Error23) {
  auto uri = URI::Parse(
    "http\n://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\n' character in the scheme
}

TEST(URI, Parse_Error24) {
  auto uri = URI::Parse(
    "http://example.com/path\v");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\v' character in the path
}

TEST(URI, Parse_Error25) {
  auto uri = URI::Parse(
    "http://example.com/path\f");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\f' character in the path
}

TEST(URI, Parse_Error26) {
  auto uri = URI::Parse(
    "http://example.com/path\r");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\r' character in the path
}

TEST(URI, Parse_Error27) {
  auto uri = URI::Parse(
    "http://example.com?query\v");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\v' character in the query
}

TEST(URI, Parse_Error28) {
  auto uri = URI::Parse(
    "http://example.com#fragment\v");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\v' character in the fragment
}

TEST(URI, Parse_Error29) {
  auto uri = URI::Parse(
    "http://example\v.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\v' character in the authority
}

TEST(URI, Parse_Error30) {
  auto uri = URI::Parse(
    "http\v://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\v' character in the scheme
}

TEST(URI, Parse_Error31) {
  auto uri = URI::Parse(
    "http://example.com?query\f");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\f' character in the query
}

TEST(URI, Parse_Error32) {
  auto uri = URI::Parse(
    "http://example.com#fragment\f");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\f' character in the fragment
}

TEST(URI, Parse_Error33) {
  auto uri = URI::Parse(
    "http://example\f.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\f' character in the authority
}

TEST(URI, Parse_Error34) {
  auto uri = URI::Parse(
    "http\f://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\f' character in the scheme
}

TEST(URI, Parse_Error35) {
  auto uri = URI::Parse(
    "http://example.com?query\r");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\r' character in the query
}

TEST(URI, Parse_Error36) {
  auto uri = URI::Parse(
    "http://example.com#fragment\r");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\r' character in the fragment
}

TEST(URI, Parse_Error37) {
  auto uri = URI::Parse(
    "http://example\r.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\r' character in the authority
}

TEST(URI, Parse_Error38) {
  auto uri = URI::Parse(
    "http\r://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\r' character in the scheme
}

TEST(URI, Parse_Error39) {
  auto uri = URI::Parse(
    "http\?://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\?' character in the scheme
}

TEST(URI, Parse_Error40) {
  auto uri = URI::Parse(
    "http://example\a.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger display_char with '\a' character in the authority
}

TEST(URI, Parse_Error60) {
  auto uri = URI::Parse(
    "://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "First character should be a letter" error
}

TEST(URI, Parse_Error41) {
  auto uri = URI::Parse(
    "http");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Expected a colon after the URI scheme" error
}

TEST(URI, Parse_Error42) {
  auto uri = URI::Parse(
    "https");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Expected a colon after the URI scheme" error
}

TEST(URI, Parse_Error43) {
  auto uri = URI::Parse(
    "http@://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (@) in scheme" error
}

TEST(URI, Parse_Error44) {
  auto uri = URI::Parse(
    "http#://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (#) in scheme" error
}

TEST(URI, Parse_Error45) {
  auto uri = URI::Parse(
    "http[://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found ([) in scheme" error
}

TEST(URI, Parse_Error46) {
  auto uri = URI::Parse(
    "http]://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (]) in scheme" error
}

TEST(URI, Parse_Error47) {
  auto uri = URI::Parse(
    "1http://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "First character should be a letter" error
}

TEST(URI, Parse_Error48) {
  auto uri = URI::Parse(
    "9http://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "First character should be a letter" error
}

TEST(URI, Parse_Error49) {
  auto uri = URI::Parse(
    "@http://example.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "First character should be a letter" error
}

TEST(URI, Parse_Error50) {
  auto uri = URI::Parse(
    "http://example\a.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\a) in authority" error
}

TEST(URI, Parse_Error51) {
  auto uri = URI::Parse(
    "http://example\b.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\b) in authority" error
}

TEST(URI, Parse_Error52) {
  auto uri = URI::Parse(
    "http://example\t.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\t) in authority" error
}

TEST(URI, Parse_Error53) {
  auto uri = URI::Parse(
    "http://example\n.com");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\n) in authority" error
}

TEST(URI, Parse_Error54) {
  auto uri = URI::Parse(
    "http://example.com/path\a");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\a) in path" error
}

TEST(URI, Parse_Error55) {
  auto uri = URI::Parse(
    "http://example.com/path\b");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\b) in path" error
}

TEST(URI, Parse_Error56) {
  auto uri = URI::Parse(
    "http://example.com/?query\a");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\a) in query" error
}

TEST(URI, Parse_Error57) {
  auto uri = URI::Parse(
    "http://example.com/?query\b");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\b) in query" error
}

TEST(URI, Parse_Error58) {
  auto uri = URI::Parse(
    "http://example.com/#fragment\a");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\a) in fragment" error
}

TEST(URI, Parse_Error59) {
  auto uri = URI::Parse(
    "http://example.com/#fragment\b");
  EXPECT_FALSE(uri.valid());
  // This should trigger "Invalid character found (\b) in fragment" error
}

TEST(URI, Parse_AllowNonEscaped) {
  auto uri = URI::Parse(
    "http://example.com/path with spaces", URI::ParseOpts::AllowNonEscaped);
  EXPECT_TRUE(uri.valid());
  // This should allow spaces in path with AllowNonEscaped option
}

}  // namespace dali
