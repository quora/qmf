/*
 * Copyright 2016 Quora, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <vector>

#include <qmf/utils/Util.h>

#include <gtest/gtest.h>

using qmf::split;

TEST(TestUtil, split) {
  using vec = std::vector<std::string>;
  EXPECT_EQ(split("", ','), vec({}));
  EXPECT_EQ(split("hello", ','), vec({"hello"}));
  EXPECT_EQ(split("hello,world", ','), vec({"hello", "world"}));
  EXPECT_EQ(split("hello,world,!", ','), vec({"hello", "world", "!"}));
  EXPECT_EQ(split("hello world !", ' '), vec({"hello", "world", "!"}));
}
