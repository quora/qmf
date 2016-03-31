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

#include <qmf/Vector.h>

#include <gtest/gtest.h>

TEST(Vector, basic) {
  qmf::Vector v(5);

  for (size_t i = 0; i < v.size(); ++i) {
    v(i) = i;
  }
  EXPECT_EQ(v.size(), 5);
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_DOUBLE_EQ(v(i), i);
  }

  EXPECT_ANY_THROW(qmf::Vector(-1));
}


