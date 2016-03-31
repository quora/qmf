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

#include <sstream>

#include <qmf/DatasetReader.h>

#include <gtest/gtest.h>

namespace qmf {

TEST(DatasetReader, readOne) {
  DatasetReader reader;
  reader.stream_ = std::make_unique<std::istringstream>("1 2 3");
  DatasetElem elem;
  EXPECT_TRUE(reader.readOne(elem));
  EXPECT_EQ(elem.userId, 1);
  EXPECT_EQ(elem.itemId, 2);
  EXPECT_DOUBLE_EQ(elem.value, 3);
  EXPECT_FALSE(reader.readOne(elem));
}

TEST(DatasetReader, readOneBadFormat) {
  DatasetReader reader;
  reader.stream_ = std::make_unique<std::istringstream>("1 3");
  DatasetElem elem;
  EXPECT_DEATH(reader.readOne(elem), ".*");
}

TEST(DatasetReader, readAll) {
  DatasetReader reader;
  std::string str;
  const int nelems = 5;
  for (int i = 0; i < nelems; ++i) {
    str += "1 2 3\n";
  }
  reader.stream_ = std::make_unique<std::istringstream>(str);
  std::vector<DatasetElem> dataset = reader.readAll();
  EXPECT_EQ(dataset.size(), nelems);
  for (auto& elem : dataset) {
    EXPECT_EQ(elem.userId, 1);
    EXPECT_EQ(elem.itemId, 2);
    EXPECT_DOUBLE_EQ(elem.value, 3);
  }
}
}
