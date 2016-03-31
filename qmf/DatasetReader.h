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

#pragma once

#include <istream>
#include <memory>
#include <string>

#include <qmf/Types.h>

#include <gtest/gtest.h>

namespace qmf {

struct DatasetElem {
  int64_t userId;
  int64_t itemId;
  Double value = 1.0;
};

class DatasetReader {
 public:
  // for unit tests
  DatasetReader() = default;

  explicit DatasetReader(const std::string& fileName);

  // reads one line from the file
  bool readOne(DatasetElem& elem);

  // reads entire file
  std::vector<DatasetElem> readAll();

 private:
  std::unique_ptr<std::istream> stream_;

  std::string line_;

  // for unit tests
  FRIEND_TEST(DatasetReader, readOne);
  FRIEND_TEST(DatasetReader, readOneBadFormat);
  FRIEND_TEST(DatasetReader, readAll);
};
}
