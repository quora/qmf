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

#include <fstream>

#include <qmf/DatasetReader.h>

#include <glog/logging.h>

namespace qmf {

DatasetReader::DatasetReader(const std::string& fileName)
  : stream_(std::make_unique<std::ifstream>(fileName)) {}

bool DatasetReader::readOne(DatasetElem& elem) {
  CHECK(stream_);
  if (!std::getline(*stream_, line_)) {
    return false;
  }
  int32_t value = 0;
  const int result = sscanf(line_.c_str(), "%ld %ld %d", &elem.userId,
      &elem.itemId, &value);
  CHECK_EQ(result, 3) << "the file format is incorrect: " << line_;
  elem.value = static_cast<Double>(value);
  return true;
}

std::vector<DatasetElem> DatasetReader::readAll() {
  std::vector<DatasetElem> dataset;
  DatasetElem elem;
  while (readOne(elem)) {
    dataset.push_back(elem);
  }
  return dataset;
}
}
