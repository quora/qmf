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

#include <vector>

#include <qmf/Types.h>

namespace qmf {

class Vector {
 public:
  explicit Vector(const size_t n);

  Double operator()(const size_t i) const {
    return data_[i];
  }

  Double& operator()(const size_t i) {
    return data_[i];
  }

  size_t size() const {
    return data_.size();
  }

  Double* const data() {
    return &data_[0];
  }

 private:
  std::vector<Double> data_;
};
}
