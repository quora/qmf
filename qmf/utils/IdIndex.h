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

#include <limits>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
using std::size_t;
using std::int64_t;

namespace qmf {

// helper class for converting from raw ids ("id") to contiguous indices
// ("idx").
class IdIndex {
 public:
  static const size_t missingIdx = std::numeric_limits<size_t>::max();

  IdIndex() = default;

  int64_t id(const size_t idx) const {
    return ids_[idx];
  }

  size_t idx(const int64_t id) const {
    const auto pos = idxMap_.find(id);
    return (pos != idxMap_.end() ? pos->second : missingIdx);
  }

  // returns idx if it is present, otherwise adds an entry.
  size_t getOrSetIdx(const int64_t id);

  size_t size() const {
    return ids_.size();
  }

  const std::vector<int64_t>& ids() const {
    return ids_;
  }

 private:
  std::vector<int64_t> ids_;

  std::unordered_map<int64_t, size_t> idxMap_;
};
}
