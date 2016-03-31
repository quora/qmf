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

namespace qmf {

template <typename FuncT, typename GenT>
void BPREngine::iterate(FuncT func, const size_t numNeg, GenT&& gen) const {
  for (const auto& elem : data_) {
    for (size_t i = 0; i < numNeg; ++i) {
      func(PosNegTriplet{
        elem.userIdx,
        elem.posItemIdx,
        sampleRandomNegative(elem.userIdx, std::forward<GenT>(gen))});
    }
  }
}

template <typename FuncT, typename GenT>
void BPREngine::iterateBlock(FuncT func,
                             const size_t start,
                             const size_t end,
                             const size_t numNeg,
                             GenT&& gen) const {
  for (size_t i = start; i < end; ++i) {
    const auto& elem = data_[i];
    for (size_t j = 0; j < numNeg; ++j) {
      func(PosNegTriplet{
        elem.userIdx,
        elem.posItemIdx,
        sampleRandomNegative(elem.userIdx, std::forward<GenT>(gen))});
    }
  }
}

template <typename GenT>
size_t BPREngine::sampleRandomNegative(const size_t userIdx,
                                       GenT&& gen,
                                       const bool useTestItemMap) const {
  const auto& userPosSet =
    useTestItemMap ? testItemMap_.at(userIdx) : itemMap_.at(userIdx);
  std::uniform_int_distribution<> dis(0, static_cast<int>(nitems()) - 1);
  size_t negIdx;
  do {
    negIdx = dis(gen);
  } while (userPosSet.count(negIdx) > 0);
  return negIdx;
}
}
