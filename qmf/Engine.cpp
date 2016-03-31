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

#include <qmf/Engine.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace qmf {

void Engine::initAvgTestData(std::vector<size_t>& testUsers,
                             std::vector<std::vector<Double>>& testLabels,
                             std::vector<std::vector<Double>>& testScores,
                             const std::vector<DatasetElem>& testDataset,
                             const IdIndex& userIndex,
                             const IdIndex& itemIndex,
                             const size_t numTestUsers,
                             const int32_t seed) {
  std::unordered_set<size_t> userSet;
  for (const auto& elem : testDataset) {
    const size_t uidx = userIndex.idx(elem.userId);
    const size_t pidx = itemIndex.idx(elem.itemId);
    if (uidx == IdIndex::missingIdx || pidx == IdIndex::missingIdx) {
      continue;
    }
    userSet.insert(uidx);
  }

  testUsers.assign(userSet.begin(), userSet.end());
  if (numTestUsers > 0 && numTestUsers < testUsers.size()) {
    std::shuffle(testUsers.begin(), testUsers.end(), std::mt19937(seed));
    testUsers.erase(testUsers.begin() + numTestUsers, testUsers.end());
    testUsers.shrink_to_fit();
  }

  // map for userIdx -> index in testUsers
  std::unordered_map<size_t, size_t> userMap;
  testLabels.reserve(testUsers.size());
  testScores.reserve(testUsers.size());
  for (size_t i = 0; i < testUsers.size(); ++i) {
    userMap[testUsers[i]] = i;
    testLabels.emplace_back(itemIndex.size());
    testScores.emplace_back(itemIndex.size());
  }

  for (const auto& elem : testDataset) {
    const size_t uidx = userIndex.idx(elem.userId);
    const size_t pidx = itemIndex.idx(elem.itemId);
    if (uidx == IdIndex::missingIdx || pidx == IdIndex::missingIdx ||
        userMap.count(uidx) == 0) {
      continue;
    }
    testLabels[userMap[uidx]][pidx] = elem.value;
  }
}

void Engine::computeTestScores(std::vector<std::vector<Double>>& testScores,
                               const std::vector<size_t>& testUsers,
                               const FactorData& userFactors,
                               const FactorData& itemFactors,
                               ParallelExecutor& parallel) {

  const size_t ntasks = testUsers.size();
  auto func =
    [&testUsers, &testScores, &userFactors, &itemFactors](const size_t taskId) {
      const size_t uidx = testUsers[taskId];
      const size_t nfactors = userFactors.nfactors();
      auto& scores = testScores[taskId];
      for (size_t idx = 0; idx < itemFactors.nelems(); ++idx) {
        scores[idx] =
          itemFactors.withBiases() ? itemFactors.biasAt(idx) : 0.0;
        for (size_t fidx = 0; fidx < nfactors; ++fidx) {
          scores[idx] +=
            userFactors.at(uidx, fidx) * itemFactors.at(idx, fidx);
        }
      }
    };

  parallel.execute(ntasks, func);
}

void Engine::saveFactors(const FactorData& factorData,
                         const IdIndex& index,
                         const std::string& fileName) {
  std::ofstream fout(fileName);
  saveFactors(factorData, index, fout);
}

void Engine::saveFactors(const FactorData& factorData,
                         const IdIndex& index,
                         std::ostream& out) {
  CHECK_EQ(factorData.nelems(), index.size());
  out << std::fixed;
  out << std::setprecision(9);
  for (size_t idx = 0; idx < factorData.nelems(); ++idx) {
    const int64_t id = index.id(idx);
    out << id;
    if (factorData.withBiases()) {
      out << ' ' << factorData.biasAt(idx);
    }
    for (size_t fidx = 0; fidx < factorData.nfactors(); ++fidx) {
      out << ' ' << factorData.at(idx, fidx);
    }
    out << '\n';
  }
}
}
