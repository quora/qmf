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

#include <gtest/gtest.h>

namespace qmf {

TEST(Engine, initAvgTestData) {
  Engine engine;

  IdIndex userIndex;
  IdIndex itemIndex;

  userIndex.getOrSetIdx(1);
  userIndex.getOrSetIdx(2);
  userIndex.getOrSetIdx(3);

  itemIndex.getOrSetIdx(1);
  itemIndex.getOrSetIdx(2);
  itemIndex.getOrSetIdx(4);
  itemIndex.getOrSetIdx(3);

  EXPECT_EQ(userIndex.size(), 3);
  EXPECT_EQ(itemIndex.size(), 4);

  // only first two elements should be valid
  std::vector<DatasetElem> testDataset = {{1, 4}, {2, 1}, {4, 2}, {1, 5}};

  std::vector<size_t> testUsers;
  std::vector<std::vector<Double>> testLabels;
  std::vector<std::vector<Double>> testScores;

  engine.initAvgTestData(
    testUsers, testLabels, testScores, testDataset, userIndex, itemIndex);

  EXPECT_EQ(testUsers.size(), 2);
  EXPECT_EQ(testLabels.size(), 2);
  EXPECT_EQ(testScores.size(), 2);

  for (size_t i = 0; i < testUsers.size(); ++i) {
    EXPECT_EQ(testLabels[i].size(), itemIndex.size());
    EXPECT_EQ(testScores[i].size(), itemIndex.size());
    for (auto score : testScores[i]) {
      EXPECT_DOUBLE_EQ(score, 0.0);
    }
  }

  // check labels for user 1
  auto labels = testLabels[testUsers[userIndex.idx(1)]];
  for (size_t i = 0; i < itemIndex.size(); ++i) {
    EXPECT_DOUBLE_EQ(labels[i], (i == itemIndex.idx(4)) ? 1.0 : 0.0);
  }
  // check labels for user 2
  labels = testLabels[testUsers[userIndex.idx(2)]];
  for (size_t i = 0; i < itemIndex.size(); ++i) {
    EXPECT_DOUBLE_EQ(labels[i], (i == itemIndex.idx(1)) ? 1.0 : 0.0);
  }
}

TEST(Engine, computeTestScores) {
  const size_t nfactors = 3;
  const size_t nusers = 4;
  const size_t nitems = 5;
  const std::vector<size_t> testUsers = {2, 0};

  FactorData userFactors(nusers, nfactors);
  FactorData itemFactors(nitems, nfactors, /*useBiases=*/true);
  int val = 0;
  auto setter = [&val](auto...) { return ++val; };
  userFactors.setFactors(setter);
  itemFactors.setFactors(setter);
  itemFactors.setBiases(setter);

  Engine engine;

  std::vector<std::vector<Double>> testScores(
    testUsers.size(), std::vector<Double>(nitems));

  for (auto nthreads : {1, 2, 4}) {
    ParallelExecutor parallel(nthreads);
    engine.computeTestScores(
      testScores, testUsers, userFactors, itemFactors, parallel);

    for (size_t i = 0; i < testScores.size(); ++i) {
      const size_t uidx = testUsers[i];
      for (size_t idx = 0; idx < nitems; ++idx) {
        Double res = itemFactors.biasAt(idx);
        for (size_t fidx = 0; fidx < nfactors; ++fidx) {
          res += userFactors.at(uidx, fidx) * itemFactors.at(idx, fidx);
        }
        CHECK_GT(res, 0.0);
        EXPECT_DOUBLE_EQ(testScores[i][idx], res);
      }
    }
  }
}

TEST(Engine, saveFactors) {
  const size_t nitems = 2;
  const size_t nfactors = 3;
  IdIndex index;
  index.getOrSetIdx(3);
  index.getOrSetIdx(5);
  {
    FactorData factorData(nitems, nfactors);
    factorData.setFactors([](size_t i, size_t j) { return i * nfactors + j; });
    std::ostringstream sout;
    Engine::saveFactors(factorData, index, sout);
    EXPECT_EQ(sout.str(), "3 0.000000000 1.000000000 2.000000000\n5 "
                          "3.000000000 4.000000000 5.000000000\n");
  }

  {
    FactorData factorData(nitems, nfactors, /*withBiases=*/true);
    factorData.setFactors([](size_t i, size_t j) { return i * nfactors + j; });
    factorData.setBiases([](size_t i) { return 5 + i; });
    std::ostringstream sout;
    Engine::saveFactors(factorData, index, sout);
    EXPECT_EQ(
      sout.str(),
      "3 5.000000000 0.000000000 1.000000000 2.000000000\n5 6.000000000 "
      "3.000000000 4.000000000 5.000000000\n");
  }
}
}
