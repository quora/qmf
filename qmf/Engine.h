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
#include <iomanip>

#include <qmf/DatasetReader.h>
#include <qmf/FactorData.h>
#include <qmf/Types.h>
#include <qmf/utils/IdIndex.h>
#include <qmf/utils/ParallelExecutor.h>

#include <gtest/gtest.h>

namespace qmf {

class Engine {
 public:
  Engine() = default;
  virtual ~Engine() = default;

  Engine(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine& operator=(Engine&&) = delete;

  // for initialization
  virtual void init(const std::vector<DatasetElem>& dataset) {
  }

  // for initialization from test data
  virtual void initTest(const std::vector<DatasetElem>& testDataset) {
  }

  // for running the optimizer
  virtual void optimize() {
  }

  // for evaluation after each epoch
  virtual void evaluate(const size_t epoch) {
  }

  // for saving user factors to a file
  virtual void saveUserFactors(const std::string& fileName) const {
  }

  // for saving item factors to a file
  virtual void saveItemFactors(const std::string& fileName) const {
  }

 protected:
  // initialize test data for evaluating test averaged metrics
  static void initAvgTestData(std::vector<size_t>& testUsers,
                              std::vector<std::vector<Double>>& testLabels,
                              std::vector<std::vector<Double>>& testScores,
                              const std::vector<DatasetElem>& testDataset,
                              const IdIndex& userIndex,
                              const IdIndex& itemIndex,
                              const size_t numTestUsers = 0,
                              const int32_t seed = 0);

  // compute predicted scores for all items and all test users
  static void computeTestScores(std::vector<std::vector<Double>>& testScores,
                                const std::vector<size_t>& testUsers,
                                const FactorData& userFactors,
                                const FactorData& itemFactors,
                                ParallelExecutor& parallel);

  static void saveFactors(const FactorData& factorData,
                          const IdIndex& index,
                          const std::string& fileName);

  static void saveFactors(const FactorData& factorData,
                          const IdIndex& index,
                          std::ostream& out);

  // for unit tests
  FRIEND_TEST(Engine, initAvgTestData);
  FRIEND_TEST(Engine, computeTestScores);
  FRIEND_TEST(Engine, saveFactors);
};
}
