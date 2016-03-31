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

#include <memory>
#include <vector>

#include <qmf/Engine.h>
#include <qmf/FactorData.h>
#include <qmf/metrics/MetricsEngine.h>
#include <qmf/Types.h>
#include <qmf/utils/IdIndex.h>
#include <qmf/utils/ParallelExecutor.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace qmf {

struct WALSConfig {
  size_t nepochs;
  size_t nfactors;
  Double regularizationLambda;
  Double confidenceWeight;
  Double initDistributionBound;
};

class WALSEngine : public Engine {
 public:
  explicit WALSEngine(
    const WALSConfig& config,
    const std::unique_ptr<MetricsEngine>& metricsEngine,
    const size_t nthreads = 16);

  void init(const std::vector<DatasetElem>& dataset) override;

  void initTest(const std::vector<DatasetElem>& testDataset) override;

  void optimize() override;

  void evaluate(const size_t epoch) override;

  size_t nusers() const;

  size_t nitems() const;

  void saveUserFactors(const std::string& fileName) const override;

  void saveItemFactors(const std::string& fileName) const override;

 private:
  struct Signal {
    int64_t id;
    Double value;
  };

  struct SignalGroup {
    int64_t sourceId;
    std::vector<Signal> group;
  };

  static void groupSignals(std::vector<SignalGroup>& signals,
                           IdIndex& index,
                           std::vector<DatasetElem>& dataset);

  static void sortDataset(std::vector<DatasetElem>& dataset);

  Double iterate(FactorData& leftData,
                 const IdIndex& leftIndex,
                 const std::vector<SignalGroup>& leftSignals,
                 const FactorData& rightData,
                 const IdIndex& rightIndex);

  Matrix computeXtX(const Matrix& X);

  /*
   *
   */
  static Double updateFactorsForOne(Matrix& X,
                                    const IdIndex& leftIndex,
                                    const Matrix& Y,
                                    const IdIndex& rightIndex,
                                    const SignalGroup& signalGroup,
                                    Matrix A,
                                    const Double alpha,
                                    const Double lambda);

  const WALSConfig& config_;

  const std::unique_ptr<MetricsEngine>& metricsEngine_;

  ParallelExecutor parallel_;

  // indexes
  IdIndex userIndex_;
  IdIndex itemIndex_;

  // factors
  std::unique_ptr<FactorData> userFactors_;
  std::unique_ptr<FactorData> itemFactors_;

  // signals
  std::vector<SignalGroup> userSignals_;
  std::vector<SignalGroup> itemSignals_;

  // test data
  std::vector<size_t> testUsers_; // indexes of test users
  std::vector<std::vector<Double>> testLabels_;
  std::vector<std::vector<Double>> testScores_;

  // for unit tests
  FRIEND_TEST(WALSEngine, init);
  FRIEND_TEST(WALSEngine, initTest);
  FRIEND_TEST(WALSEngine, computeXtX);
  FRIEND_TEST(WALSEngine, updateFactorsForOne);
};
}
