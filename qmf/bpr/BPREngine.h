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
#include <random>
#include <vector>
#include <unordered_map>
#include <unordered_set>

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

struct BPRConfig {
  size_t nepochs;
  size_t nfactors;
  Double initLearningRate;
  Double biasLambda;
  Double userLambda;
  Double itemLambda;
  Double decayRate;
  bool useBiases;
  Double initDistributionBound;
  size_t numNegativeSamples;
  size_t numHogwildThreads;
  bool shuffleTrainingSet;
};

class BPREngine : public Engine {
 public:
  explicit BPREngine(const BPRConfig& config,
                     const std::unique_ptr<MetricsEngine>& metricsEngine,
                     const size_t evalNumNeg = 3,
                     const int32_t evalSeed = 42,
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
  struct PosPair {
    size_t userIdx;
    size_t posItemIdx;
  };

  struct PosNegTriplet {
    size_t userIdx;
    size_t posItemIdx;
    size_t negItemIdx;
  };

  // for storing the set of positive items of a given user
  using ItemMap = std::vector<std::unordered_set<size_t>>;

  // sgd update on an example triplet
  void update(const PosNegTriplet& triplet);

  // compute score difference
  Double predictDifference(const size_t userIdx,
                           const size_t posItemIdx,
                           const size_t negItemIdx) const;

  // loss function
  Double loss(const Double scoreDifference) const;

  // derivative of the loss
  Double lossDerivative(const Double scoreDifference) const;

  // randomly shuffle dataset
  void shuffle();

  template <typename FuncT, typename GenT>
  void iterate(FuncT func, const size_t numNeg, GenT&& gen) const;

  template <typename FuncT, typename GenT>
  void iterateBlock(FuncT func,
                    const size_t start,
                    const size_t end,
                    const size_t numNeg,
                    GenT&& gen) const;

  template <typename GenT>
  size_t sampleRandomNegative(const size_t userIdx,
                              GenT&& gen,
                              const bool useTestItemMap = false) const;

  const BPRConfig& config_;
  const std::unique_ptr<MetricsEngine>& metricsEngine_;
  const size_t evalNumNeg_;
  const int32_t evalSeed_;

  ParallelExecutor parallel_;

  std::mt19937 gen_;

  Double learningRate_;

  std::vector<PosPair> data_;

  std::vector<PosNegTriplet> evalSet_;
  std::vector<PosNegTriplet> testEvalSet_;

  ItemMap itemMap_;
  ItemMap testItemMap_;

  IdIndex userIndex_;
  IdIndex itemIndex_;

  std::unique_ptr<FactorData> userFactors_;
  std::unique_ptr<FactorData> itemFactors_;

  std::vector<size_t> testUsers_; // indexes of test users
  std::vector<std::vector<Double>> testLabels_;
  std::vector<std::vector<Double>> testScores_;

  // for unit tests
  FRIEND_TEST(BPREngine, init);
  FRIEND_TEST(BPREngine, optimize);
};
}

#include <qmf/bpr/BPREngine-inl.h>
