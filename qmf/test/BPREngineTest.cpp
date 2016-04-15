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

#include <qmf/bpr/BPREngine.h>

#include <gtest/gtest.h>

namespace qmf {

namespace {
std::unique_ptr<MetricsEngine> kNullMetricEngine = nullptr;
}

TEST(BPREngine, init) {
  BPRConfig config;
  config.nfactors = 30;
  config.initDistributionBound = 0.1;
  BPREngine engine(config, kNullMetricEngine, /*evalNumNeg=*/2);

  std::vector<DatasetElem> dataset = {{3, 2}, {5, 2}, {3, 4}, {6, 2}, {7, 10}};
  engine.init(dataset);

  EXPECT_EQ(engine.nusers(), 4);
  EXPECT_EQ(engine.userFactors_->nelems(), 4);
  EXPECT_EQ(engine.userFactors_->nfactors(), 30);
  EXPECT_EQ(engine.nitems(), 3);
  EXPECT_EQ(engine.itemFactors_->nelems(), 3);
  EXPECT_EQ(engine.itemFactors_->nfactors(), 30);

  EXPECT_EQ(engine.data_.size(), dataset.size());
  EXPECT_EQ(engine.itemMap_.size(), engine.nusers());

  // check id indexes and item map
  const size_t uidx = engine.userIndex_.idx(3);
  EXPECT_EQ(engine.itemMap_[uidx].size(), 2);
  EXPECT_GT(engine.itemMap_[uidx].count(engine.itemIndex_.idx(2)), 0);
  EXPECT_GT(engine.itemMap_[uidx].count(engine.itemIndex_.idx(4)), 0);

  // check eval set
  EXPECT_EQ(engine.evalSet_.size(), 2 * dataset.size());
  for (const auto& triplet : engine.evalSet_) {
    const size_t uidx = triplet.userIdx;
    EXPECT_GT(engine.itemMap_[uidx].count(triplet.posItemIdx), 0);
    EXPECT_EQ(engine.itemMap_[uidx].count(triplet.negItemIdx), 0);
  }

  // test dataset
  std::vector<DatasetElem> testDataset = {{5, 4}, {3, 10}, {6, 12}, {8, 13}};
  // only the first 2 examples are valid in the training data
  engine.initTest(testDataset);
  // training item map shouldn't be affected
  EXPECT_EQ(engine.itemMap_[uidx].size(), 2);

  EXPECT_EQ(engine.testItemMap_.size(), engine.nusers());
  EXPECT_EQ(engine.testItemMap_[uidx].size(), 1);
  EXPECT_GT(engine.testItemMap_[uidx].count(engine.itemIndex_.idx(10)), 0);

  // check test eval set
  EXPECT_EQ(engine.testEvalSet_.size(), 2 * 2);
  for (const auto& triplet : engine.testEvalSet_) {
    const size_t uidx = triplet.userIdx;
    EXPECT_GT(engine.testItemMap_[uidx].count(triplet.posItemIdx), 0);
    EXPECT_EQ(engine.testItemMap_[uidx].count(triplet.negItemIdx), 0);
  }
}

TEST(BPREngine, optimize) {
  const int logLevel = FLAGS_minloglevel;
  FLAGS_minloglevel = 2;
  BPRConfig config;
  config.nepochs = 40;
  config.nfactors = 1;
  config.initLearningRate = 0.1;
  config.decayRate = 1.0;
  config.initDistributionBound = 0.1;
  config.numNegativeSamples = 1;

  int totalChecks = 0;
  int successChecks = 0;
  // check that preferences are correctly ordered
  auto checkPreference = [&totalChecks, &successChecks](
    const BPREngine& engine, const int64_t userId, const int64_t posItemId,
    const int64_t negItemId) {

    ++totalChecks;
    if (engine.predictDifference(engine.userIndex_.idx(userId),
                                 engine.itemIndex_.idx(posItemId),
                                 engine.itemIndex_.idx(negItemId)) > 0.0) {
      ++successChecks;
    }
  };

  for (size_t trial = 0; trial < 10; ++trial) {
    BPREngine engine(config, kNullMetricEngine, /*evalNumNeg=*/1);
    std::vector<DatasetElem> dataset = {{1, 1}, {2, 2}};
    engine.init(dataset);
    engine.optimize();

    checkPreference(engine, 1, 1, 2);
    checkPreference(engine, 2, 2, 1);
  }
  EXPECT_GT(successChecks, 0.9 * totalChecks);

  totalChecks = 0;
  successChecks = 0;
  for (size_t trial = 0; trial < 10; ++trial) {
    BPREngine engine(config, kNullMetricEngine, /*evalNumNeg=*/1);
    std::vector<DatasetElem> dataset = {{1, 1}, {1, 3}, {2, 2}, {3, 1}};
    engine.init(dataset);
    engine.optimize();

    // check that preferences are correctly ordered
    checkPreference(engine, 1, 1, 2);
    checkPreference(engine, 1, 3, 2);
    checkPreference(engine, 2, 2, 1);
    checkPreference(engine, 2, 2, 3);
    checkPreference(engine, 3, 1, 2);
    checkPreference(engine, 3, 3, 2);
  }
  EXPECT_GT(successChecks, 0.9 * totalChecks);

  totalChecks = 0;
  successChecks = 0;
  config.nfactors = 3;
  for (size_t trial = 0; trial < 10; ++trial) {
    BPREngine engine(config, kNullMetricEngine, /*evalNumNeg=*/1);
    std::vector<DatasetElem> dataset = {{1, 1}, {1, 3}, {2, 2}, {3, 1}};
    engine.init(dataset);
    engine.optimize();

    // check that preferences are correctly ordered
    checkPreference(engine, 1, 1, 2);
    checkPreference(engine, 1, 3, 2);
    checkPreference(engine, 2, 2, 1);
    checkPreference(engine, 2, 2, 3);
    checkPreference(engine, 3, 1, 2);
    checkPreference(engine, 3, 3, 2);
  }
  EXPECT_GT(successChecks, 0.9 * totalChecks);

  FLAGS_minloglevel = logLevel;
}
}
