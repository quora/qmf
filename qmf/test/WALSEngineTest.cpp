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

#include <random>

#include <qmf/wals/WALSEngine.h>

#include <gtest/gtest.h>

namespace qmf {

namespace {
std::unique_ptr<MetricsEngine> kNullMetricEngine = nullptr;
}

TEST(WALSEngine, init) {
  WALSConfig config;
  config.nfactors = 30;
  WALSEngine engine(config, kNullMetricEngine);

  std::vector<DatasetElem> dataset = {
    {1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 4}};
  engine.init(dataset);

  // check user data
  EXPECT_EQ(engine.nusers(), 3);
  EXPECT_EQ(engine.userFactors_->nelems(), 3);
  EXPECT_EQ(engine.userFactors_->nfactors(), 30);
  EXPECT_EQ(engine.userSignals_.size(), 3);
  // group 0
  EXPECT_EQ(engine.userSignals_[0].sourceId, 1);
  EXPECT_EQ(engine.userSignals_[0].group.size(), 3);
  EXPECT_EQ(engine.userSignals_[0].group[0].id, 1);
  EXPECT_EQ(engine.userSignals_[0].group[1].id, 2);
  EXPECT_EQ(engine.userSignals_[0].group[2].id, 3);
  // group 1
  EXPECT_EQ(engine.userSignals_[1].sourceId, 2);
  EXPECT_EQ(engine.userSignals_[1].group.size(), 2);
  EXPECT_EQ(engine.userSignals_[1].group[0].id, 1);
  EXPECT_EQ(engine.userSignals_[1].group[1].id, 3);
  // group 2
  EXPECT_EQ(engine.userSignals_[2].sourceId, 3);
  EXPECT_EQ(engine.userSignals_[2].group.size(), 1);
  EXPECT_EQ(engine.userSignals_[2].group[0].id, 4);

  // check item data
  EXPECT_EQ(engine.nitems(), 4);
  EXPECT_EQ(engine.itemFactors_->nelems(), 4);
  EXPECT_EQ(engine.itemFactors_->nfactors(), 30);
  // group 0
  EXPECT_EQ(engine.itemSignals_[0].sourceId, 1);
  EXPECT_EQ(engine.itemSignals_[0].group.size(), 2);
  EXPECT_EQ(engine.itemSignals_[0].group[0].id, 1);
  EXPECT_EQ(engine.itemSignals_[0].group[1].id, 2);
  // group 1
  EXPECT_EQ(engine.itemSignals_[1].sourceId, 2);
  EXPECT_EQ(engine.itemSignals_[1].group.size(), 1);
  EXPECT_EQ(engine.itemSignals_[1].group[0].id, 1);
  // group 2
  EXPECT_EQ(engine.itemSignals_[2].sourceId, 3);
  EXPECT_EQ(engine.itemSignals_[2].group.size(), 2);
  EXPECT_EQ(engine.itemSignals_[2].group[0].id, 1);
  EXPECT_EQ(engine.itemSignals_[2].group[1].id, 2);
  // group 1
  EXPECT_EQ(engine.itemSignals_[3].sourceId, 4);
  EXPECT_EQ(engine.itemSignals_[3].group.size(), 1);
  EXPECT_EQ(engine.itemSignals_[3].group[0].id, 3);

  // can't init twice
  EXPECT_DEATH(engine.init(dataset), ".*");
}

TEST(WALSEngine, initTest) {
  WALSConfig config;
  config.nfactors = 30;
  const auto metricsEngine = std::make_unique<MetricsEngine>();
  metricsEngine->addTestAvgMetric("auc");
  WALSEngine engine(config, metricsEngine);

  std::vector<DatasetElem> dataset = {
    {1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 4}};
  engine.init(dataset);

  std::vector<DatasetElem> testDataset = {{1, 4}, {2, 1}, {4, 2}};
  engine.initTest(testDataset);

  EXPECT_EQ(engine.nusers(), 3);
  EXPECT_EQ(engine.nitems(), 4);

  // check test sizes
  EXPECT_EQ(engine.testUsers_.size(), 2);
  EXPECT_EQ(engine.testLabels_.size(), 2);
  EXPECT_EQ(engine.testScores_.size(), 2);

  // can't init twice
  EXPECT_DEATH(engine.initTest(testDataset), ".*");
}

TEST(WALSEngine, computeXtX) {
  const std::vector<size_t> nthreadsTests {1, 2, 3, 5, 7, 8, 10, 16, 32};
  const size_t nfactors = 5;
  const size_t n = 17;
  for (size_t nthreads : nthreadsTests) {
    WALSConfig config;
    config.nfactors = nfactors;
    WALSEngine engine(config, kNullMetricEngine, nthreads);

    std::mt19937 gen(123);
    std::uniform_real_distribution<Double> distr(-1.0, 1.0);
    Matrix X(n, nfactors);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < nfactors; ++j) {
        X(i, j) = distr(gen);
      }
    }

    Matrix XtX = engine.computeXtX(X);
    EXPECT_EQ(XtX.nrows(), nfactors);
    EXPECT_EQ(XtX.ncols(), nfactors);
    for (size_t i = 0; i < nfactors; ++i) {
      for (size_t j = 0; j < nfactors; ++j) {
        Double value = 0.0;
        for (size_t k = 0; k < n; ++k) {
          value += X(k, i) * X(k, j);
        }
        EXPECT_NEAR(XtX(i, j), value, 1e-8);
      }
    }
  }
}

TEST(WALSEngine, updateFactorsForOne) {
  const size_t nusers = 3;
  const size_t nitems = 2;
  const size_t nfactors = 3;

  Matrix X(nusers, nfactors);
  Matrix Y(nitems, nfactors);
  for (size_t i = 0; i < nitems; ++i) {
    for (size_t j = 0; j < nfactors; ++j) {
      Y(i, j) = 0.1;
    }
  }

  IdIndex userIndex;
  for (size_t i = 0; i < nusers; ++i) {
    userIndex.getOrSetIdx(static_cast<int64_t>(i));
  }
  IdIndex itemIndex;
  for (size_t i = 0; i < nitems; ++i) {
    itemIndex.getOrSetIdx(static_cast<int64_t>(i));
  }
  Matrix YtY(nfactors, nfactors);
  for (size_t i = 0; i < nfactors; ++i) {
    for (size_t j = 0; j < nfactors; ++j) {
      for (size_t k = 0; k < nitems; ++k) {
        YtY(i, j) += Y(k, i) * Y(k, j);
      }
    }
  }

  WALSEngine::SignalGroup signalGroup{0, {{0, 1.0}, {1, 1.0}}};

  const Double loss = WALSEngine::updateFactorsForOne(
    X, userIndex, Y, itemIndex, signalGroup, YtY, 1.0, 1.0);

  for (size_t i = 0; i < nfactors; ++i) {
    EXPECT_NEAR(X(0, i), 0.357, 1e-2);
  }

  for (size_t i = 1; i < nusers; ++i) {
    for (size_t j = 0; j < nfactors; ++j) {
      EXPECT_NEAR(X(i, j), 0.0, 1e-8);
    }
  }

  Double trueLoss = 0.0;
  for (size_t i = 0; i < nusers; ++i) {
    for (size_t j = 0; j < nitems; ++j) {
      Double pred = 0.0;
      for (size_t k = 0; k < nfactors; ++k) {
        pred += X(i, k) * Y(j, k);
      }
      if (i == 0) { // both items liked for this user
        trueLoss += 2.0 * (1.0 - pred) * (1.0 - pred);
      } else {
        trueLoss += (0.0 - pred) * (0.0 - pred);
      }
    }
  }
  EXPECT_NEAR(loss, trueLoss, 1e-2);
}
}
