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

#include <algorithm>
#include <random>

#include <qmf/wals/WALSEngine.h>

namespace qmf {

WALSEngine::WALSEngine(const WALSConfig& config,
                       const std::unique_ptr<MetricsEngine>& metricsEngine,
                       const size_t nthreads)
  : config_(config),
    metricsEngine_(metricsEngine),
    parallel_(nthreads) {
  if (metricsEngine_ && !metricsEngine_->testAvgMetrics().empty() &&
      metricsEngine_->config().numTestUsers == 0) {
    LOG(WARNING) << "computing average test metrics on all users can be slow! "
                    "Set numTestUsers > 0 to sample some of them";
  }
}

void WALSEngine::init(const std::vector<DatasetElem>& dataset) {
  CHECK(!userFactors_ && !itemFactors_)
    << "engine was already initialized with train data";
  auto mutableDataset = dataset;
  groupSignals(userSignals_, userIndex_, mutableDataset);
  // swap userId with itemId
  for (auto& elem : mutableDataset) {
    std::swap(elem.userId, elem.itemId);
  }
  groupSignals(itemSignals_, itemIndex_, mutableDataset);

  userFactors_ = std::make_unique<FactorData>(nusers(), config_.nfactors);
  itemFactors_ = std::make_unique<FactorData>(nitems(), config_.nfactors);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Double> distr(
    -config_.initDistributionBound, config_.initDistributionBound);
  auto genUnif = [&distr, &gen](auto...) { return distr(gen); };
  // don't need to initialize user factors
  itemFactors_->setFactors(genUnif);
}

void WALSEngine::initTest(const std::vector<DatasetElem>& testDataset) {
  CHECK(testUsers_.empty()) << "engine was already initialized with test data";

  // initialize data for test average metrics
  if (metricsEngine_ && !metricsEngine_->testAvgMetrics().empty()) {
    initAvgTestData(
      testUsers_, testLabels_, testScores_, testDataset, userIndex_, itemIndex_,
      metricsEngine_->config().numTestUsers, metricsEngine_->config().seed);
  }
}

void WALSEngine::optimize() {
  CHECK(userFactors_ && itemFactors_)
    << "no factor data, have you initialized the engine?";

  for (size_t epoch = 1; epoch <= config_.nepochs; ++epoch) {
    // fix item factors, update user factors
    iterate(*userFactors_, userIndex_, userSignals_, *itemFactors_, itemIndex_);
    // fix user factors, update item factors
    const Double loss = iterate(*itemFactors_, itemIndex_, itemSignals_, *userFactors_, userIndex_);
    LOG(INFO) << "epoch " << epoch << ": train loss = " << loss;
    // evaluate
    evaluate(epoch);
  }
}

void WALSEngine::evaluate(const size_t epoch) {
  // evaluate test average metrics
  if (metricsEngine_ && !metricsEngine_->testAvgMetrics().empty() &&
      !testUsers_.empty() &&
      (metricsEngine_->config().alwaysCompute || epoch == config_.nepochs)) {
    computeTestScores(
      testScores_, testUsers_, *userFactors_, *itemFactors_, parallel_);
    metricsEngine_->computeAndRecordTestAvgMetrics(
      epoch, testLabels_, testScores_, parallel_);
  }
}

void WALSEngine::saveUserFactors(const std::string& fileName) const {
  CHECK(userFactors_) << "user factors wasn't initialized";
  saveFactors(*userFactors_, userIndex_, fileName);
}

void WALSEngine::saveItemFactors(const std::string& fileName) const {
  CHECK(itemFactors_) << "item factors wasn't initialized";
  saveFactors(*itemFactors_, itemIndex_, fileName);
}

size_t WALSEngine::nusers() const {
  return userIndex_.size();
}

size_t WALSEngine::nitems() const {
  return itemIndex_.size();
}

void WALSEngine::groupSignals(std::vector<SignalGroup>& signals,
                              IdIndex& index,
                              std::vector<DatasetElem>& dataset) {
  sortDataset(dataset);
  const int64_t InvalidId = std::numeric_limits<int64_t>::min();
  int64_t prevId = InvalidId;
  std::vector<Signal> group;
  for (const auto& elem : dataset) {
    if (elem.userId != prevId) {
      if (prevId != InvalidId) {
        signals.emplace_back(SignalGroup{prevId, group});
      }
      prevId = elem.userId;
      group.clear();
    }
    group.emplace_back(Signal{elem.itemId, elem.value});
  }
  if (prevId != InvalidId) {
    signals.emplace_back(SignalGroup{prevId, group});
  }
  for (size_t i = 0; i < signals.size(); ++i) {
    const size_t idx = index.getOrSetIdx(signals[i].sourceId);
    CHECK_EQ(idx, i);
  }
}

void WALSEngine::sortDataset(std::vector<DatasetElem>& dataset) {
  std::sort(dataset.begin(), dataset.end(), [](const auto& x, const auto& y) {
    if (x.userId != y.userId) {
      return x.userId < y.userId;
    }
    return x.itemId < y.itemId;
  });
}

Double WALSEngine::iterate(FactorData& leftData,
                           const IdIndex& leftIndex,
                           const std::vector<SignalGroup>& leftSignals,
                           const FactorData& rightData,
                           const IdIndex& rightIndex) {
  auto genZero = [](auto...) { return 0.0; };
  leftData.setFactors(genZero);

  Matrix& X = leftData.getFactors();
  const Matrix& Y = rightData.getFactors();
  Matrix YtY = computeXtX(Y);

  auto map = [
    &X,
    &leftIndex,
    &Y,
    &rightIndex,
    &leftSignals,
    YtY,
    alpha = config_.confidenceWeight,
    lambda = config_.regularizationLambda
  ](const size_t taskId) {
    return updateFactorsForOne(
      X, leftIndex, Y, rightIndex, leftSignals[taskId], YtY, alpha, lambda);
  };

  auto reduce = [](Double sum, Double x) { return sum + x; };

  Double loss = parallel_.mapReduce(leftSignals.size(), map, reduce, 0.0);
  return loss / nusers() / nitems();
}

Matrix WALSEngine::computeXtX(const Matrix& X) {
  const size_t nrows = X.nrows();
  const size_t ntasks = parallel_.nthreads();
  const size_t taskSize = (nrows + ntasks - 1) / ntasks;

  auto map = [&X, taskSize](const size_t taskId) {
    const size_t ncols = X.ncols();
    Matrix XtX(ncols, ncols);
    const size_t l = taskId * taskSize;
    const size_t r = std::min(X.nrows(), (taskId + 1) * taskSize);
    for (size_t k = l; k < r; ++k) {
      for (size_t i = 0; i < ncols; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
          XtX(i, j) += X(k, i) * X(k, j);
        }
      }
    }
    return XtX;
  };

  auto reduce = [](const Matrix& S, const Matrix& X) {
    return S + X;
  };

  Matrix O(X.ncols(), X.ncols());
  return parallel_.mapReduce(ntasks, map, reduce, O);
}

Double WALSEngine::updateFactorsForOne(Matrix& X,
                                       const IdIndex& leftIndex,
                                       const Matrix& Y,
                                       const IdIndex& rightIndex,
                                       const SignalGroup& signalGroup,
                                       Matrix A,
                                       const Double alpha,
                                       const Double lambda) {
  Double loss = 0.0;
  const size_t n = X.ncols();
  Vector b(n);
  for (const auto& signal : signalGroup.group) {
    const size_t rightIdx = rightIndex.idx(signal.id);
    for (size_t i = 0; i < n; ++i) {
      b(i) += Y(rightIdx, i) * (1.0 + alpha * signal.value);
      for (size_t j = 0; j < n; ++j) {
        A(i, j) += Y(rightIdx, i) * alpha * signal.value * Y(rightIdx, j);
      }
    }
    // for term p^t * C * p
    loss += 1.0 + alpha * signal.value;
  }
  // B = Y^t * C * Y
  Matrix B = A;
  for (size_t i = 0; i < n; ++i) {
    A(i, i) += lambda;
  }
  // A * x = b
  Vector x = linearSymmetricSolve(A, b);
  // x^t * Y^t * C * Y * x
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      loss += B(i, j) * x(i) * x(j);
    }
  }
  // -2 * x^t * Y^t * C * p
  for (size_t i = 0; i < n; ++i) {
    loss -= 2 * x(i) * b(i);
  }
  const size_t leftIdx = leftIndex.idx(signalGroup.sourceId);
  for (size_t i = 0; i < n; ++i) {
    X(leftIdx, i) = x(i);
  }
  return loss;
}
}
