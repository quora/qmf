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

#include <algorithm>
#include <cmath>

namespace qmf {

BPREngine::BPREngine(const BPRConfig& config,
                     const std::unique_ptr<MetricsEngine>& metricsEngine,
                     const size_t evalNumNeg,
                     const int32_t evalSeed,
                     const size_t nthreads)
  : config_(config),
    metricsEngine_(metricsEngine),
    evalNumNeg_(evalNumNeg),
    evalSeed_(evalSeed),
    parallel_(nthreads),
    gen_(std::random_device()()) {
  if (config_.numHogwildThreads > nthreads) {
    LOG(WARNING)
      << "number of hogwild threads should be smaller than number of "
         "threads in the threadpool";
  }
  if (metricsEngine_ && !metricsEngine_->testAvgMetrics().empty() &&
      metricsEngine_->config().numTestUsers == 0) {
    LOG(WARNING) << "computing average test metrics on all users can be slow! "
                    "Set numTestUsers > 0 to sample some of them";
  }
}

size_t BPREngine::nusers() const {
  return userIndex_.size();
}

size_t BPREngine::nitems() const {
  return itemIndex_.size();
}

void BPREngine::saveUserFactors(const std::string& fileName) const {
  CHECK(userFactors_) << "user factors wasn't initialized";
  saveFactors(*userFactors_, userIndex_, fileName);
}

void BPREngine::saveItemFactors(const std::string& fileName) const {
  CHECK(itemFactors_) << "item factors wasn't initialized";
  saveFactors(*itemFactors_, itemIndex_, fileName);
}

void BPREngine::init(const std::vector<DatasetElem>& dataset) {
  CHECK(!userFactors_ && !itemFactors_)
    << "engine was already initialized with train data";
  // populate data
  for (const auto& elem : dataset) {
    if (elem.value < 1.0) {
      continue;
    }
    const size_t uidx = userIndex_.getOrSetIdx(elem.userId);
    const size_t pidx = itemIndex_.getOrSetIdx(elem.itemId);

    data_.push_back(PosPair{uidx, pidx});
  }

  itemMap_.resize(userIndex_.size());
  for (const auto& p : data_) {
    itemMap_[p.userIdx].insert(p.posItemIdx);
  }

  // generate evaluation set
  iterate([& evalSet = evalSet_](PosNegTriplet && triplet) {
    evalSet.push_back(std::move(triplet));
  }, evalNumNeg_, std::mt19937(evalSeed_));

  // initialize model
  learningRate_ = config_.initLearningRate;
  userFactors_ = std::make_unique<FactorData>(nusers(), config_.nfactors);
  itemFactors_ =
    std::make_unique<FactorData>(nitems(), config_.nfactors, config_.useBiases);

  std::uniform_real_distribution<Double> distr(
    -config_.initDistributionBound, config_.initDistributionBound);
  auto genUnif = [&distr, & gen = gen_](auto...) {
    return distr(gen);
  };
  userFactors_->setFactors(genUnif);
  itemFactors_->setFactors(genUnif);
  if (config_.useBiases) {
    itemFactors_->setBiases(genUnif);
  }
}

void BPREngine::initTest(const std::vector<DatasetElem>& testDataset) {
  CHECK(testEvalSet_.empty())
    << "engine was already initialzied with test data";
  // populate item map
  std::vector<std::pair<size_t, size_t>> validElems;
  validElems.reserve(testDataset.size());
  testItemMap_.resize(userIndex_.size());
  for (const auto& elem : testDataset) {
    if (elem.value < 1.0) {
      continue;
    }
    const size_t uidx = userIndex_.idx(elem.userId);
    const size_t pidx = itemIndex_.idx(elem.itemId);
    if (uidx == IdIndex::missingIdx || pidx == IdIndex::missingIdx) {
      continue;
    }
    testItemMap_[uidx].insert(pidx);
    validElems.emplace_back(uidx, pidx);
  }
  // generate evaluation set
  std::mt19937 gen(evalSeed_);
  testEvalSet_.reserve(evalNumNeg_ * validElems.size());
  for (const auto& p : validElems) {
    for (size_t i = 0; i < evalNumNeg_; ++i) {
      testEvalSet_.push_back(PosNegTriplet{
        p.first,
        p.second,
        sampleRandomNegative(p.first, gen, /*useTestItemMap=*/true)});
    }
  }

  // initialize data for test average metrics
  if (metricsEngine_ && !metricsEngine_->testAvgMetrics().empty()) {
    initAvgTestData(
      testUsers_, testLabels_, testScores_, testDataset, userIndex_, itemIndex_,
      metricsEngine_->config().numTestUsers, metricsEngine_->config().seed);
  }
}

void BPREngine::optimize() {
  CHECK(userFactors_ && itemFactors_)
    << "no factor data, have you initialized the engine?";

  for (size_t epoch = 1; epoch <= config_.nepochs; ++epoch) {
    // run SGD
    auto updateOne = [this](const auto& triplet) { update(triplet); };
    if (config_.numHogwildThreads <= 1) {
      iterate(updateOne, config_.numNegativeSamples, gen_);
    } else {
      const size_t numTasks = config_.numHogwildThreads;
      const size_t blockSize = data_.size() / numTasks;
      auto func = [this, numTasks, blockSize, updateOne](const size_t taskId) {
        iterateBlock(updateOne, taskId * blockSize,
                     std::min(data_.size(), (taskId + 1) * blockSize),
                     config_.numNegativeSamples, gen_);
      };
      parallel_.execute(numTasks, func);
    }

    evaluate(epoch);

    // update learning rate/dataset
    if (config_.decayRate < 1.0) {
      learningRate_ *= config_.decayRate;
    }
    if (config_.shuffleTrainingSet) {
      shuffle();
    }
  }
}

void BPREngine::update(const PosNegTriplet& triplet) {
  const size_t uidx = triplet.userIdx;
  const size_t pidx = triplet.posItemIdx;
  const size_t nidx = triplet.negItemIdx;

  const Double e = lossDerivative(predictDifference(uidx, pidx, nidx));
  CHECK(std::isfinite(e)) << "gradients too big, try decreasing the learning "
                             "rate (--init_learning_rate)";
  const Double lr = learningRate_;

  // update biases
  if (config_.useBiases) {
    // b_i <- b_i + lr * (e - b_lambda * b_i)
    Double step = lr * (e - config_.biasLambda * itemFactors_->biasAt(pidx));
    itemFactors_->biasAt(pidx) += step;
    // b_j <- b_j + b_lr * (-e - b_lambda * b_j)
    step = lr * (-e - config_.biasLambda * itemFactors_->biasAt(nidx));
    itemFactors_->biasAt(nidx) += step;
  }

  // update user factors
  // p_u <- p_u + lr * (e * (q_i - q_j) - f_lambda * p_u)
  for (size_t i = 0; i < config_.nfactors; ++i) {
    const Double step =
      lr * (e * (itemFactors_->at(pidx, i) - itemFactors_->at(nidx, i)) -
            config_.userLambda * userFactors_->at(uidx, i));
    userFactors_->at(uidx, i) += step;
  }
  // update pos item factors
  // q_i <- q_i + lr * (e * p_u - f_lambda * q_i)
  for (size_t i = 0; i < config_.nfactors; ++i) {
    const Double step = lr * (e * userFactors_->at(uidx, i) -
                              config_.itemLambda * itemFactors_->at(pidx, i));
    itemFactors_->at(pidx, i) += step;
  }
  // update neg item factors
  // q_j <- q_j + lr * (-e * p_u - f_lambda * q_j)
  for (size_t i = 0; i < config_.nfactors; ++i) {
    const Double step = lr * (-e * userFactors_->at(uidx, i) -
                              config_.itemLambda * itemFactors_->at(nidx, i));
    itemFactors_->at(nidx, i) += step;
  }
}

Double BPREngine::predictDifference(const size_t userIdx,
                                    const size_t posItemIdx,
                                    const size_t negItemIdx) const {
  // score difference: b_i - b_j + p_u'(q_i - q_j)
  Double pred = 0.0;
  if (config_.useBiases) {
    pred += itemFactors_->biasAt(posItemIdx) - itemFactors_->biasAt(negItemIdx);
  }
  for (size_t i = 0; i < config_.nfactors; ++i) {
    pred += userFactors_->at(userIdx, i) *
            (itemFactors_->at(posItemIdx, i) - itemFactors_->at(negItemIdx, i));
  }
  return pred;
}

Double BPREngine::loss(const Double scoreDifference) const {
  return log(1.0 + exp(-scoreDifference));
}

Double BPREngine::lossDerivative(const Double scoreDifference) const {
  // e = d/dx log sigmoid(x) = 1 / (1 + exp(x))
  return 1.0 / (1.0 + exp(scoreDifference));
}

void BPREngine::evaluate(const size_t epoch) {
  // evaluate on train/test evaluation sets
  auto evalLoss = [this](const PosNegTriplet& triplet) {
    return loss(predictDifference(
      triplet.userIdx, triplet.posItemIdx, triplet.negItemIdx));
  };

  const Double trainEvalLoss =
    evalSet_.empty() ? -1.0 : parallel_.mapReduce(
                                evalSet_, evalLoss, std::plus<Double>(), 0.0) /
                                evalSet_.size();
  const Double testEvalLoss =
    testEvalSet_.empty() ?
      -1.0 :
      parallel_.mapReduce(testEvalSet_, evalLoss, std::plus<Double>(), 0.0) /
        testEvalSet_.size();
  LOG(INFO) << "epoch " << epoch << ": train loss = " << trainEvalLoss
            << ", test loss = " << testEvalLoss;

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

void BPREngine::shuffle() {
  std::shuffle(data_.begin(), data_.end(), gen_);
}
}
