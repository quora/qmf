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

#include <qmf/metrics/Metrics.h>

#include <algorithm>
#include <cmath>
#include <functional>

#include <glog/logging.h>

namespace qmf {

Double Metric::compute(const std::vector<std::vector<Double>>& labels,
                       const std::vector<std::vector<Double>>& scores) const {
  CHECK_EQ(labels.size(), scores.size());
  CHECK_GT(labels.size(), 0);
  Double sum = 0.0;
  for (size_t i = 0; i < labels.size(); ++i) {
    sum += compute(labels[i], scores[i]);
  }
  return sum / labels.size();
}

Double Metric::compute(const std::vector<std::vector<Double>>& labels,
                       const std::vector<std::vector<Double>>& scores,
                       ParallelExecutor& parallel) const {
  CHECK_EQ(labels.size(), scores.size());
  CHECK_GT(labels.size(), 0);
  const Double tot = parallel.mapReduce(
    /*numTasks=*/labels.size(),
    /*mapper=*/
    [this, &labels, &scores](const size_t taskId) {
      return this->compute(labels[taskId], scores[taskId]);
    },
    /*reducer=*/std::plus<Double>(),
    /*neutralElem=*/0.0);
  return tot / labels.size();
}

Double MeanSquaredError::compute(const std::vector<Double>& labels,
                                 const std::vector<Double>& scores) const {
  CHECK_EQ(labels.size(), scores.size());
  CHECK_GT(labels.size(), 0);
  Double sum = 0.0;
  for (size_t i = 0; i < labels.size(); ++i) {
    sum += std::pow(labels[i] - scores[i], 2);
  }
  return sum / labels.size();
}

Double AUC::compute(const std::vector<Double>& labels,
                    const std::vector<Double>& scores) const {
  CHECK_EQ(labels.size(), scores.size());
  std::vector<std::pair<Double, bool>> scored;
  scored.reserve(labels.size());
  int32_t pos = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0.0) {
      ++pos;
    }
    scored.emplace_back(scores[i], static_cast<bool>(labels[i] > 0.0));
  }
  int32_t neg = labels.size() - pos;

  // make sure we have at least 1 example in each class
  if (pos == 0 || neg == 0) {
    LOG(ERROR) << "AUC needs at least 1 example in each class";
    return 1.0;
  }

  std::sort(
    scored.begin(), scored.end(), std::greater<std::pair<Double, bool>>());
  int tp = 0;
  Double auc = 0;
  for (const auto& p : scored) {
    if (p.second) {
      ++tp;
    } else {
      // tpr * d(fpr) = (tp/pos) * (1/neg)
      auc += static_cast<Double>(tp) / pos / neg;
    }
  }

  return auc;
}

Double Precision::compute(const std::vector<Double>& labels,
                          const std::vector<Double>& scores) const {
  CHECK_EQ(labels.size(), scores.size());
  CHECK_GE(labels.size(), k_) << "P@k needs at least k ranked elements";
  std::vector<std::pair<Double, bool>> scored;
  scored.reserve(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    scored.emplace_back(scores[i], static_cast<bool>(labels[i] > 0.0));
  }
  std::nth_element(scored.begin(), scored.begin() + k_, scored.end(),
                   std::greater<std::pair<Double, bool>>());
  const auto pos = std::count_if(scored.begin(), scored.begin() + k_,
                                 [](const auto& p) { return p.second; });
  return static_cast<Double>(pos) / k_;
}

Double Recall::compute(const std::vector<Double>& labels,
                       const std::vector<Double>& scores) const {
  CHECK_EQ(labels.size(), scores.size());
  CHECK_GE(labels.size(), k_) << "R@k needs at least k ranked elements";
  std::vector<std::pair<Double, bool>> scored;
  scored.reserve(labels.size());
  int32_t totalPos = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0.0) {
      ++totalPos;
    }
    scored.emplace_back(scores[i], static_cast<bool>(labels[i] > 0.0));
  }
  CHECK_GT(totalPos, 0) << "R@k needs at least 1 positive";

  std::nth_element(scored.begin(), scored.begin() + k_, scored.end(),
                   std::greater<std::pair<Double, bool>>());
  const auto pos = std::count_if(scored.begin(), scored.begin() + k_,
                                 [](const auto& p) { return p.second; });
  return static_cast<Double>(pos) / totalPos;
}

Double AveragePrecision::compute(const std::vector<Double>& labels,
                                 const std::vector<Double>& scores) const {
  CHECK_EQ(labels.size(), scores.size());
  std::vector<std::pair<Double, bool>> scored;
  scored.reserve(labels.size());
  int32_t totalPos = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0.0) {
      ++totalPos;
    }
    scored.emplace_back(scores[i], static_cast<bool>(labels[i] > 0.0));
  }
  CHECK_GT(totalPos, 0) << "AP needs at least 1 positive";
  std::sort(
    scored.begin(), scored.end(), std::greater<std::pair<Double, bool>>());

  Double ap = 0.0;
  int32_t pos = 0;
  for (size_t i = 0; i < scored.size(); ++i) {
    if (scored[i].second) {
      ++pos;
      ap += static_cast<Double>(pos) / (i + 1);
    }
  }
  return ap / totalPos;
}
}
