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

#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <qmf/metrics/MetricsManager.h>
#include <qmf/Types.h>

namespace qmf {

struct MetricsConfig {
  size_t numTestUsers;
  bool alwaysCompute;
  int32_t seed;
};

/**
 * MetricsEngine class, used to specify the desired metrics and
 * compute/record them when desired. Typically passed to training algorithms.
 */
class MetricsEngine {
 public:
  MetricsEngine(const MetricsConfig& config = {}, const bool log = true);

  const MetricsConfig& config() const {
    return config_;
  }

  bool addTrainMetric(const std::string& metric) {
    return addMetric(trainMetrics_, metric);
  }

  bool addTestMetric(const std::string& metric) {
    return addMetric(testMetrics_, metric);
  }

  bool addTrainAvgMetric(const std::string& metric) {
    return addMetric(trainAvgMetrics_, metric);
  }

  bool addTestAvgMetric(const std::string& metric) {
    return addMetric(testAvgMetrics_, metric);
  }

  void computeAndRecordTrainMetrics(const size_t epoch,
                                    const std::vector<Double>& labels,
                                    const std::vector<Double>& scores) {
    computeAndRecordMetrics(trainMetrics_, "train_", epoch, labels, scores);
  }

  void computeAndRecordTestMetrics(const size_t epoch,
                                   const std::vector<Double>& labels,
                                   const std::vector<Double>& scores) {
    computeAndRecordMetrics(testMetrics_, "test_", epoch, labels, scores);
  }

  template <typename... ComputeArgs>
  void computeAndRecordTrainAvgMetrics(
    const size_t epoch,
    ComputeArgs&... args) {
    computeAndRecordMetrics(trainAvgMetrics_, "train_avg_", epoch, args...);
  }

  template <typename... ComputeArgs>
  void computeAndRecordTestAvgMetrics(
    const size_t epoch,
    ComputeArgs&... args) {
    computeAndRecordMetrics(testAvgMetrics_, "test_avg_", epoch, args...);
  }

  const std::vector<std::string>& trainMetrics() const {
    return trainMetrics_;
  }

  const std::vector<std::string>& testMetrics() const {
    return testMetrics_;
  }

  const std::vector<std::string>& trainAvgMetrics() const {
    return trainAvgMetrics_;
  }

  const std::vector<std::string>& testAvgMetrics() const {
    return testAvgMetrics_;
  }

  using MetricVector = std::vector<std::pair<size_t, Double>>;

 private:
  bool addMetric(std::vector<std::string>& metrics,
                 const std::string& metric);

  template <typename... ComputeArgs>
  void computeAndRecordMetrics(std::vector<std::string>& metrics,
                               const std::string& prefix,
                               const size_t epoch,
                               ComputeArgs&... args) {
    for (const auto& metric : metrics) {
      const auto& m = MetricsManager::get().getMetric(metric);
      CHECK(m) << "missing metric " << prefix + metric;
      const Double val = m->compute(args...);
      recordMetric(prefix + metric, epoch, val);
    }
  }

  void recordMetric(const std::string& metricKey,
                    const size_t epoch,
                    const Double val);

  const MetricsConfig& config_;
  const bool log_; // whether to log recorded metrics to stderr

  std::vector<std::string> trainMetrics_;
  std::vector<std::string> trainAvgMetrics_;
  std::vector<std::string> testMetrics_;
  std::vector<std::string> testAvgMetrics_;
  std::unordered_map<std::string, MetricVector> metricsMap_;
};
}
