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

#include <qmf/metrics/MetricsEngine.h>

namespace qmf {

MetricsEngine::MetricsEngine(const MetricsConfig& config, const bool log)
  : config_(config), log_(log) {
}

bool MetricsEngine::addMetric(
  std::vector<std::string>& metrics,
  const std::string& metric) {
  if (MetricsManager::get().exists(metric)) {
    metrics.emplace_back(metric);
    return true;
  } else {
    return false;
  }
}

void MetricsEngine::recordMetric(const std::string& metricKey,
                                 const size_t epoch,
                                 const Double val) {
  metricsMap_[metricKey].emplace_back(epoch, val);
  if (log_) {
    LOG(INFO) << "epoch " << epoch << ": recorded metric " << metricKey << " = "
              << val;
  }
}
}
