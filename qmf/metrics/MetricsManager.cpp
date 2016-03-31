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

#include <qmf/metrics/MetricsManager.h>

#include <string>

namespace {
const std::unique_ptr<qmf::Metric> kNullMetricPtr = nullptr;
}

namespace qmf {

MetricsManager MetricsManager::instance_;

MetricsManager::MetricsManager() {
  init();
}

void MetricsManager::init() {
  registerMetric<MeanSquaredError>("mse");
  registerMetric<AUC>("auc");
  registerMetric<AveragePrecision>("ap");
}

namespace detail {

bool parseAtKMetric(const std::string& name,
                    std::string& metricName,
                    size_t& k) {
  const auto pos = name.find('@');
  if (pos == 0 || pos == std::string::npos) {
    return false;
  }
  metricName = name.substr(0, pos);
  try {
    k = std::stoul(name.substr(pos + 1));
  } catch (std::exception& e) {
    return false;
  }
  return true;
}
}

/*
 * lazily construct for @k ranking metrics (e.g. "p@k")
 */
bool MetricsManager::initFromName(const std::string& name) const {
  std::string str;
  size_t k;
  if (!detail::parseAtKMetric(name, str, k)) {
    return false;
  }
  if (str == "p") {
    registerMetric<Precision>(name, k);
  } else if (str == "r") {
    registerMetric<Recall>(name, k);
  } else {
    return false;
  }
  return true;
}

const std::unique_ptr<Metric>&
  MetricsManager::getMetric(const std::string& name) const {
  if (exists(name)) {
    return metrics_[name];
  }
  return kNullMetricPtr;
}

bool MetricsManager::exists(const std::string& name) const {
  auto it = metrics_.find(name);
  if (it != metrics_.end()) {
    return true;
  }
  return initFromName(name);
}

const MetricsManager& MetricsManager::get() {
  return instance_;
}
}
