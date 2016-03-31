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
#include <string>
#include <unordered_map>

#include <qmf/metrics/Metrics.h>

namespace qmf {

namespace detail {

/**
 * for parsing metrics like "p@5"
 */
bool parseAtKMetric(const std::string& name,
                    std::string& metricName,
                    size_t& k);
}

/**
 * Singleton class for managing metrics and accessing them by name.
 */
class MetricsManager {
 public:
  MetricsManager();

  MetricsManager(const MetricsManager&) = delete;
  MetricsManager(MetricsManager&&) = delete;
  MetricsManager& operator=(const MetricsManager&) = delete;
  MetricsManager& operator=(MetricsManager&&) = delete;

  void init();

  bool initFromName(const std::string& name) const;

  template <typename MetricT, typename... Args>
  void registerMetric(const std::string& name, Args&&... args) const {
    metrics_.emplace(
      name, std::make_unique<MetricT>(std::forward<Args>(args)...));
  }

  const std::unique_ptr<Metric>& getMetric(const std::string& name) const;

  bool exists(const std::string& name) const;

  static const MetricsManager& get();

 private:
  mutable std::unordered_map<std::string, std::unique_ptr<Metric>> metrics_;

  static MetricsManager instance_;
};
}
