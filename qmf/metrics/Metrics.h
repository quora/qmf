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

#include <vector>

#include <qmf/Types.h>
#include <qmf/utils/ParallelExecutor.h>

namespace qmf {

class Metric {
 public:
  virtual ~Metric() = default;

  virtual Double compute(const std::vector<Double>& labels,
                         const std::vector<Double>& scores) const = 0;

  virtual Double compute(const std::vector<std::vector<Double>>& labels,
                         const std::vector<std::vector<Double>>& scores) const;

  virtual Double compute(const std::vector<std::vector<Double>>& labels,
                         const std::vector<std::vector<Double>>& scores,
                         ParallelExecutor& parallel) const;
};

class MeanSquaredError : public Metric {
 public:
  Double compute(const std::vector<Double>& labels,
                 const std::vector<Double>& scores) const override;
};

class AUC : public Metric {
 public:
  Double compute(const std::vector<Double>& labels,
                 const std::vector<Double>& scores) const override;
};

class Precision : public Metric {
 public:
  explicit Precision(const size_t k) : k_(k) {
  }

  Double compute(const std::vector<Double>& labels,
                 const std::vector<Double>& scores) const override;

 private:
  const size_t k_;  // precision window size
};

class Recall : public Metric {
 public:
  explicit Recall(const size_t k) : k_(k) {
  }

  Double compute(const std::vector<Double>& labels,
                 const std::vector<Double>& scores) const override;

 private:
  const size_t k_;  // recall window size
};

// Average Precision
class AveragePrecision : public Metric {
 public:
  Double compute(const std::vector<Double>& labels,
                 const std::vector<Double>& scores) const override;
};
}
