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

#include <qmf/Matrix.h>
#include <qmf/Vector.h>

#include <glog/logging.h>

namespace qmf {

class FactorData {
 public:
  FactorData(const size_t nelems,
             const size_t nfactors,
             const bool withBiases = false)
    : withBiases_(withBiases),
      factors_(nelems, nfactors),
      biases_(withBiases ? nelems : 0) {
  }

  Double at(const size_t idx, const size_t fidx) const {
    return factors_(idx, fidx);
  }

  Double& at(const size_t idx, const size_t fidx) {
    return factors_(idx, fidx);
  }

  Double biasAt(const size_t idx) const {
    return withBiases_ ? biases_(idx) : 0.0;
  }

  Double& biasAt(const size_t idx) {
    CHECK(withBiases_) << "can't access bias when withBiases = false";
    return biases_(idx);
  }

  template <typename FuncT>
  void setFactors(FuncT func) {
    for (size_t idx = 0; idx < nelems(); ++idx) {
      for (size_t fidx = 0; fidx < nfactors(); ++fidx) {
        factors_(idx, fidx) = func(idx, fidx);
      }
    }
  }

  template <typename FuncT>
  void setBiases(FuncT func) {
    for (size_t idx = 0; idx < biases_.size(); ++idx) {
      biases_(idx) = func(idx);
    }
  }

  size_t nelems() const {
    return factors_.nrows();
  }

  size_t nfactors() const {
    return factors_.ncols();
  }

  bool withBiases() const {
    return withBiases_;
  }

  const Matrix& getFactors() const {
    return factors_;
  }

  Matrix& getFactors() {
    return factors_;
  }

  const Vector& getBiases() const {
    return biases_;
  }

  Vector& getBiases() {
    return biases_;
  }

 private:
  const bool withBiases_;

  Matrix factors_;
  Vector biases_;
};
}
