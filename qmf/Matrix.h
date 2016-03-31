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
#include <qmf/Vector.h>

namespace qmf {

// class for a row-wise matrix
class Matrix {
 public:
  Matrix(const size_t nrows, const size_t ncols);

  // default copy
  Matrix(const Matrix& X) = default;
  Matrix& operator=(const Matrix& X) = default;

  // move semantics
  Matrix(Matrix&& X);
  Matrix& operator=(Matrix&& X);

  Double operator()(const size_t r, const size_t c) const {
    return data_[index(r, c)];
  }

  Double& operator()(const size_t r, const size_t c) {
    return data_[index(r, c)];
  }

  size_t nrows() const {
    return nrows_;
  }

  size_t ncols() const {
    return ncols_;
  }

  // computes matrix transpose, X^T
  Matrix transpose() const;

  Matrix operator+(const Matrix& X) const;

  // returns a raw pointer to the data
  Double* const data() {
    return &data_[0];
  }

 private:
  size_t index(const size_t r, const size_t c) const {
    return r * ncols_ + c;
  }

  size_t nrows_;

  size_t ncols_;

  std::vector<Double> data_;
};

// solves a system of linear equations, A * x = b.
// matrix A should symmetric and vector b should have the same number of rows as A.
Vector linearSymmetricSolve(Matrix A, Vector b);

}
