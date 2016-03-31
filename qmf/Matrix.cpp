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

#include <qmf/Matrix.h>

#include <glog/logging.h>

namespace qmf {

namespace detail {

extern "C" void dsysv_(char* uplo,
                       int* n,
                       int* nrhs,
                       double* a,
                       int* lda,
                       int* ipiv,
                       double* b,
                       int* ldb,
                       double* work,
                       int* lwork,
                       int* info);
}


Matrix::Matrix(const size_t nrows, const size_t ncols)
  : nrows_(nrows),
    ncols_(ncols),
    data_(nrows * ncols, 0.0) {
  CHECK_GT(nrows * ncols, 0) << "matrix's dimensions should be positive";
}

Matrix::Matrix(Matrix&& X) {
  nrows_ = X.nrows_;
  ncols_ = X.ncols_;
  data_ = std::move(X.data_);
}

Matrix& Matrix::operator=(Matrix&& X) {
  nrows_ = X.nrows_;
  ncols_ = X.ncols_;
  data_ = std::move(X.data_);
  return *this;
}

Matrix Matrix::transpose() const {
  Matrix T(ncols_, nrows_);
  for (size_t i = 0; i < nrows_; ++i) {
    for (size_t j = 0; j < ncols_; ++j) {
      T(j, i) = operator()(i, j);
    }
  }
  return T;
}

Matrix Matrix::operator+(const Matrix& X) const {
  CHECK_EQ(nrows_, X.nrows());
  CHECK_EQ(ncols_, X.ncols());
  Matrix S(nrows_, ncols_);
  for (size_t i = 0; i < nrows_; ++i) {
    for (size_t j = 0; j < ncols_; ++j) {
      S(i, j) = operator()(i, j) + X(i, j);
    }
  }
  return S;
}

Vector linearSymmetricSolve(Matrix A, Vector b) {
  CHECK_EQ(A.nrows(), A.ncols()) << "A should be squared";
  CHECK_EQ(A.nrows(), b.size()) << "b should have the same number of rows as A";
  int n = static_cast<int>(A.nrows());
  int bncols = 1;
  // dgesv_ assumes a columnar order
  A = A.transpose();
  std::vector<int> pivot(n);
  std::vector<Double> work(n);
  int result = 0;
  const char* uplo = "Upper";
  detail::dsysv_(const_cast<char*>(uplo), &n, &bncols, A.data(), &n, &pivot[0],
                 b.data(), &n, &work[0], &n, &result);
  CHECK_EQ(result, 0) << "dgesv failed, code " << result;
  return b;
}
}
