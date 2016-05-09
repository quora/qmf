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

#include <random>

#include <qmf/Matrix.h>

#include <gtest/gtest.h>

TEST(Matrix, basic) {
  const size_t n = 3;
  const size_t m = 4;
  qmf::Matrix x(n, m);
  EXPECT_EQ(x.nrows(), n);
  EXPECT_EQ(x.ncols(), m);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      x(i, j) = i * j;
    }
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      EXPECT_EQ(x(i, j), i * j);
    }
  }

  EXPECT_DEATH(qmf::Matrix(0, 0), ".*");
}

TEST(Matrix, operatorPlus) {
  const size_t n = 3;
  qmf::Matrix x(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      x(i, j) = i * j;
    }
  }
  qmf::Matrix s = x + x;
  EXPECT_EQ(s.nrows(), n);
  EXPECT_EQ(s.ncols(), n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(s(i, j), 2 * i * j);
    }
  }
}

TEST(Matrix, operatorPlusIncorrect) {
  const size_t n1 = 3;
  qmf::Matrix x1(n1, n1);

  const size_t n2 = 4;
  qmf::Matrix x2(n2, n2);

  EXPECT_DEATH((x1 + x2), ".*");
}

TEST(Matrix, transpose) {
  const size_t nrows = 4;
  const size_t ncols = 5;

  qmf::Matrix X(nrows, ncols);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      X(i, j) = rand();
    }
  }

  qmf::Matrix XTT = X.transpose().transpose();
  EXPECT_EQ(XTT.nrows(), X.nrows());
  EXPECT_EQ(XTT.ncols(), X.ncols());
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      EXPECT_EQ(X(i, j), XTT(i, j));
    }
  }
}

TEST(Matrix, linearSolve) {
  const size_t n = 50;
  std::mt19937 gen(123);
  std::uniform_real_distribution<qmf::Double> distr(-1.0, 1.0);
  qmf::Matrix A(n, n);
  qmf::Vector b(n);

  for (size_t i = 0; i < n; ++i) {
    b(i) = distr(gen);
    for (size_t j = i; j < n; ++j) {
      qmf::Double x = distr(gen);
      A(i, j) = A(j, i) = x;
    }
  }

  qmf::Vector x = qmf::linearSymmetricSolve(A, b);
  EXPECT_EQ(x.size(), n);
  for (size_t i = 0; i < n; ++i) {
    qmf::Double prod = 0.0;
    for (size_t j = 0; j < n; ++j) {
      prod += A(i, j) * x(j);
    }
    EXPECT_NEAR(b(i), prod, 1e-8);
  }
}
