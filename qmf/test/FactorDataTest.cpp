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

#include <qmf/FactorData.h>

#include <gtest/gtest.h>

TEST(FactorData, withBiases) {
  qmf::FactorData fd(3, 2, /*withBiases=*/true);

  EXPECT_EQ(fd.nelems(), 3);
  EXPECT_EQ(fd.nfactors(), 2);
  EXPECT_EQ(fd.withBiases(), true);

  fd.at(0, 0) = 1.5;
  fd.at(2, 1) = 3.0;
  fd.biasAt(2) = -0.5;
  EXPECT_DOUBLE_EQ(fd.at(0, 0), 1.5);
  EXPECT_DOUBLE_EQ(fd.at(2, 1), 3.0);
  EXPECT_DOUBLE_EQ(fd.biasAt(2), -0.5);

  fd.setFactors(
    [](const size_t idx, const size_t fidx) { return 2 * idx + fidx; });
  for (size_t i = 0; i < 3; ++i) {
    for (size_t f = 0; f < 2; ++f) {
      EXPECT_EQ(fd.at(i, f), 2 * i + f);
    }
  }

  fd.setBiases([](const size_t idx) { return 42 + idx; });
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(fd.biasAt(i), 42 + i);
  }
}

TEST(FactorData, noBiases) {
  qmf::FactorData fd(3, 2, /*withBiases=*/false);

  EXPECT_EQ(fd.nelems(), 3);
  EXPECT_EQ(fd.nfactors(), 2);
  EXPECT_EQ(fd.withBiases(), false);

  fd.at(0, 0) = 1.5;
  fd.at(2, 1) = 3.0;
  EXPECT_DOUBLE_EQ(fd.at(0, 0), 1.5);
  EXPECT_DOUBLE_EQ(fd.at(2, 1), 3.0);

  EXPECT_DEATH(
    fd.biasAt(0) = 1.0, ".*withBiases = false");
}
