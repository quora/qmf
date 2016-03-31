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

#include <vector>

#include <qmf/metrics/Metrics.h>

#include <gtest/gtest.h>

qmf::Double compute(const qmf::Metric& metric,
                    const std::vector<qmf::Double>& labels,
                    const std::vector<qmf::Double>& scores) {
  return metric.compute(labels, scores);
}

qmf::Double computeAverage(const qmf::Metric& metric,
                           const std::vector<std::vector<qmf::Double>>& labels,
                           const std::vector<std::vector<qmf::Double>>& scores) {
  return metric.compute(labels, scores);
}

TEST(TestMetrics, MeanSquaredError) {
  qmf::MeanSquaredError m;
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0}, {0.5, 0.5}), 0.25);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0, 1.0}, {0.0, 1.0, 2.0}), 1.0);

  // average metric
  EXPECT_DOUBLE_EQ(computeAverage(m, {{1.0, 0.0}, {1.0, 0.0, 1.0}},
                                  {{0.5, 0.5}, {0.0, 1.0, 2.0}}),
                   0.5 * (0.25 + 1.0));
}

TEST(TestMetrics, AUC) {
  qmf::AUC m;
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0}, {3.0, 2.0}), 0.0);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 1.0, 0.0}, {3.0, 2.0, 0.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0, 1.0}, {3.0, 2.0, 0.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0, 1.0}, {3.0, 2.0, 0.0}), 0.0);
}

TEST(TestMetrics, Precision) {
  qmf::Precision m(/*k=*/1);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 1.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0}, {3.0, 2.0}), 0.0);
  qmf::Precision m2(/*k=*/2);
  EXPECT_DOUBLE_EQ(compute(m2, {1.0, 0.0}, {3.0, 2.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m2, {1.0, 1.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m2, {0.0, 1.0}, {3.0, 2.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m2, {0.0, 1.0, 0.0}, {3.0, 2.0, 1.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m2, {0.0, 1.0, 0.0}, {3.0, 1.0, 2.0}), 0.0);
}

TEST(TestMetrics, Recall) {
  qmf::Recall m(/*k=*/1);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 1.0}, {3.0, 2.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0}, {3.0, 2.0}), 0.0);
  qmf::Recall m2(/*k=*/2);
  EXPECT_DOUBLE_EQ(compute(m2, {1.0, 0.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m2, {1.0, 1.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m2, {0.0, 1.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m2, {0.0, 1.0, 0.0}, {3.0, 2.0, 1.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m2, {0.0, 1.0, 0.0}, {3.0, 1.0, 2.0}), 0.0);
}

TEST(TestMetrics, AveragePrecision) {
  qmf::AveragePrecision m;
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 0.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {1.0, 1.0}, {3.0, 2.0}), 1.0);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0}, {3.0, 2.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0, 0.0}, {3.0, 2.0, 1.0}), 0.5);
  EXPECT_DOUBLE_EQ(compute(m, {0.0, 1.0, 0.0}, {3.0, 1.0, 2.0}), 1.0 / 3);
}
