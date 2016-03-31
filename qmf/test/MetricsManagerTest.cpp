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

#include <qmf/metrics/MetricsManager.h>

#include <gtest/gtest.h>

TEST(TestMetricsManager, parseAtKMetric) {
  std::string name;
  size_t k;
  EXPECT_TRUE(qmf::detail::parseAtKMetric("p@5", name, k));
  EXPECT_EQ(name, "p");
  EXPECT_EQ(k, 5);

  EXPECT_FALSE(qmf::detail::parseAtKMetric("p5", name, k));
  EXPECT_FALSE(qmf::detail::parseAtKMetric("@5", name, k));
  EXPECT_FALSE(qmf::detail::parseAtKMetric("p@", name, k));
}

TEST(TestMetricsManager, exists) {
  qmf::MetricsManager m;
  m.init();
  EXPECT_TRUE(m.exists("mse"));
  EXPECT_TRUE(m.exists("auc"));
  EXPECT_TRUE(m.exists("ap"));
  EXPECT_TRUE(m.exists("p@5"));
  EXPECT_TRUE(m.exists("p@10"));
  EXPECT_TRUE(m.exists("r@5"));
  EXPECT_TRUE(m.exists("r@10"));
}
