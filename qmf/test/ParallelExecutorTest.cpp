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

#include <functional>
#include <vector>
#include <utility>

#include <qmf/utils/ParallelExecutor.h>

#include <gtest/gtest.h>

TEST(ParallelExecutor, execute) {
  const size_t nthreads = 4;
  const size_t ntasks = 1000;
  qmf::ParallelExecutor parallel(nthreads);

  std::vector<int> vals(ntasks);
  auto func = [&vals](const size_t taskId) { vals[taskId] = 1; };

  parallel.execute(ntasks, func);
  for (int val : vals) {
    EXPECT_EQ(val, 1);
  }
}

TEST(ParallelExecutor, mapReduce) {
  const size_t nthreads = 4;
  const size_t ntasks = 1000;
  qmf::ParallelExecutor parallel(nthreads);

  const int sum = parallel.mapReduce(ntasks, [](const size_t taskId) {
    return taskId * taskId;
  }, std::plus<int>(), 0);
  EXPECT_EQ(sum, (ntasks - 1) * ntasks * (2 * ntasks - 1) / 6);
}

TEST(ParallelExecutor, mapReduceElems) {
  const size_t nthreads = 4;
  const size_t ntasks = 1000;
  qmf::ParallelExecutor parallel(nthreads);

  std::vector<std::pair<int, int>> elems;
  for (size_t i = 0; i < ntasks; ++i) {
    elems.emplace_back(i, i);
  }
  const int sum = parallel.mapReduce(elems, [](const auto& p) {
    return p.first * p.second;
  }, std::plus<int>(), 0);
  EXPECT_EQ(sum, (ntasks - 1) * ntasks * (2 * ntasks - 1) / 6);
}
