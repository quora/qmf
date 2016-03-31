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

#include <qmf/utils/ThreadPool.h>

#include <gtest/gtest.h>

TEST(ThreadPool, map) {
  const size_t nthreads = 4;
  qmf::ThreadPool pool(nthreads);
  EXPECT_EQ(pool.nthreads(), nthreads);
  std::vector<std::future<int>> futures;
  for (int i = 0; i < 10; ++i) {
    futures.emplace_back(pool.addTask([i](){ return i;}));
  }
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(futures[i].get(), i);
  }
}
