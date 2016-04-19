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

namespace qmf {

ThreadPool::ThreadPool(const size_t nthreads) : poison_(false) {
  CHECK_GT(nthreads, 0) << "the number of threads should be positive";
  for (size_t i = 0; i < nthreads; ++i) {
    threads_.emplace_back(std::bind(&ThreadPool::threadRun, this));
  }
}

ThreadPool::~ThreadPool() {
  WITH_LOCK(mutex_) {
    poison_ = true;
  }
  cond_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ThreadPool::threadRun() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_.wait(lock, [this]{ return this->poison_ || !this->tasks_.empty(); });
      if (poison_ && tasks_.empty()) {
        break;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}
}
