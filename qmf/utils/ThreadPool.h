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

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace qmf {

#define WITH_LOCK(mu)           \
  if (bool __state__ = false) { \
  } else                        \
    for (std::lock_guard<std::mutex> lock(mu); !__state__; __state__ = true)

// generic pool of tasks, where a task can be an arbitrary function
class ThreadPool {
 public:
  using Task = std::function<void()>;

  // nthreads is the number of threads in the pool to be used during the execution
  explicit ThreadPool(const size_t nthreads);

  ~ThreadPool();

  // not copyable, not movable
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  size_t nthreads() const {
    return threads_.size();
  }

  template <typename FuncT, typename... Args>
  auto addTask(FuncT&& func, Args&&... args)
    -> std::future<typename std::result_of<FuncT(Args...)>::type>;

 private:
  void threadRun();

  std::queue<Task> tasks_;

  std::vector<std::thread> threads_;

  std::mutex mutex_;

  std::atomic_bool poison_;

  std::condition_variable cond_;
};
}

#include <qmf/utils/ThreadPool-inl.h>
