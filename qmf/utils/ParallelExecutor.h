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

#include <algorithm>

#include <qmf/utils/ThreadPool.h>

namespace qmf {

// a simple interface for basic parallel execution primitives
class ParallelExecutor {
 public:
  explicit ParallelExecutor(const size_t nthreads)
    : threadPool_(std::make_unique<ThreadPool>(nthreads)) {
  }

  // executes `func` on `ntasks` tasks.
  // `func`'s signature is void(const size_t taskId)
  template <typename FuncT>
  void execute(const size_t ntasks, FuncT&& func);

  // runs `mapper` task `ntasks` times, then reduces on mappers' output.
  // `mapper`'s signature is T(const size_t taskId).
  // `reducer`'s signature is T(T, T).
  template <typename T, typename MapperT, typename ReducerT>
  T mapReduce(const size_t ntasks,
              MapperT&& mapper,
              ReducerT&& reducer,
              T neutral);

  // runs `mapper` against each element of `elems`, then reduces on the output.
  // `mapper`'s signature is T(const ElemT).
  // `reducer`'s signature is T(T, T).
  template <typename T, typename ElemT, typename MapperT, typename ReducerT>
  T mapReduce(const std::vector<ElemT>& elems,
              MapperT&& mapper,
              ReducerT&& reducer,
              T neutral);

  size_t nthreads() const {
    return threadPool_->nthreads();
  }

 private:
  std::unique_ptr<ThreadPool> threadPool_;
};
}

#include <qmf/utils/ParallelExecutor-inl.h>
