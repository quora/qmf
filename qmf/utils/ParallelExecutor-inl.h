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

namespace qmf {

template <typename FuncT>
void ParallelExecutor::execute(const size_t ntasks, FuncT&& func) {
  const size_t nthreads = threadPool_->nthreads();
  std::vector<std::future<void>> futures;
  futures.reserve(nthreads);
  for (size_t threadId = 0; threadId < nthreads; ++threadId) {
    auto task = [threadId, func, nthreads, ntasks]() {
      for (size_t taskId = threadId; taskId < ntasks; taskId += nthreads) {
        func(taskId);
      }
    };
    futures.emplace_back(threadPool_->addTask(task));
  }
  for (auto& future : futures) {
    future.get();
  }
}

template <typename T, typename MapperT, typename ReducerT>
T ParallelExecutor::mapReduce(const size_t ntasks,
                              MapperT&& mapper,
                              ReducerT&& reducer,
                              T neutral) {
  const size_t nthreads = threadPool_->nthreads();
  std::vector<std::future<T>> futures;
  futures.reserve(nthreads);
  for (size_t threadId = 0; threadId < nthreads; ++threadId) {
    auto task = [threadId, mapper, reducer, neutral, nthreads, ntasks]() {
      T res = neutral;
      for (size_t taskId = threadId; taskId < ntasks; taskId += nthreads) {
        res = reducer(res, mapper(taskId));
      }
      return res;
    };
    futures.emplace_back(threadPool_->addTask(task));
  }
  return std::accumulate(
    futures.begin(), futures.end(), neutral,
    [reducer](T res, auto& future) { return reducer(res, future.get()); });
}

template <typename T, typename ElemT, typename MapperT, typename ReducerT>
T ParallelExecutor::mapReduce(const std::vector<ElemT>& elems,
                              MapperT&& mapper,
                              ReducerT&& reducer,
                              T neutral) {
  const size_t nelems = elems.size();
  const size_t nthreads = threadPool_->nthreads();
  std::vector<std::future<T>> futures;
  futures.reserve(nthreads);
  for (size_t threadId = 0; threadId < nthreads; ++threadId) {
    auto task =
      [&elems, threadId, mapper, reducer, neutral, nthreads, nelems]() {
        const size_t blockSize = nelems / nthreads;
        return std::accumulate(
          elems.begin() + threadId * blockSize,
          elems.begin() + std::min((threadId + 1) * blockSize, nelems),
          neutral, [mapper, reducer](T res, const auto& elem) {
            return reducer(res, mapper(elem));
          });
      };
    futures.emplace_back(threadPool_->addTask(task));
  }
  return std::accumulate(
    futures.begin(), futures.end(), neutral,
    [reducer](T res, auto& future) { return reducer(res, future.get()); });
}
}
