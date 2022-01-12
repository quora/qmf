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

#if __GNUC__ >= 3
 #include <functional>
#endif

#include <condition_variable>

#include <glog/logging.h>

namespace qmf {

template <typename FuncT, typename... Args>
auto ThreadPool::addTask(FuncT&& func, Args&&... args)
  -> std::future<typename std::result_of<FuncT(Args...)>::type> {

  using ResultT = typename std::result_of<FuncT(Args...)>::type;

  auto package = std::make_shared<std::packaged_task<ResultT()>>(
    std::bind(std::forward<FuncT>(func), std::forward<Args>(args)...));
  auto result = package->get_future();
  WITH_LOCK(mutex_) {
    CHECK(!poison_) << "destructor was called";
    tasks_.emplace([package]{ (*package)(); });
  }
  cond_.notify_one();
  return result;
}
}
