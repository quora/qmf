# Copyright 2016 Quora, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8)
project(qmf)

# settings
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall -Wextra -Wuninitialized")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
set(CMAKE_FILES_DIRECTORY "build/")

# includes
include(FetchContent)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR})
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/qmf/third-party/gtest-1.7.0/fused-src)

# subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/qmf/third-party/gtest-1.7.0/fused-src/gtest)

set(SOURCES
    ${PROJECT_SOURCE_DIR}/qmf/DatasetReader.cpp
    ${PROJECT_SOURCE_DIR}/qmf/Engine.cpp
    ${PROJECT_SOURCE_DIR}/qmf/Matrix.cpp
    ${PROJECT_SOURCE_DIR}/qmf/Vector.cpp
    ${PROJECT_SOURCE_DIR}/qmf/bpr/BPREngine.cpp
    ${PROJECT_SOURCE_DIR}/qmf/metrics/Metrics.cpp
    ${PROJECT_SOURCE_DIR}/qmf/metrics/MetricsEngine.cpp
    ${PROJECT_SOURCE_DIR}/qmf/metrics/MetricsManager.cpp
    ${PROJECT_SOURCE_DIR}/qmf/wals/WALSEngine.cpp
    ${PROJECT_SOURCE_DIR}/qmf/utils/IdIndex.cpp
    ${PROJECT_SOURCE_DIR}/qmf/utils/ThreadPool.cpp
    ${PROJECT_SOURCE_DIR}/qmf/utils/Util.cpp
)

set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  glog  
  GIT_REPOSITORY https://github.com/google/glog.git 
  GIT_TAG v0.4.0 
  GIT_PROGRESS TRUE
)
FetchContent_Declare(glog)
FetchContent_MakeAvailable(glog)

FetchContent_Declare(
  gflags  
  GIT_REPOSITORY https://github.com/gflags/gflags.git 
  GIT_TAG v2.2.2
  GIT_PROGRESS TRUE
)
FetchContent_Declare(gflags)
FetchContent_MakeAvailable(gflags)

add_library(qmf STATIC ${SOURCES})
target_link_libraries(qmf glog gflags lapack)

# binaries
macro(make_binary binary_source binary_name)
    add_executable(${binary_name} qmf/${binary_source})
    target_link_libraries(${binary_name} qmf pthread)
    set_target_properties(${binary_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin/")
endmacro(make_binary)

make_binary(bpr.cpp bpr)
make_binary(wals.cpp wals)

# unit testing
macro(make_test test_source test_name)
    add_executable(${test_name} qmf/test/${test_source})
    target_link_libraries(${test_name} qmf gtest gtest_main lapack pthread)
    set_target_properties(${test_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "test/")
    add_test(${test_name} test/${test_name})
endmacro(make_test)

enable_testing()
make_test(BPREngineTest.cpp BPREngineTest)
make_test(DatasetReaderTest.cpp DatasetReaderTest)
make_test(EngineTest.cpp EngineTest)
make_test(FactorDataTest.cpp FactorDataTest)
make_test(MatrixTest.cpp MatrixTest)
make_test(MetricsTest.cpp MetricsTest)
make_test(MetricsManagerTest.cpp MetricsManagerTest)
make_test(ParallelExecutorTest.cpp ParallelExecutorTest)
make_test(ThreadPoolTest.cpp ThreadPoolTest)
make_test(UtilTest.cpp UtilTest)
make_test(VectorTest.cpp VectorTest)
make_test(WALSEngineTest.cpp WALSEngineTest)
