cmake_minimum_required(VERSION 2.8)

add_library(gtest STATIC EXCLUDE_FROM_ALL gtest.h gtest-all.cc)
add_library(gtest_main STATIC gtest_main.cc)
target_link_libraries(gtest_main gtest)
