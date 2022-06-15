cmake_minimum_required(VERSION 3.5)
project(lampc_test)

enable_testing()

add_executable(
  func_test
  func_test.cpp
)
target_link_libraries(
  func_test
  GTest::Main
  GTest::GTest
)

add_executable(
  bs_test
  bs_test.cpp
)
target_link_libraries(
  bs_test
  GTest::Main
  GTest::GTest
)

add_executable(
  problem_test
  problem_test.cpp
)
target_link_libraries(
  problem_test
  GTest::Main
  GTest::GTest
)

add_executable(
  indexedvector_test
  indexedvector_test.cpp
)
target_link_libraries(
  indexedvector_test
  GTest::Main
  GTest::GTest
)

add_executable(
  ipopt_test
  ipopt_test.cpp
)
target_link_libraries(
  ipopt_test
  ${Ipopt_LIBRARIES}
  GTest::Main
  GTest::GTest
)

include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_test)
gtest_discover_tests(problem_test)
