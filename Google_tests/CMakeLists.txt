cmake_minimum_required(VERSION 3.5)
project(lampc_test)

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


include_directories(
	include 
	/usr/local/include/ 
	${EIGEN3_INCLUDE_DIR} 
	/opt/homebrew/include/coin-or/
	/opt/homebrew/Cellar/fmt/8.0.1/include
	/Users/cnjones/git/lampc/src
)

link_directories(
	/Users/cnjones/git/lampc 
	/opt/homebrew/lib/
	/opt/homebrew/Cellar/fmt/8.0.1/lib
)

enable_testing()

add_executable(
  func_test
  func_test.cpp
)
target_link_libraries(
  func_test
  gtest_main
  gtest
)

add_executable(
  bs_test
  bs_test.cpp
)
target_link_libraries(
  bs_test
  gtest_main
  gtest
)

add_executable(
  problem_test
  problem_test.cpp
)
target_link_libraries(
  problem_test
  gtest_main
  gtest
)

add_executable(
  indexedvector_test
  indexedvector_test.cpp
)
target_link_libraries(
  indexedvector_test
  gtest_main
  gtest
)

add_executable(
  ipopt_test
  ipopt_test.cpp
)
target_link_libraries(
  ipopt_test
  gtest_main
  gtest
  ipopt
)


include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_test)
gtest_discover_tests(problem_test)
