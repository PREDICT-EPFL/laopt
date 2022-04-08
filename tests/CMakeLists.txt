cmake_minimum_required(VERSION 3.5)
project(lampc_test)

include_directories(
	include 
	/usr/local/include/ 
	${EIGEN3_INCLUDE_DIR} 
	/opt/homebrew/Cellar/ipopt/3.14.2/include/coin-or/ 
	/opt/homebrew/Cellar/fmt/8.0.1/include 
	/opt/homebrew/opt/googletest/include
	/Users/cnjones/git/lampc/src
)

link_directories(
	/Users/cnjones/git/lampc 
	/opt/homebrew/Cellar/ipopt/3.14.2/lib/ 
	/opt/homebrew/Cellar/fmt/8.0.1/lib
	/opt/homebrew/opt/googletest/lib
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

include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_test)
gtest_discover_tests(problem_test)
