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
  /Users/cnjones/git/lampc/src/lampc_function.hpp
)
target_link_libraries(
  func_test
  gtest_main
  gtest
)

#  ipopt
#  Eigen3::Eigen

# add_executable(
#   bs_test
#   bs_test.cpp
# )
# target_link_libraries(
#   bs_test
#   gtest_main
#   gtest
# )

# add_executable(
#   con_test
#   constraints_test.cpp
# )
# target_link_libraries(
#   con_test
#   gtest_main
#   gtest
# )

#  ipopt
#  Eigen3::Eigen


include(GoogleTest)
gtest_discover_tests(func_test)
