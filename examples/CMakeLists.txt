cmake_minimum_required(VERSION 3.5)

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


add_executable(
  double_integrator_discrete_ipopt
  double_integrator_discrete_ipopt.cpp
)
target_link_libraries(
  double_integrator_discrete_ipopt
  ipopt
)


add_executable(
  ipopt_example
  ipopt_example.cpp
)
target_link_libraries(
  ipopt_example
  ipopt
)

