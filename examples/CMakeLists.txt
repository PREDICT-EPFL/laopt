cmake_minimum_required(VERSION 3.5)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR} /opt/homebrew/Cellar/ipopt/3.14.2/include/coin-or/ /opt/homebrew/Cellar/fmt/8.0.1/include /Users/cnjones/git/lampc/src)
link_directories(/Users/cnjones/git/lampc /opt/homebrew/Cellar/ipopt/3.14.2/lib/ /opt/homebrew/Cellar/fmt/8.0.1/lib)

add_executable(qp_make qp_make.cpp)
add_executable(qp qp.cpp)
