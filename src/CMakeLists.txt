cmake_minimum_required(VERSION 3.5)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR} /opt/homebrew/Cellar/ipopt/3.14.2/include/coin-or/ /Users/cnjones/git/lampc/src)
link_directories(/Users/cnjones/git/lampc /opt/homebrew/Cellar/ipopt/3.14.2/lib/)

add_library(la_compiler STATIC la_compiler.cpp)
