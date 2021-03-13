cmake_minimum_required(VERSION 3.5)
project(lampc)

# add_executable(lampc mpc.cpp)

# find_package(autodiff REQUIRED NO_MODULE)
# find_package(Hana REQUIRED)
# find_package(Boost REQUIRED COMPONENTS Hana)

if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# Show message stating path to Eigen3
message(STATUS "Using Eigen3 library found at ${EIGEN3_INCLUDE_DIR}")

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-DEIGEN_STACK_ALLOCATION_LIMIT=0 -O3 ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "-DEIGEN_STACK_ALLOCATION_LIMIT=0 -O3 -march=native -msse2 -ffast-math -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_VERBOSE_MAKEFILE “ON”)
if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
else()
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include_directories(src src/examples /usr/local/include/)
# add_subdirectory(src)
add_subdirectory(examples)
