cmake_minimum_required(VERSION 3.5)
project(lampc)

if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# Show message stating path to Eigen3
message(STATUS "Using Eigen3 library found at ${EIGEN3_INCLUDE_DIR}")

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=10000000) # 1 MB - For both Windows and UNIX
# add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB - For both Windows and UNIX
include_directories(polympc/src)
include_directories(qpmad/include/)



set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-O3 -march=native -msse2 -ffast-math -Wall -ftime-trace")
set(CMAKE_CXX_FLAGS "-O3 -msse2 -ffast-math -Wall -ftime-trace")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_VERBOSE_MAKEFILE “ON”)
if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
else()
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include_directories(src src/examples /usr/local/include/)

add_subdirectory(examples)
add_subdirectory(polympc)
add_subdirectory(qpmad  )



# add_executable(lampc mpc.cpp)
# find_package(autodiff REQUIRED NO_MODULE)
# find_package(Hana REQUIRED)
# find_package(Boost REQUIRED COMPONENTS Hana)

# set(CMAKE_CXX_FLAGS "-DEIGEN_STACK_ALLOCATION_LIMIT=0 -O3 -march=native -msse2 -ffast-math -Wall -ftime-trace")
