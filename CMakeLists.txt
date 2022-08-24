cmake_minimum_required(VERSION 3.5)
project(laopt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake)

if(NOT DEFINED EIGEN3_INCLUDE_DIRS)
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")

find_package(Casadi REQUIRED)
find_package(Ipopt REQUIRED)

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB - For both Windows and UNIX

option(BUILD_RELEASE "Build release version w/o debugging symbols" ON)
option(ENABLE_SANITIZERS "Build with sanitizers enabled" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(ENABLE_SANITIZERS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Building release version")
else()
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Building debug version")
endif()

include_directories(${EIGEN3_INCLUDE_DIRS} ${CASADI_INCLUDE_DIR} ${Ipopt_INCLUDE_DIRS})
include_directories(include)

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(benchmarks)