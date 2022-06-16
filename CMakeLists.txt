cmake_minimum_required(VERSION 3.5)
project(lampc)

set(CMAKE_MODULE_PATH /usr/local/share/cmake/Modules/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake)

#set(EIGEN3_INCLUDE_DIR /absolute/path/eigen) # If find_package does not find required package, clone eigen source, checkout v3.4 or higher, and set the path here
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()
find_package(CASADI REQUIRED)
find_package(Ipopt REQUIRED)

# Show message stating path to Eigen3 and IPOPT
message(STATUS "Using Eigen3 library found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Using IPOPT library found at ${Ipopt_INCLUDE_DIRS}")

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB - For both Windows and UNIX

# include_directories(polympc/src)
# include_directories(qpmad/include/)

option(BUILD_RELEASE "Build release version w/o debugging symbols" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_VERBOSE_MAKEFILE "ON")

if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Building release version")
else()
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Building debug version")
endif()


include_directories(src ${EIGEN3_INCLUDE_DIR} ${CASADI_INCLUDE_DIR} ${Ipopt_INCLUDE_DIRS})

add_subdirectory(examples)
add_subdirectory(transpiler)
add_subdirectory(tests)
