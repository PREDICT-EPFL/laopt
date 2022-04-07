cmake_minimum_required(VERSION 3.5)
project(lampc)

if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# find_package(fmt)
 

# Show message stating path to Eigen3
message(STATUS "Using Eigen3 library found at ${EIGEN3_INCLUDE_DIR}")

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB - For both Windows and UNIX

# include_directories(polympc/src)
# include_directories(qpmad/include/)

option(BUILD_RELEASE "Build release version w/o debugging symbols" ON)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-O3 -march=native -msse2 -ffast-math -Wall -ftime-trace")
# set(CMAKE_CXX_FLAGS "-O3 -msse2 -ffast-math -Wall -ftime-trace")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

set(CMAKE_VERBOSE_MAKEFILE “ON”)
if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Building release version")
else()
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Building debug version")
endif()



include_directories(src bsmatrix src/examples /usr/local/include/)

add_subdirectory(src     )
add_subdirectory(examples)
add_subdirectory(transpiler)
# add_subdirectory(polympc)
# add_subdirectory(qpmad  )
add_subdirectory(tests  )

