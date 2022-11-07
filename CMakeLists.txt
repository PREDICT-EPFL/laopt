cmake_minimum_required(VERSION 3.5)
project(laopt)

#### User options ####
option(WITH_IPOPT "Build IPOPT interface" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)

#### Developer options ####
option(BUILD_RELEASE "Build release version w/o debugging symbols" ON)
option(ENABLE_SANITIZERS "Build with sanitizers enabled" OFF)

if ("$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
    list(APPEND compiler_flags -Wall -Wextra "-Wno-unused-parameter")
elseif ("$<COMPILE_LANG_AND_ID:CXX,MSVC>")
    list(APPEND compiler_flags -W3)
endif ()

#### Build laOPT library with selected options ####
## Set up configuration
if (BUILD_RELEASE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Building RELEASE")
else ()
    set(CMAKE_BUILD_TYPE "Debug")
    message(STATUS "Building DEBUG")
endif ()
if (ENABLE_SANITIZERS)
    list(APPEND sanitizer_flags "-fsanitize=address" "-fno-omit-frame-pointer")
    list(APPEND compiler_flags ${sanitizer_flags})
    message(STATUS "Building with sanitizers: ${sanitizer_flags}")
    unset(sanitizer_flags)
endif ()

## Find packages
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (DEFINED EIGEN3_INCLUDE_DIRS)
    # Create target for user-defined Eigen 3.4+ path to be used for laopt
    add_library(Eigen3.4 INTERFACE IMPORTED)
    target_include_directories(Eigen3.4 INTERFACE ${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}") # Use custom Eigen path instead of package
else ()
    # Wrap Eigen3 3.4+ from system package into imported target for consistent use afterwards
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
    add_library(Eigen3.4 INTERFACE IMPORTED)
    target_link_libraries(Eigen3.4 INTERFACE Eigen3::Eigen)
    message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}")
endif ()

## Find solvers, append to laOPT interface library
if (WITH_IPOPT)
    find_package(Ipopt REQUIRED)
    list(APPEND SOLVER_INCLUDE_DIRS ${Ipopt_INCLUDE_DIRS})
    list(APPEND SOLVER_LIBRARIES ${Ipopt_LIBRARIES})
endif ()

## Define laOPT interface library (to be linked into user code)
add_library(laopt INTERFACE)
target_include_directories(laopt INTERFACE include/ ${SOLVER_INCLUDE_DIRS})
target_link_libraries(laopt INTERFACE Eigen3.4 ${SOLVER_LIBRARIES})

target_compile_definitions(laopt INTERFACE -DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB - For both Windows and UNIX)

target_compile_features(laopt INTERFACE cxx_std_14)

target_compile_options(laopt INTERFACE ${compiler_flags})
target_link_options(laopt INTERFACE ${compiler_flags})
message(STATUS "Compiler flags: ${compiler_flags}")

unset(compiler_flags)

## Build optional components depending on laOPT library
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()