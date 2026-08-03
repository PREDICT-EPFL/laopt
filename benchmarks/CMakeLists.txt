cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Keep the explicit FetchContent populate/add_subdirectory flow for CMake 3.21
# support while still excluding third-party install rules.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
    cmake_policy(SET CMP0169 OLD)
endif()

# Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/84c71faa8126e4eedc2bb520352615cb4484d6ad.zip
    URL_HASH SHA256=bcab43290d4ef38e8a44af8ecc93755712f6fc2080e4f0bfc4ceef4b62834dcb
)
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(laopt-benchmark INTERFACE)
target_compile_options(laopt-benchmark INTERFACE ${laopt_compile_options})
target_link_options(laopt-benchmark INTERFACE ${laopt_link_options})
target_link_libraries(laopt-benchmark INTERFACE laopt::laopt benchmark::benchmark)

add_library(laopt-benchmark-casadi INTERFACE)
target_link_libraries(laopt-benchmark-casadi INTERFACE laopt-benchmark)
laopt_target_enable_casadi(laopt-benchmark-casadi)

add_executable(simple_function_benchmark simple_function_benchmark.cpp simple_function_benchmark_casadi_codegen.cpp)
target_link_libraries(simple_function_benchmark PRIVATE laopt-benchmark-casadi)

add_executable(chain_mass_benchmark chain_mass_benchmark.cpp chain_mass_benchmark_casadi_codegen.cpp)
target_link_libraries(chain_mass_benchmark PRIVATE laopt-benchmark-casadi)
