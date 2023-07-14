cmake_minimum_required(VERSION 3.5)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/84c71faa8126e4eedc2bb520352615cb4484d6ad.zip
)
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

find_package(Casadi REQUIRED)

add_executable(simple_function_benchmark simple_function_benchmark.cpp simple_function_benchmark_casadi_codegen.cpp)
target_link_libraries(simple_function_benchmark PRIVATE laopt benchmark::benchmark casadi)

add_executable(chain_mass_benchmark chain_mass_benchmark.cpp chain_mass_benchmark_casadi_codegen.cpp)
target_link_libraries(chain_mass_benchmark PRIVATE laopt benchmark::benchmark casadi)