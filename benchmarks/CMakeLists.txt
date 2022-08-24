cmake_minimum_required(VERSION 3.5)

find_package(benchmark REQUIRED)

add_executable(simple_function_benchmark simple_function_benchmark.cpp simple_function_benchmark_casadi_codegen.cpp)
target_link_libraries(simple_function_benchmark benchmark::benchmark ${CASADI_LIBRARIES})
