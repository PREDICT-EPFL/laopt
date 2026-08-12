cmake_minimum_required(VERSION 3.21)
project(laopt_first_optimal_control_problem LANGUAGES CXX)

get_filename_component(LAOPT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(EXISTS "${LAOPT_SOURCE_DIR}/include/laopt/laopt.hpp")
    set(LAOPT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LAOPT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LAOPT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(LAOPT_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory("${LAOPT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/laopt")
else()
    find_package(laopt REQUIRED)
endif()
find_package(piqp REQUIRED)

add_executable(generate_trajectory generate_trajectory.cpp)
target_compile_features(generate_trajectory PRIVATE cxx_std_17)
target_link_libraries(generate_trajectory PRIVATE
    laopt::laopt
    piqp::piqp
)
