cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/86add13493e5c881d7e4ba77fb91c1f57752b3a4.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

enable_testing()

add_library(laopt-test INTERFACE)
target_compile_options(laopt-test INTERFACE ${compiler_flags})
target_link_options(laopt-test INTERFACE ${compiler_flags})
target_link_libraries(laopt-test INTERFACE laopt gtest_main gtest)

add_executable(func_test func_test.cpp)
target_link_libraries(func_test PRIVATE laopt-test)

add_executable(bs_matrix_test bs_matrix_test.cpp)
target_link_libraries(bs_matrix_test PRIVATE laopt-test)

add_executable(differentiable_functions_test differentiable_functions_test.cpp)
target_link_libraries(differentiable_functions_test PRIVATE laopt-test)

add_executable(variable_map_test variable_map_test.cpp)
target_link_libraries(variable_map_test PRIVATE laopt-test)

add_executable(tag_test tag_test.cpp)
target_link_libraries(tag_test PRIVATE laopt-test)

if(WITH_IPOPT)
    add_executable(ipopt_test ipopt_test.cpp)
    target_link_libraries(ipopt_test PRIVATE laopt-test)
endif()

add_executable(sqp_test sqp_test.cpp)
target_link_libraries(sqp_test PRIVATE laopt-test)

include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_matrix_test)
gtest_discover_tests(differentiable_functions_test)
gtest_discover_tests(variable_map_test)
if(WITH_IPOPT)
    gtest_discover_tests(ipopt_test)
endif()
gtest_discover_tests(sqp_test)
