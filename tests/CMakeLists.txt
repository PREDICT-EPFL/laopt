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

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
    URL_HASH SHA256=a9607c9215866bd425a725610c5e0f739eeb50887a57903df48891446ce6fb3c
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
target_compile_options(laopt-test INTERFACE ${laopt_compile_options})
target_link_options(laopt-test INTERFACE ${laopt_link_options})
target_link_libraries(laopt-test INTERFACE GTest::gtest_main GTest::gtest laopt::laopt)

add_library(laopt-test-casadi INTERFACE)
target_link_libraries(laopt-test-casadi INTERFACE laopt-test)
laopt_target_enable_casadi(laopt-test-casadi)

add_library(laopt-test-sqp INTERFACE)
target_link_libraries(laopt-test-sqp INTERFACE laopt-test)
laopt_target_enable_qpalm(laopt-test-sqp)
laopt_target_enable_piqp(laopt-test-sqp)

if (LAOPT_WITH_IPOPT)
    add_library(laopt-test-ipopt INTERFACE)
    target_link_libraries(laopt-test-ipopt INTERFACE laopt-test)
    laopt_target_enable_ipopt(laopt-test-ipopt)
endif()

add_executable(func_test func_test.cpp)
target_link_libraries(func_test PRIVATE laopt-test-casadi)

add_executable(bs_matrix_test bs_matrix_test.cpp)
target_link_libraries(bs_matrix_test PRIVATE laopt-test)

add_executable(differentiable_functions_test differentiable_functions_test.cpp)
target_link_libraries(differentiable_functions_test PRIVATE laopt-test)

add_executable(variable_map_test variable_map_test.cpp)
target_link_libraries(variable_map_test PRIVATE laopt-test)

add_executable(tag_test tag_test.cpp)
target_link_libraries(tag_test PRIVATE laopt-test)

if(LAOPT_WITH_IPOPT)
    add_executable(ipopt_test ipopt_test.cpp)
    target_link_libraries(ipopt_test PRIVATE laopt-test-ipopt)
endif()

add_executable(sqp_test sqp_test.cpp)
target_link_libraries(sqp_test PRIVATE laopt-test-sqp)

include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_matrix_test)
gtest_discover_tests(differentiable_functions_test)
gtest_discover_tests(variable_map_test)
gtest_discover_tests(tag_test)
if(LAOPT_WITH_IPOPT)
    gtest_discover_tests(ipopt_test)
endif()
if(WITH_OSQP)
    gtest_discover_tests(sqp_test)
endif()