cmake_minimum_required(VERSION 3.5)
project(lampc_test)

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
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(func_test func_test.cpp)
target_link_libraries(func_test
        PRIVATE laopt gtest_main gtest)

add_executable(bs_matrix_test bs_matrix_test.cpp)
target_link_libraries(bs_matrix_test
        PRIVATE laopt gtest_main gtest)

add_executable(indexed_vector_test indexed_vector_test.cpp)
target_link_libraries(indexed_vector_test
        PRIVATE laopt gtest_main gtest)

add_executable(tag_test tag_test.cpp)
target_link_libraries(tag_test
        PRIVATE laopt)

if(WITH_IPOPT)
    add_executable(ipopt_test ipopt_test.cpp)
    target_link_libraries(ipopt_test
            PRIVATE laopt ${Ipopt_LIBRARIES} gtest_main gtest)
endif()

if(WITH_OSQP)
    add_executable(sqp_test sqp_test.cpp)
    target_link_libraries(sqp_test
            PRIVATE laopt gtest_main gtest)
endif()

include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_matrix_test)
gtest_discover_tests(indexed_vector_test)
if(WITH_IPOPT)
    gtest_discover_tests(ipopt_test)
endif()
gtest_discover_tests(sqp_test)
