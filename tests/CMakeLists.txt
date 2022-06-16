cmake_minimum_required(VERSION 3.5)
project(lampc_test)

# Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  func_test
  func_test.cpp
)
target_link_libraries(
  func_test
  gtest_main
  gtest
)

add_executable(
  bs_test
  bs_test.cpp
)
target_link_libraries(
  bs_test
  gtest_main
  gtest
)

add_executable(
  problem_test
  problem_test.cpp
)
target_link_libraries(
  problem_test
  gtest_main
  gtest
)

add_executable(
  indexedvector_test
  indexedvector_test.cpp
)
target_link_libraries(
  indexedvector_test
  gtest_main
  gtest
)

add_executable(
  ipopt_test
  ipopt_test.cpp
)
target_link_libraries(
  ipopt_test
  ${Ipopt_LIBRARIES}
  gtest_main
  gtest
)

include(GoogleTest)
gtest_discover_tests(func_test)
gtest_discover_tests(bs_test)
gtest_discover_tests(problem_test)
