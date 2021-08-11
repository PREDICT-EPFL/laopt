cmake_minimum_required(VERSION 3.5)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR} /usr/local/Cellar/ipopt/3.13.4/include/coin-or/ /usr/local/Cellar/fmt/7.1.3/include /Users/cnjones/git/lampc/src)
link_directories(/Users/cnjones/git/lampc /usr/local/Cellar/ipopt/3.13.4/lib/ /usr/local/Cellar/fmt/7.1.3/lib)

add_definitions(-std=c++11 -DEIGEN_MATRIX_PLUGIN="Matrix_initializer_list.h" -DEIGEN_ARRAY_PLUGIN="Array_initializer_list.h")

add_executable(bob bob.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
target_link_libraries(bob ipopt Eigen3::Eigen)

add_executable(continuous_bob continuous_bob.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
target_link_libraries(continuous_bob ipopt Eigen3::Eigen)

# add_executable(simple_ipopt_test simple_ipopt_test.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(simple_ipopt_test ipopt Eigen3::Eigen)

add_executable(polybob polybob.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
target_link_libraries(polybob ipopt Eigen3::Eigen)

add_executable(qp_test qp_test.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
target_link_libraries(qp_test ipopt Eigen3::Eigen)

add_executable(cstr cstr.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
target_link_libraries(cstr ipopt Eigen3::Eigen)

add_executable(double_integrator double_integrator.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
target_link_libraries(double_integrator ipopt Eigen3::Eigen)


# link_directories(${CMAKE_SOURCE_DIR}/third_party/opencv/lib)
