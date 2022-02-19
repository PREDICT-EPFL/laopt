cmake_minimum_required(VERSION 3.5)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR} /opt/homebrew/Cellar/ipopt/3.14.2/include/coin-or/ /opt/homebrew/Cellar/fmt/8.0.1/include /Users/cnjones/git/lampc/src)
link_directories(/Users/cnjones/git/lampc /opt/homebrew/Cellar/ipopt/3.14.2/lib/ /opt/homebrew/Cellar/fmt/8.0.1/lib)

# add_executable(bob bob.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(bob ipopt Eigen3::Eigen)

# add_executable(continuous_bob continuous_bob.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(continuous_bob ipopt Eigen3::Eigen)

# add_executable(simple_ipopt_test simple_ipopt_test.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(simple_ipopt_test ipopt Eigen3::Eigen)

# add_executable(polybob polybob.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(polybob ipopt Eigen3::Eigen)

# add_executable(qp_test qp_test.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(qp_test ipopt Eigen3::Eigen)

# add_executable(cstr cstr.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(cstr ipopt Eigen3::Eigen)

# add_executable(size_test size_test.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(size_test Eigen3::Eigen)

# add_executable(size_test size_test.cpp)
# target_link_libraries(size_test)

add_executable(bstest bstest.cpp)

# add_executable(index_test index_test.cpp)

add_executable(testCompiler testCompiler.cpp)
add_executable(testSolver test_lasolver.cpp)

add_executable(testQP_compile test_solveqp.cpp)
# target_compile_definitions(testQP_compile PUBLIC MAKE_PROBLEM=1)

# add_executable(LinearMPC.compiled.hpp test_solveqp.cpp)

add_executable(testQP test_solveqp.cpp)
# add_dependencies(testQP LinearMPC.compiled.hpp)


# add_executable(double_integrator double_integrator.cpp /Users/cnjones/git/lampc/src/lampc.hpp)
# target_link_libraries(double_integrator ipopt Eigen3::Eigen)


# link_directories(${CMAKE_SOURCE_DIR}/third_party/opencv/lib)
