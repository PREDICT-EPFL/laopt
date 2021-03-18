cmake_minimum_required(VERSION 3.5)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR})
link_directories(/Users/cnjones/git/lampc)

# if(BUILD_EXAMPLES)

# add_executable(mpc mpc.cpp)
# target_link_libraries(mpc hana Eigen3::Eigen)

add_executable(test test.cpp myproblem.hpp)
add_executable(model model.cpp gen.c)
add_executable(gen gen.cpp)
# add_executable(test4 test4.cpp)
# add_executable(test2 test2.cpp)

# target_link_libraries(model gen)

# target_include_directories(mpc PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
    # target_link_libraries(robot_control_test mobile_robot Eigen3::Eigen)

# get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message(STATUS "dir='${dir}'")
# endforeach()

# endif()
