cmake_minimum_required(VERSION 3.5)

add_executable(rocket_example rocket_example.cpp)
target_link_libraries(rocket_example PRIVATE laopt-examples)
