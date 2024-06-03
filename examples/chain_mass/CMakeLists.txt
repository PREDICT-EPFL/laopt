cmake_minimum_required(VERSION 3.5)

add_executable(chain_mass_example chain_mass_example.cpp)
target_link_libraries(chain_mass_example PRIVATE laopt-examples)
