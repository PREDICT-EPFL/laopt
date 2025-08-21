cmake_minimum_required(VERSION 3.5)

add_library(laopt-examples INTERFACE)
target_include_directories(laopt-examples INTERFACE include)
target_compile_options(laopt-examples INTERFACE ${compiler_flags})
target_link_options(laopt-examples INTERFACE ${compiler_flags})
target_link_libraries(laopt-examples INTERFACE laopt)

add_subdirectory(double_integrator)
add_subdirectory(inverted_pendulum)
add_subdirectory(chain_mass)
add_subdirectory(fixed_wing)
add_subdirectory(rocket)
