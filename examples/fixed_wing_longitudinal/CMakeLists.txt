cmake_minimum_required(VERSION 3.5)

find_package(yaml-cpp REQUIRED)

add_executable(test_fixed_wing_longitudinal
        fw_longitudinal_test.cpp
        )
target_link_libraries(test_fixed_wing_longitudinal
        PRIVATE laopt casadi yaml-cpp
        )
add_custom_command(
        TARGET test_fixed_wing_longitudinal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/eg4_xflr-Pvw-YR.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/eg4_xflr-Pvw-YR.yaml)