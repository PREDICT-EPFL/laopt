cmake_minimum_required(VERSION 3.21)

find_package(yaml-cpp REQUIRED)

add_executable(fixed_wing_longitudinal_example fixed_wing_longitudinal_example.cpp)
target_link_libraries(fixed_wing_longitudinal_example PRIVATE laopt-examples yaml-cpp::yaml-cpp)
laopt_target_enable_casadi(fixed_wing_longitudinal_example)
laopt_target_enable_ipopt(fixed_wing_longitudinal_example)
laopt_target_enable_piqp(fixed_wing_longitudinal_example)
add_custom_command(
        TARGET fixed_wing_longitudinal_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/fixed_wing_model/eg4_xflr-Pvw-YR.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/
)

add_executable(fixed_wing_example fixed_wing_example.cpp)
target_link_libraries(fixed_wing_example PRIVATE laopt-examples yaml-cpp::yaml-cpp)
laopt_target_enable_casadi(fixed_wing_example)
laopt_target_enable_ipopt(fixed_wing_example)
laopt_target_enable_piqp(fixed_wing_example)
add_custom_command(
        TARGET fixed_wing_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/fixed_wing_model/eg4_xflr-Pvw-YR.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/
)
