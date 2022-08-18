cmake_minimum_required(VERSION 3.5)

add_executable(double_integrator_v0_test
        double_integrator_test.cpp
)
target_link_libraries(double_integrator_v0_test
        ${CASADI_LIBRARIES}
        ${Ipopt_LIBRARIES}
)