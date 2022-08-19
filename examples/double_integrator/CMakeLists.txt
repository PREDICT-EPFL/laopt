cmake_minimum_required(VERSION 3.5)

add_executable(test_double_integrator
        double_integrator_test.cpp
)
target_link_libraries(test_double_integrator
        ${CASADI_LIBRARIES}
        ${Ipopt_LIBRARIES}
)