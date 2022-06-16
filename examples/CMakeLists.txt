cmake_minimum_required(VERSION 3.5)

add_executable(
  double_integrator_discrete_ipopt
  double_integrator_discrete_ipopt.cpp
)
target_link_libraries(
  double_integrator_discrete_ipopt
  ${CASADI_LIBRARIES}
  ${Ipopt_LIBRARIES}
)


add_executable(
  ipopt_example
  ipopt_example.cpp
)
target_link_libraries(
  ipopt_example
  ${CASADI_LIBRARIES}
  ${Ipopt_LIBRARIES}
)

add_library(
  double_integrator_tape
  double_integrator_tape.cpp
)

add_library(
  double_integrator_problem
  double_integrator_problem.cpp
)

add_executable(
  double_integrator
  double_integrator.cpp
)
target_link_libraries(
  double_integrator
  double_integrator_tape
  double_integrator_problem
  ${CASADI_LIBRARIES}
  ${Ipopt_LIBRARIES}
)

add_subdirectory(double_integrator_v0)
