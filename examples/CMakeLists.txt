cmake_minimum_required(VERSION 3.5)

include_directories(
        base
        transcription
)
#add_subdirectory(double_integrator_old)
add_subdirectory(double_integrator)
add_subdirectory(fixed_wing_longitudinal)
add_subdirectory(inverted_pendulum)