cmake_minimum_required(VERSION 3.5)

add_executable(test_inverted_pendulum
        inverted_pendulum_test.cpp
)
target_link_libraries(test_inverted_pendulum
        PRIVATE laopt
)