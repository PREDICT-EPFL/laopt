cmake_minimum_required(VERSION 3.5)

add_executable(test_chain_mass
        chain_mass_test.cpp
        )
target_link_libraries(test_chain_mass
        PRIVATE laopt
        )