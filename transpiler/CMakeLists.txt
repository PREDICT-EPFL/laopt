cmake_minimum_required(VERSION 3.5)
project(pybind_demo)

add_subdirectory(submodules/pybind11)

pybind11_add_module(example example.cpp)
target_compile_features(example PUBLIC cxx_std_14)
set_target_properties(example PROPERTIES SUFFIX ".so")
