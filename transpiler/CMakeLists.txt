cmake_minimum_required(VERSION 3.13)
project(pybind_demo)
set(CMAKE_CXX_STANDARD 14)	

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.6.2
	GIT_SHALLOW    TRUE
)

include_directories(include /usr/local/include/ ${EIGEN3_INCLUDE_DIR})

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(example example.cpp)
target_compile_features(example PUBLIC cxx_std_14)
set_target_properties(example PROPERTIES SUFFIX ".so")
