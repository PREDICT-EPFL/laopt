---
title: Installation
layout: default
parent: Getting Started
nav_order: 1
---

# Installation

{% root_include _common/requirements.md %}

## Build and install laOPT
Clone repository:
```shell
git clone https://github.com/PREDICT-EPFL/laopt.git
cd laopt
```

Build and install with cmake:
```shell
cmake -S . -B build -DLAOPT_BUILD_TESTS=OFF -DLAOPT_BUILD_EXAMPLES=OFF
cmake --build build
cmake --install build
```

_Alternatively, build and install with cmake / make_:
```shell
mkdir build && cd build
cmake .. -DLAOPT_BUILD_TESTS=OFF -DLAOPT_BUILD_EXAMPLES=OFF
make
make install
```

To install to a non-system location, set `CMAKE_INSTALL_PREFIX` during configuration or pass `--prefix` to `cmake --install`.

## How to use laOPT in CMake
If laOPT has been installed:
```cmake
find_package(laopt REQUIRED)
target_link_libraries(my_target PRIVATE laopt::laopt)
```

_Alternatively, include laOPT directly in a source tree_:
```cmake
add_subdirectory(path/to/laopt)
target_link_libraries(my_target PRIVATE laopt::laopt)
```

## Link a solver

`laopt::laopt` provides the core headers and Eigen dependency. A target using an external solver must be linked to that solver's libraries.

For example, an SQP application using PIQP links both laOPT and PIQP packages:
```cmake
find_package(laopt REQUIRED)
find_package(piqp REQUIRED)

add_executable(my_target_using_PIQP main.cpp)
target_link_libraries(my_target_using_PIQP PRIVATE
    laopt::laopt
    piqp::piqp
)
```

{: .note }
The `LAOPT_WITH_IPOPT`, `LAOPT_WITH_PIQP`, and other `LAOPT_WITH_*` options only enable dependencies for laOPT's own tests, examples, and benchmarks. They do not configure downstream solver dependencies.

## Build the examples

Enable examples together with the locally installed dependencies they use. For example:
```shell
cmake -S . -B build -DLAOPT_BUILD_EXAMPLES=ON -DLAOPT_WITH_PIQP=ON -DLAOPT_WITH_IPOPT=ON
```
or
```shell
cd build
cmake .. -DLAOPT_BUILD_EXAMPLES=ON -DLAOPT_WITH_PIQP=ON -DLAOPT_WITH_IPOPT=ON
```
 
