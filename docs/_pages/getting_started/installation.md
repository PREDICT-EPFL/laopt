---
title: Installation
layout: default
parent: Getting Started
nav_order: 1
---

# Installation

{% root_include _common/requirements.md %}

## Build and install laOPT

```shell
git clone https://github.com/PREDICT-EPFL/laopt.git
cd laopt
cmake -S . -B build \
      -DLAOPT_BUILD_TESTS=OFF \
      -DLAOPT_BUILD_EXAMPLES=OFF
cmake --build build
cmake --install build
```

Set `CMAKE_INSTALL_PREFIX` during configuration or pass `--prefix` to `cmake --install` to install into a non-system location.

## Use the core package

For an installed copy:

```cmake
find_package(laopt REQUIRED)
target_link_libraries(my_target PRIVATE laopt::laopt)
```

To include laOPT directly in a source tree:

```cmake
add_subdirectory(path/to/laopt)
target_link_libraries(my_target PRIVATE laopt::laopt)
```

## Link a solver

`laopt::laopt` provides the core headers and Eigen dependency. A consuming target must find and link the external solver used by the selected interface.

For example, an SQP application using PIQP links both packages:

```cmake
find_package(laopt REQUIRED)
find_package(piqp REQUIRED)

add_executable(my_target main.cpp)
target_link_libraries(my_target PRIVATE
    laopt::laopt
    piqp::piqp
)
```

{: .note }
The `LAOPT_WITH_IPOPT`, `LAOPT_WITH_PIQP`, and other `LAOPT_WITH_*` options only enable dependencies for laOPT's own tests, examples, and benchmarks. They do not configure downstream solver dependencies.

## Build the examples

Enable examples together with the locally installed dependencies they use. For example:

```shell
cmake -S . -B build \
      -DLAOPT_BUILD_EXAMPLES=ON \
      -DLAOPT_WITH_PIQP=ON \
      -DLAOPT_WITH_IPOPT=ON
cmake --build build
```

