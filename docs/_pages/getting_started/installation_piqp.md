---
title: Compatibility between PIQP and laOPT
layout: default
parent: Installation
nav_order: 2
---

# Compatibility between PIQP and laOPT
When building/installing PIQP (before laOPT), pay attention to the Eigen alignment settings (see "Warning" in the [PIQP docs](https://predict-epfl.github.io/piqp/interfaces/c_cpp/installation#building-and-installing-piqp)).
To ensure compatibility and make PIQP export the Eigen alignment settings with which it has been compiled, set `-DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON` when building PIQP, e.g.:

```
cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_BENCHMARKS=OFF -DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON
```
