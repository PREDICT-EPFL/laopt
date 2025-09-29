# laOPT
## Installation
### Dependencies
#### Eigen
laOPT requires Eigen v3.4 or higher. If not provided by your system's package manager, there are two options:
- (Recommended) Manually install the Eigen 3.4 package on your system: Download the package: https://packages.ubuntu.com/jammy/all/libeigen3-dev/download, install it: `sudo dpkg -i libeigen3-dev_3.4.0-2ubuntu2_all.deb`
- Clone from https://gitlab.com/libeigen/eigen, `git checkout 3.4`. When running `cmake` for laOPT, use the argument `-DEIGEN3_INCLUDE_DIRS=/path/to/Eigen`.

#### OSQP (Optional)
laOPT is not yet compatible with the current version of OSQP.
Clone OSQP v0.6.3:
```
git clone -b v0.6.3 --recurse-submodules https://github.com/osqp/osqp.git`
```

#### PIQP
When building PIQP (before laOPT), https://predict-epfl.github.io/piqp/interfaces/c_cpp/installation#building-and-installing-piqp, add `-DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON` such that PIQP exports its architecture settings to laOPT:
```
cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON
```

### Make and Install laOPT
As laOPT is a header-only library, it can either be directly included as a subfolder of your cmake project, or be installed globally on your system.
To install globally, go to the cloned laOPT folder and run:
```
mkdir build && cd build/
cmake ..
make install
```
The headers will be copied to your system-wide include folders.
