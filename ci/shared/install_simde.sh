#!/bin/bash

SIMDE_VERSION=${SIMDE_VERSION:-v0.7.2};

echo "Installing SIMDe..."

git clone --recursive https://github.com/simd-everywhere/simde.git simde
cd simde
git checkout "$SIMDE_VERSION"

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
sudo make install
