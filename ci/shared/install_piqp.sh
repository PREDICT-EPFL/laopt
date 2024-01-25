#!/bin/bash

PIQP_VERSION=${PIQP_VERSION:-v0.2.4};

echo "Installing PIQP..."

git clone https://github.com/PREDICT-EPFL/piqp.git piqp
cd piqp
git checkout "$PIQP_VERSION"

mkdir build
cd build
cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
cmake --build .
sudo cmake --install .
