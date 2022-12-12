#!/bin/bash

OSQP_VERSION=${OSQP_VERSION:-v0.6.2};

echo "Installing osqp..."

git clone --recursive https://github.com/osqp/osqp osqp
cd osqp
git checkout "$OSQP_VERSION"

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
sudo make install
