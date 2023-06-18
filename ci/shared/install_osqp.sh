#!/bin/bash

OSQP_VERSION=${OSQP_VERSION:-v0.6.3};

echo "Installing osqp..."

git clone --recursive https://github.com/osqp/osqp osqp
cd osqp
git checkout "$OSQP_VERSION"
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
sudo make install
