#!/bin/bash

EIGEN_VERSION=${OSQP_EIGEN_VERSION:-v0.8.0};

echo "Installing osqp-eigen..."

git clone https://github.com/robotology/osqp-eigen.git osqp-eigen
cd osqp-eigen
git checkout "$OSQP_EIGEN_VERSION"

mkdir build
cd build
cmake ..
make -j4
sudo make install
