#!/bin/bash

CASADI_VERSION=${CASADI_VERSION:-3.3.5};

echo "Installing CASADI..."

git clone https://github.com/casadi/casadi.git casadi
cd casadi
git checkout "$CASADI_VERSION"

mkdir build
cd build
cmake ..
make -j2
sudo make install
