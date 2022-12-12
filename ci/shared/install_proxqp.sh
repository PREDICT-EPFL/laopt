#!/bin/bash

PROXQP_VERSION=${PROXQP_VERSION:-v0.2.13};

echo "Installing ProxQP..."

git git clone https://github.com/Simple-Robotics/proxsuite.git --recursive proxqp
cd proxqp
git checkout "$PROXQP_VERSION"

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make -j2
sudo make install
