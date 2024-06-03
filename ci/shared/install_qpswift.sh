#!/bin/bash

QPSWIFT_VERSION=${QPSWIFT_VERSION:-v0.0.2};

echo "Installing QPSwift..."

git clone --recursive https://github.com/qpSWIFT/qpSWIFT.git qpswift
cd qpswift
git checkout "$QPSWIFT_VERSION"

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j2
sudo cmake --install . --config Release
