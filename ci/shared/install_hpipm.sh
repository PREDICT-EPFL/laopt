#!/bin/bash

echo "Installing hpipm..."

git clone --recursive https://github.com/giaf/hpipm hpipm
cd hpipm

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
sudo make install
