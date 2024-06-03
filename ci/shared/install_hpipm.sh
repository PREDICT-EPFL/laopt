#!/bin/bash

echo "Installing hpipm..."

git clone --recursive https://github.com/giaf/hpipm hpipm
cd hpipm

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j2
sudo cmake --install . --config Release
