#!/bin/bash

echo "Installing blasfeo..."

git clone --recursive https://github.com/giaf/blasfeo blasfeo
cd blasfeo

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
sudo make install
