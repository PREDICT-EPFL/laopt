#!/bin/bash

echo "Installing blasfeo..."

git clone --recursive https://github.com/giaf/blasfeo blasfeo
cd blasfeo

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j2
sudo cmake --install . --config Release
