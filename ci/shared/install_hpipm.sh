#!/bin/bash

set -euo pipefail

case $(uname -m) in
    x86_64|AMD64)
        HPIPM_TARGET=AVX
        ;;
    *)
        HPIPM_TARGET=GENERIC
        ;;
esac

echo "Installing hpipm with TARGET=${HPIPM_TARGET}..."

git clone --recursive https://github.com/giaf/hpipm hpipm
cd hpipm

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTARGET="${HPIPM_TARGET}"
cmake --build . --config Release -- -j2
sudo cmake --install . --config Release
