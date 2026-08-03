#!/bin/bash

set -euo pipefail

case $(uname -m) in
    x86_64|AMD64)
        BLASFEO_TARGET=X64_INTEL_HASWELL
        ;;
    aarch64|arm64|ARM64)
        BLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76
        ;;
    *)
        echo "Unsupported architecture for BLASFEO target: $(uname -m)" >&2
        exit 1
        ;;
esac

echo "Installing blasfeo with TARGET=${BLASFEO_TARGET}..."

git clone --recursive https://github.com/giaf/blasfeo blasfeo
cd blasfeo

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTARGET="${BLASFEO_TARGET}"
cmake --build . --config Release -- -j2
sudo cmake --install . --config Release
