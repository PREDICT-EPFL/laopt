#!/bin/bash

IPOPT_VERSION=${IPOPT_VERSION:-3.14.6};

echo "Installing IPOPT using coinbrew..."

mkdir -p coinbrew
cd coinbrew
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew

./coinbrew fetch Ipopt@"$IPOPT_VERSION"
sudo ./coinbrew build Ipopt --parallel-jobs 4 --build-dir build-ipopt --tests none --prefix=/usr/local
