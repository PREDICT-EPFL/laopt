---
title: How to install Eigen on Ubuntu
layout: default
parent: Installation
nav_order: 2
---

# How to Install Eigen on Ubuntu
- (Recommended) Manually install the Eigen 3.4 package on your system: Download the package: https://packages.ubuntu.com/jammy/all/libeigen3-dev/download, install it: `sudo dpkg -i libeigen3-dev_3.4.0-2ubuntu2_all.deb`
- Clone from https://gitlab.com/libeigen/eigen, `git checkout 3.4`. When running `cmake` for laOPT, use the argument `-DEIGEN3_INCLUDE_DIRS=/path/to/Eigen`.

