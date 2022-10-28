# laOPT
## Installation
### Eigen
laOPT requires Eigen v3.4 or higher. If not provided by your system's package manager, there are two options:
- (Recommended) Manually install the Eigen 3.4 package on your system: Download the package: https://packages.ubuntu.com/jammy/all/libeigen3-dev/download, install it: `sudo dpkg -i libeigen3-dev_3.4.0-2ubuntu2_all`
- Clone from https://gitlab.com/libeigen/eigen, `git checkout 3.4`. When running cmake for laOPT, use the argument `-DEIGEN3_INCLUDE_DIRS=/path/to/Eigen`.
