# First Optimal Control Problem Plot

This directory contains the source used to generate the trajectory plot in the
First Optimal Control Problem tutorial. Its underscore-prefixed path keeps it out of
the generated Jekyll site.

From this directory, run:

```shell
cmake -S . -B build
cmake --build build
./build/generate_trajectory trajectory.csv
python3 plot_trajectory.py trajectory.csv ../../assets/images/inverted_pendulum_solution.svg
```

The plotting script requires Matplotlib.

The CMake project uses the surrounding laOPT source tree when it is available;
otherwise, it looks for an installed laOPT package. PIQP must be installed in
either case.
