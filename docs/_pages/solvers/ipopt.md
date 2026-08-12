---
title: IPOPT
layout: default
parent: Solvers
nav_order: 2
---

# IPOPT

The IPOPT interface adapts a `laopt::Problem` to IPOPT's `TNLP` interface and supplies sparse first- and second-order derivatives.

```cpp
#include <laopt/solvers/ipopt_interface.hpp>

laopt::IpoptSolver<Problem> solver(problem);
solver.set_tol(1e-8);
solver.set_max_iter(1000);

auto status = solver.solve();
const auto& solution = solver.primal();
```

The consuming project must find and link IPOPT in addition to `laopt::laopt`. laOPT's `LAOPT_WITH_IPOPT` option is only used when building laOPT's own targets.

IPOPT options can be accessed through `solver.Options()` or through the convenience setters for tolerance, maximum iterations, banner output, print level, and timing output.

