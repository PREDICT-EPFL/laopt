---
title: SQP
layout: default
parent: Solvers
nav_order: 1
---

# Sequential Quadratic Programming

The native SQP solver repeatedly linearizes the constraints and solves a quadratic approximation of the nonlinear program.

```cpp
#include <laopt/solvers/sqp_solver.hpp>
#include <laopt/solvers/piqp_interface.hpp>

using QPSolver = laopt::PIQPSolver<double>;
laopt::SQPSolver<Problem, QPSolver> solver(problem);

solver.settings().verbose = true;
solver.settings().hessian_approximation = laopt::hessian_approximation_t::EXACT;

auto info = solver.solve();
```

The solver supports exact Lagrangian Hessians and a Gauss-Newton approximation. Available globalization strategies are full steps, an L1 merit-function line search, and a filter line search.

The underlying QP solver is available through `solver.qp_solver()` for backend-specific inspection and common QP settings.

See [SQP Settings and Status]({{ site.baseurl }}/api/sqp) for the common configuration and result fields.
