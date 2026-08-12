---
title: SQP Settings and Status
layout: default
parent: API
nav_order: 1
---

# SQP Settings and Status

## Settings

| Field | Default | Description |
|:--|:--|:--|
| `globalization_strategy` | `LINE_SEARCH_FILTER` | Full-step, L1 line-search, or filter line-search strategy. |
| `hessian_approximation` | `GAUSS_NEWTON` | Exact Lagrangian Hessian or Gauss-Newton approximation. |
| `eps_prim` | `1e-6` | Primal-step termination threshold. |
| `eps_dual` | `1e-4` | Dual-step termination threshold. |
| `max_iter` | `1000` | Maximum SQP iterations. |
| `line_search_max_iter` | `100` | Maximum line-search iterations per SQP step. |
| `min_alpha` | `1e-4` | Minimum accepted step size. |
| `regularize_hessian` | `false` | Enable Gershgorin-based Hessian regularization. |
| `verbose` | `false` | Print iteration progress and final status. |

Additional fields control line-search, filter, watchdog, and elastic-mode behavior. Defaults are defined by `laopt::sqp_settings_t<Scalar>` in `laopt/solvers/sqp_solver.hpp`.

## Status

| Value | Meaning |
|:--|:--|
| `SOLVED` | Convergence criteria were satisfied. |
| `MAX_ITER_REACHED` | The iteration limit was reached. |
| `INFEASIBLE` | The problem was reported infeasible. |
| `NON_CONVEX_QP` | A QP subproblem was reported non-convex. |
| `QP_SOLVER_ERROR` | The selected QP backend returned an unhandled error. |
| `UNSOLVED` | The solver has not produced a result. |
| `INVALID_SETTINGS` | One or more settings failed validation. |

`solver.info()` returns the status, SQP iteration count, and accumulated QP iteration count. The primal and dual solutions are available through `primal()`, `dual()`, and `dual_bounds()`.
