---
title: QP Settings and Status
layout: default
parent: API
nav_order: 2
---

# QP Settings and Status

All QP interfaces expose a common settings and result layer through `laopt::QPBase`.

## Settings

| Field | Default | Description |
|:--|:--|:--|
| `eps_rel` | `1e-6` | Relative termination tolerance. |
| `eps_abs` | `1e-6` | Absolute termination tolerance. |
| `max_iter` | `200` | Maximum QP iterations before backend-specific overrides. |
| `reuse_pattern` | `false` | Reuse dimensions and sparsity structure on subsequent solves. |
| `elastic_mode` | `false` | Add relaxation variables to general constraints. |
| `elastic_weight_l1` | `1.0` | Linear relaxation penalty. |
| `elastic_weight_l2` | `1.0` | Quadratic relaxation penalty. |
| `verbose` | `false` | Enable backend progress output where supported. |

## Status

The common QP status values are `SOLVED`, `MAX_ITER_REACHED`, `INFEASIBLE`, `NON_CONVEX`, `MIN_STEP`, `UNSOLVED`, and `INVALID_SETTINGS`. A backend may map its native statuses onto only a subset of these values.
