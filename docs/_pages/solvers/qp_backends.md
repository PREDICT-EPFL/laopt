---
title: QP Backends
layout: default
parent: Solvers
nav_order: 3
---

# QP Backends

The native SQP implementation can be instantiated with one of several QP interfaces.

| Backend | laOPT type | Interface header |
|:--|:--|:--|
| PIQP | `laopt::PIQPSolver<Scalar>` | `laopt/solvers/piqp_interface.hpp` |
| OSQP | `laopt::OSQPSolver<Scalar>` | `laopt/solvers/osqp_interface.hpp` |
| ProxQP | `laopt::ProxQPSolver<Scalar>` | `laopt/solvers/proxqp_interface.hpp` |
| QPALM | `laopt::QPALMSolver<Scalar>` | `laopt/solvers/qpalm_interface.hpp` |
| qpSWIFT | `laopt::QPSwiftSolver<Scalar>` | `laopt/solvers/qpswift_interface.hpp` |
| HPIPM | `laopt::HPIPMSolver` | `laopt/solvers/hpipm_interface.hpp` |

Each interface translates laOPT's two-sided general constraints and box bounds into the format expected by the external solver.

{: .warning }
Including an interface header is not sufficient by itself. The consuming target must also find and link the corresponding external package. Backend versions and exported CMake target names are controlled by those projects.

HPIPM uses the block structure of multiple-shooting optimal control problems and has a specialized SQP integration.
