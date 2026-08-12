---
title: Examples
layout: default
nav_order: 7
---

# Examples

The repository includes a collection of optimal control problems:

| Example | Focus |
|:--|:--|
| [Double integrator](https://github.com/PREDICT-EPFL/laopt/tree/main/examples/double_integrator) | Small linear dynamics and transcription/solver combinations. |
| [Simple inverted pendulum](https://github.com/PREDICT-EPFL/laopt/tree/main/examples/inverted_pendulum_simple) | Compact `ControlProblemBase` model and Radau collocation. |
| [Inverted pendulum](https://github.com/PREDICT-EPFL/laopt/tree/main/examples/inverted_pendulum) | Multiple transcription and derivative configurations. |
| [Chain of masses](https://github.com/PREDICT-EPFL/laopt/tree/main/examples/chain_mass) | Larger structured optimal-control problem. |
| [Fixed-wing aircraft](https://github.com/PREDICT-EPFL/laopt/tree/main/examples/fixed_wing) | Nonlinear flight dynamics with YAML model data. |
| [Rocket](https://github.com/PREDICT-EPFL/laopt/tree/main/examples/rocket) | Free-final-time trajectory optimization. |

The exact build requirements differ because each executable selects its own solver and optional differentiation backend. See each example's `CMakeLists.txt` and the [Installation]({{ site.baseurl }}/getting_started/installation) guide.

