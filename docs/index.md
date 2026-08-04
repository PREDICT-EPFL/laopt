---
title: Home
layout: default
nav_order: 1
---

# laOPT

laOPT is a native C++ toolbox for high-performance nonlinear optimization and optimal control. It combines an Eigen-based modeling interface, automatic differentiation, sparse derivative assembly, direct optimal-control transcription, and interfaces to established numerical solvers.

{% root_include _common/nlp_formulation.md %}

For optimal control applications, laOPT provides a continuous-time problem interface together with multiple shooting and Legendre-Gauss-Radau collocation.

## Features

- Header-only C++17 core built on Eigen.
- Vector-valued objectives, dynamics, bounds, and constraints.
- Automatic gradients, Jacobians, and Hessians using Eigen, with optional CasADi support.
- Precomputed sparsity patterns and evaluation tapes for repeated sparse evaluations.
- Native SQP with interchangeable QP solvers, plus an IPOPT interface.
- Multiple shooting and Radau collocation transcriptions.
- BSD 2-Clause license.

## Where to start

- New users should begin with [Installation]({{ site.baseurl }}/getting_started/installation) and [First Optimal Control Problem]({{ site.baseurl }}/getting_started/first_problem).
- For trajectory optimization, continue with [Optimal control problems]({{ site.baseurl }}/optimal_control/formulating_ocp).
- To select a numerical backend, see [Solvers]({{ site.baseurl }}/solvers).
