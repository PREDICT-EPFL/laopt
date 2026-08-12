---
title: Differentiation and Sparsity
layout: default
parent: Core Concepts
nav_order: 3
---

# Differentiation and Sparsity

laOPT evaluates the same model in several modes to obtain function values and first- and second-order derivatives.

## Derivative backends

Eigen-based automatic differentiation is the default. A model can request CasADi-generated Jacobians, Hessians, or both through the differentiation options when laOPT and the consuming target are compiled with CasADi support. User-provided derivative implementations can override automatic differentiation for selected functions.

## Sparse setup

Sparse problems are prepared in three phases:

1. `generate_sparsity` evaluates the model with sparsity-aware scalar and matrix types to determine nonzero structure.
2. `generate_tape` records how derivative entries map into the sparse matrices expected by a solver.
3. `laopt::Problem` replays that tape during numerical evaluations without rediscovering the mapping.

The convenience constructor performs the setup automatically:

```cpp
auto problem = std::make_shared<laopt::Problem<MyModel>>(model);
```

Setup can also be made explicit and reused:

```cpp
auto sparsity = laopt::generate_sparsity(model);
auto tape = laopt::generate_tape(model, sparsity);
auto problem = std::make_shared<laopt::Problem<MyModel>>(model, tape);
```

This explicit form is useful when inspecting structure.

For functions with known analytical derivatives, see [Derivative Overrides]({{ site.baseurl }}/concepts/derivative_overrides).
