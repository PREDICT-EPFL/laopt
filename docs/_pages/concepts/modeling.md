---
title: Modeling NLPs
layout: default
parent: Core Concepts
nav_order: 2
---

# Modeling Nonlinear Programs

{% root_include _common/nlp_formulation.md %}

## Decision Variables

Declare fixed-size decision-variable blocks with `laopt::Variable<Scalar, Size>` and add them in the order in which they should appear in the global decision vector:

```cpp
laopt::Variable<double, 4> x;
laopt::Variable<double, 2> u;

problem.add_variable(x);
problem.add_variable(u);
```

## Tagged Differentiable Functions

A differentiable model derives from `laopt::Differentiable` and implements one or more `function_impl` overloads as member functions. Each empty tag type acts as a compile-time function name, allowing one model to define an objective and multiple constraint functions without runtime registration.

```cpp
struct Model : public laopt::Differentiable<Model, laopt::TAGGED>
{
    struct Cost {};

    template <typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Scalar function_impl(Cost, const Eigen::MatrixBase<X>& x) noexcept
    {
        return x.squaredNorm();
    }
};
```

The inherited `expression(Cost{}, x)` member selects the model's `Cost` overload and binds it to the variable block `x`. Whether the resulting expression is an objective or constraint is determined later by `add_obj` or `add_constr`.

The same tag also selects optional custom Jacobian, gradient, and Hessian implementations. See [Derivative Overrides]({{ site.baseurl }}/concepts/derivative_overrides) for the dispatch rules and complete examples.

## Differentiated Inputs

laOPT determines the differentiation variables from the arguments passed to `expression`. Only `laopt::Variable` arguments contribute derivative coordinates. Ordinary scalars and Eigen vectors are still passed to `function_impl`, but they are treated as constant parameters:

```cpp
laopt::Variable<double, 2> x;
Eigen::Vector2d coefficients;       // Constant with respect to this expression.
laopt::Variable<double, 1> u;

auto dynamics = expression(Dynamics{}, x, coefficients, u);
```

If `Dynamics` returns a vector with two elements, laOPT differentiates it with respect to

$$
z = \begin{bmatrix}x_0 & x_1 & u_0\end{bmatrix}^{\mathsf T}.
$$

The non-variable `coefficients` argument does not appear in $$z$$. The resulting local derivative sizes are

| Derivative | Size |
|:--|:--|
| Jacobian $$\partial f/\partial z$$ | $$2 \times 3$$ |
| Gradient of $$w^{\mathsf T}f$$ | $$3$$ |
| Hessian of $$w^{\mathsf T}f$$ | $$3 \times 3$$ |

Variable arguments are concatenated in the order in which they appear in `expression`: first every element of `x`, then every element of `u`. Fixed-size blocks and indexed views of a `laopt::Variable` are also differentiated, using the order of the selected elements. When the local derivatives are assembled into the NLP, each element is placed at the global index assigned by `add_variable`.

This same layout must be used by custom derivative callbacks. See [Derivative Inputs and Layout]({{ site.baseurl }}/concepts/derivative_overrides#derivative-inputs-and-layout) for the override API.

## Objectives and Constraints

`define_problem` receives an `OptProblem` evaluator. The same definition is replayed to determine dimensions, bounds, values, derivatives, and sparsity.

```cpp
problem.add_obj(1.0 * expression(Cost{}, x));
problem.add_constr(-1.0 <= x <= 1.0);
problem.add_constr(expression(Dynamics{}, x, u) == 0.0);
```

Objectives are weighted scalar expressions. Constraints may be equalities, one-sided inequalities, two-sided inequalities, or direct bounds on variable blocks and indexed variable selections.

## Construct the Solver-Facing NLP

The variables, function callbacks, and `define_problem` method are members of one model class:

```cpp
struct DirectNlp : laopt::Differentiable<DirectNlp, laopt::TAGGED>
{
    using scalar_t = double;

    struct Cost {};
    struct Equality {};

    laopt::Variable<double, 2> x;

    template <typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Scalar function_impl(Cost, const Eigen::MatrixBase<X>& value) noexcept
    {
        return (value(0) - 1.0) * (value(0) - 1.0)
             + (value(1) - 2.0) * (value(1) - 2.0);
    }

    template <typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Scalar function_impl(Equality, const Eigen::MatrixBase<X>& value) noexcept
    {
        return value(0) + value(1) - 1.0;
    }

    template <typename Problem>
    void define_problem(laopt::OptProblem<Problem>& problem)
    {
        problem.add_variable(x);
        problem.add_obj(expression(Cost{}, x));
        problem.add_constr(-2.0 <= x <= 2.0);
        problem.add_constr(expression(Equality{}, x) == 0.0);
    }
};
```

`DirectNlp` describes the model. `laopt::Problem<DirectNlp>` converts it into the NLP object accepted by a solver:

```cpp
auto model = std::make_shared<DirectNlp>();

// Solver-facing NLP: dimensions, bounds, derivatives, and sparsity.
using Nlp = laopt::Problem<DirectNlp>;
auto nlp = std::make_shared<Nlp>(model);

// SQP solves the NLP and uses PIQP for its quadratic subproblems.
using QPSolver = laopt::PIQPSolver<double>;
laopt::SQPSolver<Nlp, QPSolver> solver(nlp);

solver.settings().verbose = true;
solver.set_initial_primal(Eigen::Vector2d::Zero());
solver.solve();

std::cout << "solution: " << solver.primal().transpose() << '\n';
```

With verbose mode enabled, the program prints:

```text
----------------------------------------------------------
                        laOPT SQP
    (c) Roland Schwan, Johannes Waibel, Colin N. Jones
   Ecole Polytechnique Federale de Lausanne (EPFL) 2026
----------------------------------------------------------
variables n = 2
constraints m = 1
lagrangian hessian nnz = 2
constraints jacobian nnz = 2
globalization strategy: LINE_SEARCH_FILTER
iter    objective     primal_inf    comp_inf      stat_inf      alpha       qp_iter   elastic
   0    5.00000e+00   1.00000e+00   0.00000e+00   4.00000e+00   0.000e+00         0   0.00000
   1    2.00000e+00   1.27489e-07   2.54979e-07   1.90977e-07   1.000e+00         4  -0.00000
status: SOLVED
sqp iterations: 1
qp iterations: 4
solution: 3.0407e-07          1
```

The handoff is therefore `DirectNlp` → `Problem<DirectNlp>` → `SQPSolver`. Constructing `Problem` evaluates `define_problem`, determines the NLP dimensions, and prepares the derivative sparsity and evaluation tape. The solver receives the resulting `std::shared_ptr<Nlp>`.

The consuming target must also link the selected solver backend; see [Installation]({{ site.baseurl }}/getting_started/installation#link-a-solver).
