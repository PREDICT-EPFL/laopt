---
title: Derivative Overrides
layout: default
parent: Core Concepts
nav_order: 4
---

# Derivative Overrides

The function and derivative callbacks on this page are member functions of a model derived from `laopt::Differentiable`. laOPT differentiates the model's `function_impl` automatically by default. When a derivative has a simpler or more efficient closed form, add a matching derivative member function to the same model class. Overrides are selected at compile time and apply only to the tagged function with the matching argument list.

`laopt::Differentiable<Derived, ...>` uses the [Curiously Recurring Template Pattern (CRTP)](https://en.cppreference.com/cpp/language/crtp): the model derives from a base template instantiated with the model's own type. This lets `Differentiable` find the member callbacks implemented by `Derived` without virtual dispatch. They are extension points on the derived model, not free functions.

## Enable Tagged Dispatch

The second template parameter of `laopt::Differentiable` is a bitmask of differentiation options. Adding `laopt::TAGGED` enables callbacks whose first argument is a tag:

```cpp
struct Model : public laopt::Differentiable<Model, laopt::TAGGED>
{
    // ...
};
```

Without `laopt::TAGGED`, `laopt::Differentiable<Model>` uses the tagless interface by default:

| Default Tagless Interface | `laopt::TAGGED` Interface |
|:--|:--|
| `function_impl(x)` | `function_impl(Cost{}, x)` |
| `expression(x)` | `expression(Cost{}, x)` |
| `jacobian_impl(jacobian, alpha, x)` | `jacobian_impl(Cost{}, jacobian, alpha, x)` |

`laopt::TAGGED` tells laOPT to use the first argument for compile-time dispatch. The tag names the callback; it is not a numerical function argument and does not add a column to the Jacobian. This allows one `Differentiable` model to provide several functions with otherwise identical argument types.

Options can be combined with the bitwise OR operator. For example, `laopt::TAGGED | laopt::CASADI_ALL` enables tagged callbacks and makes CasADi the model-wide AD backend. A tag containing `UseEigen` can still override that backend for one function.

## Tag Dispatch

A tag is an empty C++ type used as a compile-time function name. It is typically declared inside the `laopt::Differentiable` model alongside its callbacks:

```cpp
struct Model : public laopt::Differentiable<Model, laopt::TAGGED>
{
    struct Cost {};
    struct Dynamics {};

    // The first argument selects the corresponding member callback.
    template <typename X>
    auto function_impl(Cost, const Eigen::MatrixBase<X>& x) noexcept;

    template <typename X, typename U>
    auto function_impl(Dynamics,
                       const Eigen::MatrixBase<X>& x,
                       const Eigen::MatrixBase<U>& u) noexcept;
};
```

In `define_problem`, `expression` captures the tag and the variable blocks passed to that function:

```cpp
problem.add_obj(expression(Cost{}, x));
problem.add_constr(expression(Dynamics{}, x, u) == 0.0);
```

There is no runtime name lookup or function registry. The C++ overload and argument types determine which implementation is called. The tag does not decide whether a function is an objective or constraint; `add_obj` and `add_constr` assign that role.

## Derivative Inputs and Layout

Derivative dimensions are determined by the arguments passed to `expression`, not by every argument accepted by `function_impl`. Only `laopt::Variable` arguments, including fixed-size blocks and indexed views of them, are differentiated. Other arguments are available to the function but are held constant.

For example, consider a function with `NY` outputs captured as

```cpp
this->expression(Dynamics{}, x, coefficients, u);
```

where `x` is a `laopt::Variable<double, NX>`, `u` is a `laopt::Variable<double, NU>`, and `coefficients` is an ordinary Eigen vector. The local differentiation vector and derivative layouts are

$$
z = \begin{bmatrix}x^{\mathsf T} & u^{\mathsf T}\end{bmatrix}^{\mathsf T},
\qquad n_z = N_X + N_U,
$$

| Callback Output | Local Size | Order |
|:--|:--|:--|
| `jacobian` | `NY` $$\times$$ `n_z` | Rows follow the function output; columns are `[x, u]`. |
| `gradient` | `n_z` | Entries are `[x, u]`. |
| `hessian` | `n_z` $$\times$$ `n_z` | Rows and columns are both `[x, u]`. |

Thus the Jacobian passed to `jacobian_impl` has the block structure

$$
J = \begin{bmatrix}\dfrac{\partial f}{\partial x} &
                    \dfrac{\partial f}{\partial u}\end{bmatrix}.
$$

The `coefficients` argument contributes no columns or Hessian rows. Variable arguments are concatenated in `expression` order, while elements within a variable or view retain their selected order. laOPT maps this local layout to the global decision-vector indices assigned by `add_variable`.

## Override a Jacobian

Use the same tag and argument order as the corresponding `function_impl`. This example provides the Jacobian of a linear dynamics function directly:

```cpp
struct DynamicsModel : public laopt::Differentiable<DynamicsModel, laopt::TAGGED>
{
    static constexpr int NX = 2;
    static constexpr int NU = 1;

    Eigen::Matrix<double, NX, NX> A;
    Eigen::Matrix<double, NX, NU> B;

    struct Dynamics {};

    template <typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Eigen::Vector<Scalar, NX> function_impl(
        Dynamics,
        const Eigen::MatrixBase<X>& x,
        const Eigen::MatrixBase<U>& u) noexcept
    {
        return A * x + B * u;
    }

    template <typename OutJacobian, typename Alpha, typename X, typename U>
    void jacobian_impl(
        Dynamics,
        OutJacobian& jacobian,
        const Alpha& alpha,
        const Eigen::MatrixBase<X>& x,
        const Eigen::MatrixBase<U>& u) noexcept
    {
        jacobian(Eigen::indexing::all,
                 Eigen::seqN(0, Eigen::fix<NX>)) = alpha * A;
        jacobian(Eigen::indexing::all,
                 Eigen::seqN(Eigen::fix<NX>, Eigen::fix<NU>)) = alpha * B;
    }
};
```

The Jacobian columns follow the argument order in `expression(Dynamics{}, x, u)`: first `x`, then `u`. laOPT supplies `alpha` when the expression is scaled, so the override must include it.

## Override a Gradient and Hessian

For a vector-valued function $$f(x)$$, laOPT requests derivatives of the weighted scalar $$w^\top f(x)$$. A scalar objective has a one-element weight vector.

```cpp
struct CostModel : public laopt::Differentiable<CostModel, laopt::TAGGED>
{
    Eigen::Matrix2d Q;
    Eigen::Vector2d q;

    struct Cost {};

    template <typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Scalar function_impl(Cost, const Eigen::MatrixBase<X>& x) noexcept
    {
        return 0.5 * x.dot(Q * x) + q.dot(x);
    }

    template <typename Weight, typename OutGradient, typename X>
    void gradient_impl(
        Cost,
        OutGradient& gradient,
        const Eigen::MatrixBase<Weight>& weight,
        const Eigen::MatrixBase<X>& x) noexcept
    {
        gradient += weight(0) * (Q * x + q);
    }

    template <typename Weight, typename OutHessian, typename X>
    void hessian_impl(
        Cost,
        OutHessian& hessian,
        const Eigen::MatrixBase<Weight>& weight,
        const Eigen::MatrixBase<X>& x) noexcept
    {
        hessian += weight(0) * Q;
    }
};
```

Derivative outputs may already contain contributions from other expressions. Gradient and Hessian overrides must therefore add their contribution with `+=`. If a derivative is identically zero, provide the matching override and leave the output unchanged.

## Fallback Behavior

Each derivative is selected independently:

| Provided by the Model | laOPT Behavior |
|:--|:--|
| No override | Differentiate `function_impl` automatically. |
| `jacobian_impl` | Use the custom Jacobian for that tag and argument list. |
| `gradient_impl` | Use the custom gradient of the weighted function. |
| `hessian_impl` | Use the custom weighted Hessian. |

If `gradient_impl` is absent, laOPT forms the gradient from the selected Jacobian. A custom Jacobian can therefore accelerate both Jacobian and gradient evaluation. The Hessian remains on its own dispatch path and continues to use automatic differentiation unless `hessian_impl` is provided.

## Select an AD Backend per Tag

When the consuming target enables and links CasADi support, the tag passed to `expression` can also select the automatic-differentiation backend for that function:

```cpp
struct MixedBackendModel
    : public laopt::Differentiable<MixedBackendModel, laopt::TAGGED>
{
    laopt::Variable<double, 2> x;

    struct Cost {
        using UseEigen = std::true_type;
    };

    struct Constraint {
        using UseCasadi = std::true_type;
    };

    template <typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Scalar function_impl(Cost, const Eigen::MatrixBase<X>& value) noexcept
    {
        return value.squaredNorm();
    }

    template <typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    Scalar function_impl(
        Constraint,
        const Eigen::MatrixBase<X>& value) noexcept
    {
        return value(0) * value(1) - 1.0;
    }

    template <typename Problem>
    void define_problem(laopt::OptProblem<Problem>& problem)
    {
        problem.add_variable(x);

        // Cost{} selects function_impl(Cost, ...) and Eigen AD.
        problem.add_obj(this->expression(Cost{}, x));

        // Constraint{} selects function_impl(Constraint, ...) and CasADi AD.
        problem.add_constr(this->expression(Constraint{}, x) == 0.0);
    }
};
```

The same tag travels with the captured expression. laOPT uses it first to select the matching `function_impl`, and then to select Eigen or CasADi when it generates that function's Jacobian and Hessian. This allows different functions in one model to use different backends.

`UseEigen` and `UseCasadi` override the model-wide backend for automatic differentiation. They have no effect when a matching custom derivative overload is present, because `jacobian_impl` or `hessian_impl` takes precedence. A tag may select either Eigen or CasADi, but not both.

{: .note }
> Keep `function_impl` and every derivative override consistent. laOPT trusts custom derivatives and does not compare them with automatic differentiation at runtime.
