---
title: Defining an Optimal Control Problem
layout: default
parent: Optimal Control
nav_order: 1
---

# Defining an Optimal Control Problem

An optimal-control model derives from `laopt_tools::ControlProblemBase`. Its template arguments fix the scalar type and the dimensions of states, inputs, optimized parameters, and path constraints.

## Mathematical Formulation

laOPT represents continuous-time OCPs over $$t \in [t_0,t_f]$$. The normalized time $$\tau=(t-t_0)/(t_f-t_0)$$ is also passed to time-varying model functions. In its most general free-final-time form, the problem is

$$
\begin{aligned}
\min_{x(\cdot),\,u(\cdot),\,p,\,t_f}\quad
& \int_{t_0}^{t_f} L\bigl(x(t),u(t),p,t_0,t_f,\tau\bigr)\,dt
  + M\bigl(x(t_f),p,t_0,t_f\bigr) \\
\text{s.t.}\quad
& \dot{x}(t) = f\bigl(x(t),u(t),p,t_0,t_f,\tau\bigr), \\
& x_{\mathrm{lb}} \leq x(t) \leq x_{\mathrm{ub}}, \\
& u_{\mathrm{lb}} \leq u(t) \leq u_{\mathrm{ub}}, \\
& p_{\mathrm{lb}} \leq p \leq p_{\mathrm{ub}}, \\
& g_{\mathrm{lb}} \leq g\bigl(x(t),u(t),p,t_0,t_f,\tau\bigr)
  \leq g_{\mathrm{ub}}, \\
& x_{0,\mathrm{lb}} \leq x(t_0) \leq x_{0,\mathrm{ub}}, \\
& x_{f,\mathrm{lb}} \leq x(t_f) \leq x_{f,\mathrm{ub}}, \\
& g_{0,\mathrm{lb}} \leq g_0\bigl(x(t_0),u(t_0),p,t_0\bigr)
  \leq g_{0,\mathrm{ub}}, \\
& g_{f,\mathrm{lb}} \leq g_f\bigl(x(t_f),p,t_0,t_f\bigr)
  \leq g_{f,\mathrm{ub}}, \\
& t_{f,\mathrm{lb}} \leq t_f \leq t_{f,\mathrm{ub}}.
\end{aligned}
$$

Here, $$x(t) \in \mathbb{R}^{N_X}$$ is the state, $$u(t) \in \mathbb{R}^{N_U}$$ is the input, and $$p \in \mathbb{R}^{N_P}$$ contains global optimized parameters. The functions $$f$$, $$L$$, and $$M$$ map to `dynamics_impl`, `lagrange_term_impl`, and `mayer_term_impl`. The constraints $$g$$, $$g_0$$, and $$g_f$$ map to `inequality_constraints_impl`, `inequality_constraints0_impl`, and `inequality_constraintsf_impl`. Setting equal lower and upper bounds fixes an initial state, terminal state, or final time. For a fixed-time problem, $$t_f$$ is a parameter rather than a decision variable.

## `ControlProblemBase` API

### Template Parameters

```cpp
namespace laopt_tools {

template <
    typename Scalar,
    int NX,
    int NU,
    int NP = 0,
    int NG = 0,
    int NG0 = 0,
    int NGF = 0,
    int Options = laopt_tools::FixedEndTime
>
class ControlProblemBase;

} // namespace laopt_tools
```

| Parameter | Meaning |
|:--|:--|
| `Scalar` | Numerical scalar type used to store bounds and solutions. |
| `NX` | Number of states. |
| `NU` | Number of inputs. |
| `NP` | Number of global optimized parameters. |
| `NG` | Number of path constraints. |
| `NG0` | Number of initial constraints. |
| `NGF` | Number of terminal constraints. |
| `Options` | `FixedEndTime` or `FreeEndTime`. |

The base class provides fixed-size aliases including `State`, `Input`, `Param`, `IneqBound`, `Ineq0Bound`, and `IneqfBound`. Their scalar-generic counterparts are `state_t<T>`, `input_t<T>`, `param_t<T>`, `ineq_constr_t<T>`, `ineq_constr0_t<T>`, and `ineq_constrf_t<T>`.

### Model Callbacks

Implement callbacks as public member functions of the derived model. The signatures below are the complete interface expected by the transcription methods:

```cpp
// Continuous-time dynamics f(x, u, p, t0, tf, tau).
template <typename X, typename U, typename P,
          typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
state_t<Scalar> dynamics_impl(
    const Eigen::MatrixBase<X>& x,
    const Eigen::MatrixBase<U>& u,
    const Eigen::MatrixBase<P>& p,
    const Eigen::MatrixBase<T0>& t0,
    const Eigen::MatrixBase<TF>& tf,
    const Tau& tau);

// Running cost L(x, u, p, t0, tf, tau).
template <typename X, typename U, typename P,
          typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
Scalar lagrange_term_impl(
    const Eigen::MatrixBase<X>& x,
    const Eigen::MatrixBase<U>& u,
    const Eigen::MatrixBase<P>& p,
    const Eigen::MatrixBase<T0>& t0,
    const Eigen::MatrixBase<TF>& tf,
    const Tau& tau);

// Terminal cost M(xf, p, t0, tf).
template <typename XF, typename P, typename T0, typename TF,
          typename Scalar = typename XF::Scalar>
Scalar mayer_term_impl(
    const Eigen::MatrixBase<XF>& xf,
    const Eigen::MatrixBase<P>& p,
    const Eigen::MatrixBase<T0>& t0,
    const Eigen::MatrixBase<TF>& tf);

// Path constraints g(x, u, p, t0, tf, tau).
template <typename X, typename U, typename P,
          typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
ineq_constr_t<Scalar> inequality_constraints_impl(
    const Eigen::MatrixBase<X>& x,
    const Eigen::MatrixBase<U>& u,
    const Eigen::MatrixBase<P>& p,
    const Eigen::MatrixBase<T0>& t0,
    const Eigen::MatrixBase<TF>& tf,
    const Tau& tau);

// Initial constraints g0(x0, u0, p, t0).
template <typename X, typename U, typename P, typename T0,
          typename Scalar = typename X::Scalar>
ineq_constr0_t<Scalar> inequality_constraints0_impl(
    const Eigen::MatrixBase<X>& x0,
    const Eigen::MatrixBase<U>& u0,
    const Eigen::MatrixBase<P>& p,
    const Eigen::MatrixBase<T0>& t0);

// Terminal constraints gf(xf, p, t0, tf).
template <typename XF, typename P, typename T0, typename TF,
          typename Scalar = typename XF::Scalar>
ineq_constrf_t<Scalar> inequality_constraintsf_impl(
    const Eigen::MatrixBase<XF>& xf,
    const Eigen::MatrixBase<P>& p,
    const Eigen::MatrixBase<T0>& t0,
    const Eigen::MatrixBase<TF>& tf);
```

`dynamics_impl` is required. The running and terminal costs default to zero. A constraint callback is required when its corresponding dimension `NG`, `NG0`, or `NGF` is nonzero. Use the inherited `unused(...)` helper for callback arguments that a model does not need.

### Bounds and Configuration

The model exposes its bounds as fixed-size Eigen vectors:

```cpp
Scalar t0;

State x_lb, x_ub;
Input u_lb, u_ub;
Param p_lb, p_ub;

State x0_lb, x0_ub;
State xf_lb, xf_ub;
Scalar tf_lb, tf_ub;

IneqBound  g_lb,  g_ub;
Ineq0Bound g0_lb, g0_ub;
IneqfBound gf_lb, gf_ub;

void set_x0(const State& x0);
void set_xf(const State& xf);
void set_tf(const Scalar& tf);

template <typename... Ts>
constexpr void unused(Ts&&...) noexcept;

void print_problem_dimension() const;
void print_diagnostics() const;
```

State, input, and boundary-state bounds are unbounded unless configured. The initial time defaults to zero, and the final time defaults to one. Set `p_lb` and `p_ub` whenever `NP` is nonzero. Inequality constraints default to $$-\infty \leq g \leq 0$$. The convenience setters fix a quantity by setting its lower and upper bounds to the same value.

## Example

```cpp
class Pendulum : public laopt_tools::ControlProblemBase<
    double,  // scalar
    2,       // states
    1,       // inputs
    0,       // optimized parameters
    0        // path constraints
>
{
public:
    template <typename X, typename U, typename P,
              typename T0, typename TF, typename Tau,
              typename Scalar = typename X::Scalar>
    state_t<Scalar> dynamics_impl(
        const Eigen::MatrixBase<X>& x,
        const Eigen::MatrixBase<U>& u,
        const Eigen::MatrixBase<P>& p,
        const Eigen::MatrixBase<T0>& t0,
        const Eigen::MatrixBase<TF>& tf,
        const Tau& tau)
    {
        unused(p, t0, tf, tau);
        state_t<Scalar> x_dot;
        x_dot << x(1), u(0);
        return x_dot;
    }
};
```

Bounds and boundary conditions are stored on the model:

```cpp
Pendulum ocp;
ocp.u_lb << -3.0;
ocp.u_ub <<  3.0;
ocp.set_x0(Pendulum::State{3.14159, 0.0});
ocp.set_tf(1.5);
```

See the problem headers in the repository's [`examples`](https://github.com/PREDICT-EPFL/laopt/tree/main/examples) directory for complete models.
