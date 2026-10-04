---
title: Defining an Optimal Control Problem
layout: default
parent: Optimal Control
nav_order: 1
---

# Defining an Optimal Control Problem

To model an optimal control problem (OCP) with laOPT, a derived class from `laopt_tools::ControlProblemBase` is implemented by the user. The template arguments of `ControlProblemBase` fix the scalar type and the dimensions of states, inputs, optimized parameters, and path constraints.

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

### Discrete-Time Dynamics

Alternatively, the model can be specified directly in discrete time by adding the `DiscreteDynamics` option. The dynamics are then given as a one-step map $$x_{k+1} = f_d(x_k, u_k, p, t_0, t_f, \tau_k)$$ that the user implements in `discrete_dynamics_impl`, and (continuous) `dynamics_impl` is not used. Over the $$N$$ samples of the horizon, the problem becomes

$$
\min_{x_k,\,u_k,\,p}\;
\sum_{k=0}^{N-1} L\bigl(x_k,u_k,p,t_0,t_f,\tau_k\bigr)
  + M\bigl(x_N,p,t_0,t_f\bigr)
\quad\text{s.t.}\quad
x_{k+1} = f_d\bigl(x_k,u_k,p,t_0,t_f,\tau_k\bigr),
$$

with the remaining bounds and constraints unchanged. Three things differ from the continuous-time formulation:

- **Multiple shooting only.** `DiscreteDynamics` is supported by `MultipleShooting`. (`RadauCollocation` requires a continuous `dynamics_impl`).
- **Sample-wise Lagrange cost.** The Lagrange term is *not* integrated. `lagrange_term_impl` is evaluated at each sample $$k = 0,\dots,N-1$$ and the values are summed without any step-size weighting. In the continuous case, the same sum is weighted by the segment length $$h$$ (left Riemann sum). If the cost should reflect a physical time step, include it in `lagrange_term_impl`.
- **No free end time.** The step size of the discrete model is chosen by the user inside `discrete_dynamics_impl`, so $$t_f$$ cannot be a decision variable. `DiscreteDynamics` cannot be combined with `FreeEndTime`.

## `ControlProblemBase` API

### Template Parameters

```cpp
namespace laopt_tools {

template <typename Scalar, 
          int NX, int NU, int NP = 0,
          int NG = 0, int NG0 = 0, int NGF = 0, 
          int Options = laopt_tools::FixedEndTime>
class ControlProblemBase;

} // namespace laopt_tools
```

| Parameter | Meaning                                                                    |
|:----------|:---------------------------------------------------------------------------|
| `Scalar`  | Numerical scalar type used to store bounds and solutions (e.g., `double`). |
| `NX`      | Number of states.                                                          |
| `NU`      | Number of inputs.                                                          |
| `NP`      | Number of global optimized parameters.                                     |
| `NG`      | Number of path constraints.                                                |
| `NG0`     | Number of initial constraints.                                             |
| `NGF`     | Number of terminal constraints.                                            |
| `Options` | `FixedEndTime` or `FreeEndTime`; optionally combined with `DiscreteDynamics` (e.g., `FixedEndTime \| DiscreteDynamics`). `DiscreteDynamics` cannot be combined with `FreeEndTime`. |

The base class provides fixed-size aliases including `State`, `Input`, `Param`, `IneqBound`, `Ineq0Bound`, and `IneqfBound`. Their scalar-generic counterparts are `state_t<T>`, `input_t<T>`, `param_t<T>`, `ineq_constr_t<T>`, `ineq_constr0_t<T>`, and `ineq_constrf_t<T>`.

### Model Callbacks

Implement callbacks as public member functions of the derived model. The signatures below are the complete interface expected by the transcription methods:

```cpp
// Continuous-time dynamics x_dot = f(x, u, p, t0, tf, tau).
template <typename X, typename U, typename P, typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
state_t<Scalar> dynamics_impl(const Eigen::MatrixBase<X>& x, 
                              const Eigen::MatrixBase<U>& u,
                              const Eigen::MatrixBase<P>& p,
                              const Eigen::MatrixBase<T0>& t0, 
                              const Eigen::MatrixBase<TF>& tf, 
                              const Tau& tau);

// Discrete-time dynamics x+ = fd(x, u, p, t0, tf, tau). Required instead of dynamics_impl when the DiscreteDynamics option is set.
template <typename X, typename U, typename P, typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
state_t<Scalar> discrete_dynamics_impl(const Eigen::MatrixBase<X>& x, 
                                       const Eigen::MatrixBase<U>& u,
                                       const Eigen::MatrixBase<P>& p,
                                       const Eigen::MatrixBase<T0>& t0, 
                                       const Eigen::MatrixBase<TF>& tf, 
                                       const Tau& tau);

// Running cost L(x, u, p, t0, tf, tau).
template <typename X, typename U, typename P, typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
Scalar lagrange_term_impl(const Eigen::MatrixBase<X>& x, 
                          const Eigen::MatrixBase<U>& u, 
                          const Eigen::MatrixBase<P>& p,
                          const Eigen::MatrixBase<T0>& t0, 
                          const Eigen::MatrixBase<TF>& tf,
                          const Tau& tau);

// Terminal cost M(xf, p, t0, tf).
template <typename XF, typename P, typename T0, typename TF,
          typename Scalar = typename XF::Scalar>
Scalar mayer_term_impl(const Eigen::MatrixBase<XF>& xf, 
                       const Eigen::MatrixBase<P>& p,
                       const Eigen::MatrixBase<T0>& t0, 
                       const Eigen::MatrixBase<TF>& tf);

// Path constraints g(x, u, p, t0, tf, tau) <= 0.
template <typename X, typename U, typename P, typename T0, typename TF, typename Tau,
          typename Scalar = typename X::Scalar>
ineq_constr_t<Scalar> inequality_constraints_impl(const Eigen::MatrixBase<X>& x, 
                                                  const Eigen::MatrixBase<U>& u,
                                                  const Eigen::MatrixBase<P>& p,
                                                  const Eigen::MatrixBase<T0>& t0, 
                                                  const Eigen::MatrixBase<TF>& tf,
                                                  const Tau& tau);

// Initial constraints g0(x0, u0, p, t0) <= 0.
template <typename X, typename U, typename P, typename T0,    
          typename Scalar = typename X::Scalar>
ineq_constr0_t<Scalar> inequality_constraints0_impl(const Eigen::MatrixBase<X>& x0,
                                                    const Eigen::MatrixBase<U>& u0,
                                                    const Eigen::MatrixBase<P>& p,
                                                    const Eigen::MatrixBase<T0>& t0);

// Terminal constraints gf(xf, p, t0, tf) <= 0.
template <typename XF, typename P, typename T0, typename TF,
          typename Scalar = typename XF::Scalar>
ineq_constrf_t<Scalar> inequality_constraintsf_impl(const Eigen::MatrixBase<XF>& xf,
                                                    const Eigen::MatrixBase<P>& p,
                                                    const Eigen::MatrixBase<T0>& t0,
                                                    const Eigen::MatrixBase<TF>& tf);
```

`dynamics_impl` is required for continuous-time problems; `discrete_dynamics_impl` is required instead when the `DiscreteDynamics` option is set. The unused one never has to be implemented. In discrete-time problems, `lagrange_term_impl` is interpreted as a per-sample cost (see [above](#discrete-time-dynamics)). The running and terminal costs default to zero. A constraint callback is required when its corresponding dimension `NG`, `NG0`, or `NGF` is nonzero. Use the inherited `unused(...)` helper to mute compiler warning on unused function arguments.

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
class DoubleIntegrator : public laopt_tools::ControlProblemBase<
                                  /*scalar*/double, /*NX*/2, /*NU*/1, /*NP*/0, /*NG*/0>
{
public:
    template <typename X, typename U, typename P, typename T0, typename TF, typename Tau,
              typename Scalar = typename X::Scalar>
    state_t<Scalar> dynamics_impl(const Eigen::MatrixBase<X>& x,
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

A discrete-time model selects the `DiscreteDynamics` option and implements `discrete_dynamics_impl` instead of `dynamics_impl`. Here, an explicit Euler step with a user-chosen step size is used:

```cpp
class DiscreteDoubleIntegrator : public laopt_tools::ControlProblemBase<
                                  /*scalar*/double, /*NX*/2, /*NU*/1, /*NP*/0, /*NG*/0, /*NG0*/0, /*NGF*/0,
                                  /*Options*/ laopt_tools::FixedEndTime | laopt_tools::DiscreteDynamics>
{
public:
    const double h = 0.1; // Step size of the discrete model, chosen by the user

    template <typename X, typename U, typename P, typename T0, typename TF, typename Tau,
              typename Scalar = typename X::Scalar>
    state_t<Scalar> discrete_dynamics_impl(const Eigen::MatrixBase<X>& x,
                                           const Eigen::MatrixBase<U>& u,
                                           const Eigen::MatrixBase<P>& p,
                                           const Eigen::MatrixBase<T0>& t0,
                                           const Eigen::MatrixBase<TF>& tf,
                                           const Tau& tau)
    {
        unused(p, t0, tf, tau);

        state_t<Scalar> x_next;
        x_next << x(0) + h * x(1), x(1) + h * u(0);

        return x_next;
    }
};
```

Bounds and boundary conditions are stored on the model:

```cpp
Pendulum ocp;
ocp.u_lb << -3.0;
ocp.u_ub <<  3.0;
ocp.set_x0(DoubleIntegrator::State{3.14159, 0.0});
ocp.set_tf(1.5);
```

See the problem headers in the repository's [`examples`](https://github.com/PREDICT-EPFL/laopt/tree/main/examples) directory for complete models of OCPs.
