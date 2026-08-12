---
title: Transcription Methods
layout: default
parent: Optimal Control
nav_order: 2
---

# Transcription Methods

laOPT currently provides two direct transcription methods.

## Multiple Shooting

`MultipleShooting<ControlProblem, Segments>` introduces state variables at the shooting nodes and enforces continuity through integrated dynamics. The default integrator is explicit fourth-order Runge-Kutta (ERK4).

```cpp
using Transcription = laopt_tools::MultipleShooting<MyOcp, 40>;
```

For `Segments = N`, the discrete solution contains `N + 1` states and `N` inputs. States are reconstructed between shooting nodes with linear interpolation. Inputs use a zero-order hold and are constant over each shooting segment.

## Radau Collocation

`RadauCollocation<ControlProblem, Segments, Degree>` uses a multi-segment pseudospectral scheme with Legendre-Gauss-Radau points.

```cpp
using Transcription = laopt_tools::RadauCollocation<MyOcp, /*S*/ 10, /*D*/ 3>;
```

For `Segments = S` and `Degree = D`, the discrete solution contains `S * D + 1` state and input nodes. Adjacent segments share their boundary node. The nodes inside each segment are non-uniformly spaced, and the state and input are represented by degree-`D` Lagrange polynomials through the `D + 1` nodes of that segment.

## Constructing the Nonlinear Program

The transcription classes wrap the same workflow described in [Modeling Nonlinear Programs]({{ site.baseurl }}/concepts/modeling). A transcription derives from `laopt::Differentiable`, owns the generated `laopt::Variable` blocks, and implements `define_problem` to add the discretized cost, dynamics, bounds, and path constraints. `laopt::Problem<Transcription>` then assembles that model into the solver-facing NLP in exactly the same way as a hand-written nonlinear program.

Both transcriptions therefore expose the same construction pattern:

```cpp
auto ocp = std::make_shared<MyOcp>();
auto transcription = std::make_shared<Transcription>(ocp);

using Problem = laopt::Problem<Transcription>;
auto problem = std::make_shared<Problem>(transcription);
```

The complete handoff is `ControlProblemBase` → transcription → `Problem<Transcription>` → solver. The solver and `Problem` retain the same transcription instance. After `solver.solve()`, the optimized decision variables are therefore available through `transcription`.

## Retrieve the Discrete Solution

Both transcription classes provide the following solution API:

```cpp
double get_tf_opt() const;
TimeTrajectory get_T_opt() const;
StateTrajectory get_X_opt() const;
InputTrajectory get_U_opt() const;
Param get_p_opt() const;

void print_diagnostics() const;
```

The getters return copies owned by the caller. Time is returned in physical units over $$[t_0,t_f]$$, not as normalized transcription time.

```cpp
solver.solve();

const double tf = transcription->get_tf_opt();
const auto T = transcription->get_T_opt();
const auto X = transcription->get_X_opt();
const auto U = transcription->get_U_opt();
const auto p = transcription->get_p_opt();
```

The trajectory layouts depend on the transcription:

| Method            | Number of Nodes                      | `T`               | `X`                | `U`                |
|:------------------|:-------------------------------------|:------------------|:-------------------|:-------------------|
| Multiple shooting | `N + 1` state nodes                  | `(N + 1) × 1`     | `NX × (N + 1)`     | `NU × N`           |
| Radau collocation | `S * D + 1` shared collocation nodes | `(S * D + 1) × 1` | `NX × (S * D + 1)` | `NU × (S * D + 1)` |

Each column of `X` corresponds to the same column of `T`. Radau input columns also correspond directly to `T`. For multiple shooting, `U.col(k)` is the input held over the interval from `T(k)` to `T(k + 1)`.

## Sample at One Time

Both methods expose point-sampling functions in physical time:

```cpp
Eigen::Vector<Scalar, ControlProblem::NX> get_x_at(const Scalar& t) const;
Eigen::Vector<Scalar, ControlProblem::NU> get_u_at(const Scalar& t) const;
```

```cpp
const double t_query = 0.35;
const auto x = transcription->get_x_at(t_query);
const auto u = transcription->get_u_at(t_query);
```

| Method            | `get_x_at(t)`                                           | `get_u_at(t)`                                           |
|:------------------|:--------------------------------------------------------|:--------------------------------------------------------|
| Multiple shooting | Linear interpolation between adjacent state nodes.      | Zero-order hold of the segment input.                   |
| Radau collocation | Evaluates the segment's degree-`D` Lagrange polynomial. | Evaluates the segment's degree-`D` Lagrange polynomial. |

For Radau collocation, use point sampling for $$t_0 \leq t < t_f$$. Retrieve the terminal node with `X.rightCols(1)` or `U.rightCols(1)`. The trajectory resampling functions below include the terminal point automatically.

## Resample the Complete Trajectory

Use the batch resampling API when a uniformly ordered, denser trajectory is needed for plotting, simulation interfaces, or a controller:

```cpp
Eigen::MatrixX<Scalar> get_TX_resampled(const Scalar& Ts_max) const;
Eigen::MatrixX<Scalar> get_TU_resampled(const Scalar& Ts_max) const;
```

`Ts_max` is the maximum spacing between returned samples in physical time. Each transcription segment is divided into an integer number of equal intervals, so the actual sample period may be smaller than `Ts_max`.

```cpp
const double Ts_max = 0.01;

const Eigen::MatrixXd TX = transcription->get_TX_resampled(Ts_max);
const Eigen::MatrixXd TU = transcription->get_TU_resampled(Ts_max);

// Time occupies the first row; trajectory values occupy the remaining rows.
const auto T_sampled = TX.row(0);
const auto X_sampled = TX.bottomRows(MyOcp::NX);
const auto U_sampled = TU.bottomRows(MyOcp::NU);
```

The packed outputs have the layouts

$$
TX = \begin{bmatrix}T\\X\end{bmatrix}
    \in \mathbb{R}^{(N_X+1)\times N_s},
\qquad
TU = \begin{bmatrix}T\\U\end{bmatrix}
    \in \mathbb{R}^{(N_U+1)\times N_s}.
$$

Multiple shooting uses the same linear-state and held-input reconstruction as its point-sampling methods. Radau resampling locates each sample in its segment and evaluates the collocation polynomial there. It does not linearly interpolate between the nonuniform Radau nodes and does not reintegrate the dynamics.

{: .note }
> `get_T_opt()`, `get_X_opt()`, and `get_U_opt()` expose the transcription nodes. For Radau collocation, use `get_x_at`, `get_u_at`, or the resampling helpers whenever values between those nodes are needed.
