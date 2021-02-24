Possible polympc structure

##### NLP

- specific NLP solvers
  - IPOpt
  - gurobi
  - Peter's SQP
  - ALADIN?
  - etc

##### Solver stubs

- interface to a particular NLP / QP / etc solver
- takes NLP description and specialized it to the specific solver

##### NLP module

- takes a set of functions $f$, $g$ and $g_e$  
  $$
  \begin{align}
  \min &\ f \\
  \text{s.t.} &
  \begin{aligned}[t]
  l \le g(x) &\le u\\
  g_e(x) &= 0
  \end{aligned}
  \end{align}
  $$
  the autodiff module should be able to realize if sections of $g$ and $g_e$ are linear and/or constant.
  
- is able to evaluate for each function $f$

  - $f(x)$
  - $\nabla f(x)$ or $J_f(x)$
  - $J_f(x) y$
  - $J_f(x)^T y$
  - sparsity structure of $J_f$

##### High-level NLP

More abstract NLP modeling

- takes a set of functions. For example $f_1(x_1)$, $f_2(x_1,x_2)$ and provides $f(x_1,x_2) = f(x_1, x_2)$ to the NLP module

##### Transcription module

Takes a continuous-time OCP formulation, and returns an NLP.

Two variants

- Multiple shooting
- Collocation





# FLow

- QP solver interface
- Think on linear model $\rightarrow$ QP solver...
- NLP + autodiff $\rightarrow$ QP solver
- SQP solver
- NLP + autodiff $\rightarrow$ SQP solver



# Linear Model $\rightarrow$ QP Solver

Simple problem example
$$
\begin{align}
\min\ & \sum l(C x_i - y_{ref}, u_i - u_{ref}) + V_f(C x_n - y_{ref})\\
\text{s.t.}\ &
\begin{aligned}[t]
x_{i+1} &= Ax_i + Bu_i + c\\
Dx_i + Eu_i &\le f
\end{aligned}
\end{align}
$$
where $D$ and $E$ are sparse, and $l$ and $V_f$ are polynomial.



Good notation:
```c++
for (int i=0; i<N; i++)
	x(:,i+1) == A * x(:,i) + B * u(:,i)
```

This requires that $x$ is an object of some sort... or does it?

