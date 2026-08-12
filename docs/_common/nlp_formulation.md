laOPT models nonlinear programs of the form

$$
\begin{aligned}
\min_{\xi} \quad & \phi(\xi) \\
\text{s.t.} \quad & c(\xi) = 0, \\
& \xi_{\mathrm{lb}} \leq \xi \leq \xi_{\mathrm{ub}}, \\
& h_{\mathrm{lb}} \leq h(\xi) \leq h_{\mathrm{ub}},
\end{aligned}
$$

where $$\xi$$ is the decision vector, $$\phi$$ is the scalar objective, $$c$$ contains equality constraints, and $$h$$ contains bounded inequality constraints. Derivatives and sparse matrix structures are generated from the user-defined model.

