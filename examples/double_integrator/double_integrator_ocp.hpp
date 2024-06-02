#ifndef LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
#define LAOPT_DOUBLE_INTEGRATOR_OCP_HPP

// End user (level 1)

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"

class DoubleIntegratorOcp : public laopt_tools::ControlProblemBase</*Scalar*/ double, /*NX*/ 2, /*NU*/ 1>
{
public:
    /* Static parameters */
    Eigen::Matrix<Scalar, NX, NX> A{{0, 1},
                                    {0, 0}};
    Eigen::Matrix<Scalar, NX, NU> B{{0},
                                    {1}};

    State x_ref{3, 0};
    Eigen::Matrix<Scalar, NX, NX> Q{{10, 0},
                                    {0,  1}};
    Eigen::Matrix<Scalar, NU, NU> R{{0.01}};

    /* Override function implementations from base class ------------------------------ */
    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const Scalar& tau)
    {
        return (x_ref - x).dot(Q * (x_ref - x)) + u.dot(R * u);
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        return (x_ref - xf).dot(Q * (x_ref - xf));
    }

    template<typename x_t, typename u_t, typename p_t,
            typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                             const Eigen::MatrixBase<u_t>& u,
                             const Eigen::MatrixBase<p_t>& p)
    {
        state_t<T> x_dot = A * x + B * u;
        return x_dot;
    }

};

#endif //LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
