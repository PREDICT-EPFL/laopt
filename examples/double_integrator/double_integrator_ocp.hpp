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
    template<typename X, typename U, typename P, typename T = typename X::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<X>& x,
                         const Eigen::MatrixBase<U>& u,
                         const Eigen::MatrixBase<P>& p)
    {
        return (x_ref - x).dot(Q * (x_ref - x)) + u.dot(R * u);
    }

    template<typename Xf, typename P, typename Ttf, typename T = typename Xf::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<Xf>& xf,
                      const Eigen::MatrixBase<P>& p,
                      const Ttf &tf)
    {
        return (x_ref - xf).dot(Q * (x_ref - xf));
    }

    template<typename X, typename U, typename P, typename T = typename X::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<X>& x,
                             const Eigen::MatrixBase<U>& u,
                             const Eigen::MatrixBase<P>& p)
    {
        state_t<T> x_dot = A * x + B * u;
        return x_dot;
    }

};

#endif //LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
