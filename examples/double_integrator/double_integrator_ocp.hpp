#ifndef LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
#define LAOPT_DOUBLE_INTEGRATOR_OCP_HPP

// End user (level 1)

#include <limits>
#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/ControlProblemBase.hpp"

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
    template<typename T> // T is scalar type
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u,
                         const Eigen::Ref<const param_t<T>> &p)
    {
        return (x_ref - x).dot(Q * (x_ref - x)) + u.dot(R * u);
    }

    template<typename T> // T is scalar type
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &xf,
                      const Eigen::Ref<const param_t<T>> &p,
                      const T &tf)
    {
        return (x_ref - xf).dot(Q * (x_ref - xf));
    }

    template<typename T> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u,
                             const Eigen::Ref<const param_t<T>> &p)
    {
        state_t<T> x_dot = A * x + B * u;
        return x_dot;
    }

};

#endif //LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
