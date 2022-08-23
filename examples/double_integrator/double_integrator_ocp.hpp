#ifndef LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
#define LAOPT_DOUBLE_INTEGRATOR_OCP_HPP

// End user (level 1)

#include <limits>
#include <Eigen/Dense>

#include "ControlProblemBase.hpp"
#include "laopt/laopt.hpp"

class DoubleIntegratorOcp : public ControlProblemBase</*Scalar*/ double, /*NX*/ 2, /*NU*/ 1>
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
    template<typename T>
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u)
    {
        return (x_ref - x).dot(Q * (x_ref - x)) + u.dot(R * u);
    }

    template<typename T>
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &x)
    {
        return (x_ref - x).dot(Q * (x_ref - x));
    }

    template<typename T>
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u)
    {
        state_t<T> x_dot = A * x + B * u;
        return tf * x_dot;
    }

};

#endif //LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
