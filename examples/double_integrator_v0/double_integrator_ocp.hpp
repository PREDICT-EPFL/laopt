#ifndef LAMPC_DOUBLE_INTEGRATOR_OCP_HPP
#define LAMPC_DOUBLE_INTEGRATOR_OCP_HPP

#include <limits>
#include <Eigen/Dense>

#include "ocp_base.hpp"

// Simple user level
class DoubleIntegratorOCP : public laopt::OCPBase<double, 2, 1>
{
public:
    Eigen::Matrix<scalar_t, NX, NX> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, NX, NU> B{{0},{1}};

    Eigen::Matrix<scalar_t, NX, NX> Q{{10,0},{0,1}};
    Eigen::Matrix<scalar_t, NU, NU> R{{0.01}};

    scalar_t tf = 2;
    state_t<scalar_t> ref{3, 0};

    state_t<scalar_t> lbx{-1, -2};
    control_t<scalar_t> lbu{-3};
    control_t<scalar_t> ubu{5};

    template<typename T>
    inline void dynamics_impl(Eigen::Ref<state_t<T>> x_dot,
                              const Eigen::Ref<const state_t<T>> &x,
                              const Eigen::Ref<const control_t<T>> &u) const noexcept
    {
        x_dot = A * x + B * u;
    }

    template<typename T>
    inline void lagrange_term_impl(T &lagrange,
                                   const Eigen::Ref<const state_t<T>> &x,
                                   const Eigen::Ref<const control_t<T>> &u) noexcept
    {
        lagrange = (ref - x).dot(Q * (ref - x)) + u.dot(R * u);
    }

    template<typename T>
    inline void mayer_term_impl(T &mayer, const Eigen::Ref<const state_t<T>> &x) noexcept
    {
        mayer = (ref - x).dot(Q * (ref - x));
    }
};

#endif //LAMPC_DOUBLE_INTEGRATOR_OCP_HPP
