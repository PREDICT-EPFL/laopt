#ifndef LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
#define LAOPT_DOUBLE_INTEGRATOR_OCP_HPP

#include <limits>
#include <Eigen/Dense>

#include "ControlProblemBase.hpp"
#include "lampc.hpp"

namespace laopt = lampc;

// End user (level 1)

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
    void lagrange_term_impl(T &lagrange,
                            constref_t <state_t<T>> &x,
                            constref_t <input_t<T>> &u)
    {
        lagrange = (x_ref - x).dot(Q * (x_ref - x)) + u.dot(R * u);
    }

    template<typename T>
    void mayer_term_impl(T &mayer,
                         constref_t <state_t<T>> &x)
    {
        mayer = (x_ref - x).dot(Q * (x_ref - x));
    }

    template<typename T>
    void dynamics_impl(ref_t <state_t<T>> x_dot,
                       constref_t <state_t<T>> &x,
                       constref_t <input_t<T>> &u)
    {
        x_dot = A * x + B * u;
    }

};

#endif //LAOPT_DOUBLE_INTEGRATOR_OCP_HPP
