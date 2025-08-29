#ifndef LAOPT_INVERTED_PENDULUM_OCP_HPP
#define LAOPT_INVERTED_PENDULUM_OCP_HPP

// End user (level 1)

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"

class InvertedPendulumOcp : public laopt_tools::ControlProblemBase</*Scalar*/ double,
        /*NX*/ 2, /*NU*/ 1, /*NP*/ 2,
        /*NG*/ 2, /*NG0*/ 2, /*NGF*/ 1,
        laopt_tools::FreeEndTime>
{
public:
    Scalar angle_ref{0};

    Scalar mayer_multiplier{10};

    Scalar W_angle_err{10};

    Eigen::Matrix<Scalar, NU, NU> R{{1}};

    Scalar w_tf{0};

    InvertedPendulumOcp()
    {
        // ref_offset, us
        p_lb << 0, 0;
        p_ub << 0, 0;
    }

    template<typename T>
    T get_non_control_cost(const Eigen::Ref<const state_t<T>> &x,
                           const Eigen::Ref<const param_t<T>> &p)
    {
        T non_control_cost = static_cast<T>(0);

        T ref_offset = p(0);
        T angle_ref_ = angle_ref + ref_offset;
        T angle_err = angle_ref_ - x(0);
        non_control_cost += W_angle_err * angle_err * angle_err;
        return non_control_cost;
    }
    template<typename T>
    T get_control_cost(const Eigen::Ref<const input_t<T>> &u,
                       const Eigen::Ref<const param_t<T>> &p)
    {
        T control_cost = static_cast<T>(0);

        input_t<T> us = p.template segment<1>(1);
        control_cost += (u + us).dot(R * (u + us));
        return control_cost;
    }

    /* Override function implementations from base class ------------------------------ */
    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const tau_t& tau)
    {
        return get_non_control_cost<T>(x, p) + get_control_cost<T>(u, p);
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        return mayer_multiplier * get_non_control_cost<T>(xf, p) + w_tf * (tf(0) - t0(0));
    }

    template<typename x_t, typename u_t, typename p_t,
            typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                             const Eigen::MatrixBase<u_t>& u,
                             const Eigen::MatrixBase<p_t>& p)
    {
        // Constants
        const double g = 9.81; // gravity constant [m/s^2]
        const double l = 0.5; // length of the rod [m]
        const double m = 0.15; // mass of the ball [kg]
        const double b = 0.1; // friction coefficient

        // Setup states & controls
        T theta = x(0);
        T theta_dot = x(1);

        // Controls
        T torque = u(0);

        // Dynamics
        state_t <T> x_dot;
        x_dot << theta_dot,
                (m * g * l * sin(theta) - b * theta_dot + torque) / (m * l * l);
        return x_dot;
    }

    template<typename x_t, typename u_t, typename p_t,
            typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr0_t<T> inequality_constraints0_impl(const Eigen::MatrixBase<x_t>& x0,
                                                   const Eigen::MatrixBase<u_t>& u0,
                                                   const Eigen::MatrixBase<p_t>& p)
    {
        ineq_constr0_t<T> initial_ineq_constr;
        initial_ineq_constr(0) = (-x0(0) + 0.2); // <= g0_ub
        initial_ineq_constr(1) = (-u0(0) - 2.8); // <= g0_ub
        return initial_ineq_constr;
    }

    template<typename x_t, typename u_t, typename p_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr_t<T> inequality_constraints_impl(const Eigen::MatrixBase<x_t>& x,
                                                 const Eigen::MatrixBase<u_t>& u,
                                                 const Eigen::MatrixBase<p_t>& p,
                                                 const tau_t& tau)
    {
        ineq_constr_t<T> ineq_constr;
        ineq_constr(0) = (-x(0) + 0.2); // <= g_ub
        ineq_constr(1) = (-u(0) - 2.8); // <= g_ub
        return ineq_constr;
    }

    template<typename xf_t, typename p_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    ineq_constrf_t<T> inequality_constraintsf_impl(const Eigen::MatrixBase<xf_t>& xf,
                                                   const Eigen::MatrixBase<p_t>& p)
    {
        ineq_constrf_t<T> final_ineq_constr;
        final_ineq_constr(0) = /* gf_lb <= */ (-xf(0) + 0.2); /* <= gf_ub */
        return final_ineq_constr;
    }
};

#endif //LAOPT_INVERTED_PENDULUM_OCP_HPP
