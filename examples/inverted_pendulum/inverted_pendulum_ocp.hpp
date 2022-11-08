#ifndef LAOPT_INVERTED_PENDULUM_OCP_HPP
#define LAOPT_INVERTED_PENDULUM_OCP_HPP

// End user (level 1)

#include <limits>
#include <Eigen/Dense>

#include "ControlProblemBase.hpp"
#include "laopt/laopt.hpp"


class InvertedPendulumOcp : public ControlProblemBase</*Scalar*/ double, /*NX*/ 2, /*NU*/ 1, /*NP*/ 2>
{
public:
    struct OptParam : ControlProblemBase<Scalar, NX, NU, NP>::OptParam
    {
        VecRef<1> ref_offset = get_parameter<1>(0);
        VecRef<1> us = get_parameter<1>(1);
    };
    OptParam opt_params_lb, opt_params_ub;

    Scalar angle_ref{0};

    Scalar mayer_multiplier{10};

    Scalar W_angle_err{10};

    Eigen::Matrix<Scalar, NU, NU> R{{1}};

    Scalar w_tf{0};

    InvertedPendulumOcp()
    {
        // ref_offset
        opt_params_lb.ref_offset << 0;
        opt_params_ub.ref_offset << 0;

        // us
        opt_params_lb.us << 0;
        opt_params_ub.us << 0;
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
    template<typename T> // T is scalar type
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u,
                         const Eigen::Ref<const param_t<T>> &p)
    {
        return get_non_control_cost(x, p) + get_control_cost(u, p);
    }

    template<typename T> // T is scalar type
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &xf,
                      const Eigen::Ref<const param_t<T>> &p,
                      const T &tf)
    {
        return mayer_multiplier * get_non_control_cost(xf, p) +
               w_tf * tf;
    }

    template<typename T> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u,
                             const Eigen::Ref<const param_t<T>> &p)
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
};

#endif //LAOPT_INVERTED_PENDULUM_OCP_HPP
