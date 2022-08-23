#ifndef LAOPT_INVERTED_PENDULUM_OCP_HPP
#define LAOPT_INVERTED_PENDULUM_OCP_HPP

// End user (level 1)

#include <limits>
#include <Eigen/Dense>

#include "ControlProblemBase.hpp"
#include "laopt/laopt.hpp"

class InvertedPendulumOcp : public ControlProblemBase</*Scalar*/ double, /*NX*/ 2, /*NU*/ 1>
{
public:
    Scalar angle_ref{0};

    Scalar mayer_multiplier{10};

    Scalar W_angle_err{10};

    Eigen::Matrix<Scalar, NU, NU> R{{1}};

    template<typename T>
    T get_non_control_cost(const Eigen::Ref<const state_t<T>> &x)
    {
        T non_control_cost = static_cast<T>(0);
        T angle_err = angle_ref - x(0);
        non_control_cost += W_angle_err * angle_err * angle_err;
        return non_control_cost;
    }
    template<typename T>
    T get_control_cost(const Eigen::Ref<const input_t<T>> &u)
    {
        T control_cost = static_cast<T>(0);
        control_cost += u.dot(R * u);
        return control_cost;
    }

    /* Override function implementations from base class ------------------------------ */
    template<typename T>
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u)
    {
        return get_non_control_cost(x) + get_control_cost(u);
    }

    template<typename T>
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &x)
    {
        return mayer_multiplier * get_non_control_cost(x);
//       return = w_tf * tf_var;
    }

    template<typename T>
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u)
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
        return tf * x_dot;
    }
};

#endif //LAOPT_INVERTED_PENDULUM_OCP_HPP
