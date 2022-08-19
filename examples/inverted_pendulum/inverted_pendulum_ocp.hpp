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

    /* Override function implementations from base class ------------------------------ */
    template<typename T>
    void lagrange_term_impl(T &lagrange,
                            constref_t <state_t<T>> &x,
                            constref_t <input_t<T>> &u)
    {
        T non_control_cost;
        get_non_control_cost(non_control_cost, x);
        T control_cost;
        get_control_cost(control_cost, u);

        lagrange = non_control_cost + control_cost;
    }

    template<typename T>
    void mayer_term_impl(T &mayer,
                         constref_t <state_t<T>> &x)
    {
        T non_control_cost;
        get_non_control_cost(non_control_cost, x);

        mayer = mayer_multiplier * non_control_cost;
    }

    template<typename T>
    void dynamics_impl(ref_t <state_t<T>> x_dot,
                       constref_t <state_t<T>> &x,
                       constref_t <input_t<T>> &u)
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
        x_dot << theta_dot,
                (m * g * l * sin(theta) - b * theta_dot + torque) / (m * l * l);
    }

    template<typename T>
    void get_non_control_cost(T &non_control_cost,
                              constref_t <state_t<T>> &x)
    {
        non_control_cost = static_cast<T>(0);
        T angle_err = angle_ref - x(0);
        non_control_cost += W_angle_err * angle_err * angle_err;
    }
    template<typename T>
    void get_control_cost(T &control_cost,
                          constref_t <input_t<T>> &u)
    {
        control_cost = static_cast<T>(0);
        control_cost += u.dot(R * u);
    }

};

#endif //LAOPT_INVERTED_PENDULUM_OCP_HPP
