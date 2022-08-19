#ifndef SRC_LONOCPEIGEN_HPP
#define SRC_LONOCPEIGEN_HPP

// End user (level 1)

#include "ControlProblemBase.hpp"
#include "LonOcpSettings.hpp"
#include "kite.hpp"

namespace lon_ocp {

using LonKite = kite_model::eigen_model::LonKiteDynamics;

class LonFlightOCP : public ControlProblemBase</*Scalar*/ double, /*NX*/ LonKite::nx + 0, /*NU*/LonKite::nu + 0>
{
public:
    ~LonFlightOCP() = default;

    using Model = LonKite;
    Model model;
    Model::SystemMat A;     // Linearized dynamics
    Model::ControlMat B;    // Linearized dynamics
    Model::State x_trim;    // Linearized dynamics
    Model::Control u_trim;  // Linearized dynamics

    ControlObjectives objectives{};

    double mayer_multiplier{10};

    double pitch_ref{-5.0 * M_PI / 180.0};
    double Va_ref{11};

    double W_pitch_err{2};
    double W_Va_err{4};

    Eigen::Matrix<double, Model::nu, Model::nu> R;

    LonFlightOCP()
    {
        x_trim = model.get_default_initial_state();
        u_trim.setZero();

        R.setZero();
        R.diagonal() << Eigen::Matrix<double, Model::nu, 1>::Ones() * 0.1;
    }

    void print_tuning_parameters()
    {
        std::cout << "\nOCP parameters set:\n"
                  << "Ref_Va: " << Va_ref << "\n"
                  << "Ref_pitch: " << pitch_ref * 180.0 / M_PI << " deg\n"
                  << "mayer_mulitplier: " << mayer_multiplier << "\n"
                  << "W_pitch_err: " << W_pitch_err << "\n"
                  << "W_Va_err: " << W_Va_err << "\n"
                  << "R diag: " << R.diagonal().transpose() << "\n"
                  << "\n";
    }

    template<typename T>
    void get_non_control_cost(T &non_control_cost,
                              constref_t <state_t<T>> &x)
    {
        non_control_cost = static_cast<T>(0);
        if (objectives[TrackAngle])
        {
            T pitch_err = pitch_ref - x(3);
            non_control_cost += W_pitch_err * pitch_err * pitch_err;
        }
        if (objectives[TrackVa])
        {
            T Va_err = Va_ref - x(0);
            non_control_cost += W_Va_err * Va_err * Va_err;
        }
    }
    template<typename T>
    void get_control_cost(T &control_cost,
                          constref_t <input_t<T>> &u)
    {
        control_cost = static_cast<T>(0);
        if (objectives[MinimizeControl])
        {
            control_cost += u.dot(R * u);
        }
    }


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
//        lagrange = exp(t) * non_control_cost + control_cost; // Exponentially decaying cost: later error is more exp.
    }

    template<typename T>
    void mayer_term_impl(T &mayer,
                         constref_t <state_t<T>> &x)
    {
        T non_control_cost;
        get_non_control_cost(non_control_cost, x);

        mayer = mayer_multiplier * non_control_cost;
//        mayer = mayer_multiplier * non_control_cost + p(0); // Time-optimal ocp

        // LQR terminal weight
//        Model::SystemMat Q_N;
//        Q_N << 0.5519, -0.8989, -0.0082, -1.1563,
//                -0.8989, 5.0973, 0.0408, 5.9988,
//                -0.0082, 0.0408, 0.0084, 0.0554,
//                -1.1563, 5.9988, 0.0554, 7.6771;
//        mayer = x.transpose() * Q_N.template cast<T>() * x;
    }

    template<typename T>
    void dynamics_impl(ref_t <state_t<T>> xdot,
                       constref_t <state_t<T>> &x,
                       constref_t <input_t<T>> &u)
    {
//         Linear(ized) dynamics
//        xdot = A.template cast<T>() * (x - x_trim.template cast<T>()) +
//               B.template cast<T>() * (u - u_trim.template cast<T>());

        // Nonlinear dynamics
        Model::param_t <T> p;
        Model::output_t <T> y;
        model.dynamics<T>(xdot, x, u, p, y);
//        xdot *= p(0); // Time-optimal ocp
    }
};

}

#endif //SRC_LONOCPEIGEN_HPP
