#ifndef SRC_FIXEDWINGOCPEIGEN_HPP
#define SRC_FIXEDWINGOCPEIGEN_HPP

// End user (level 1)

#include <iostream>

#include "laopt/tools/control_problem_base.hpp"
#include "FixedWingOcpSettings.hpp"
#include "fixed_wing_model/FixedWingDynamicsEigen.hpp"

namespace fixed_wing_ocp {

using FixedWing = flight_model::eigen_model::fixed_wing::FixedWingDynamics;

class FixedWingFlightOCP :
        public laopt_tools::ControlProblemBase</*Scalar*/ double, /*NX*/ FixedWing::nx + 0, /*NU*/FixedWing::nu + 0>
{
public:
    ~FixedWingFlightOCP() = default;

    using Model = FixedWing;
    Model model;
    Model::DynamicParams dyn_params = Model::DynamicParams::Zero();

    Model::SystemMat A;     // Linearized dynamics
    Model::ControlMat B;    // Linearized dynamics
    Model::State x_trim;    // Linearized dynamics
    Model::Control u_trim;  // Linearized dynamics

    ControlObjectives objectives{};

    double mayer_multiplier{10};

    double Va_ref{11};

    double W_Va_err{4};

    Eigen::DiagonalMatrix<double, Model::nu, Model::nu> R;

    FixedWingFlightOCP()
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
                  << "mayer_mulitplier: " << mayer_multiplier << "\n"
                  << "W_Va_err: " << W_Va_err << "\n"
                  << "R diag: " << R.diagonal().transpose() << "\n"
                  << "\n";
    }

    template<typename T>
    T get_non_control_cost(const Eigen::Ref<const state_t<T>> &x)
    {
        T non_control_cost(0);
        if (objectives[TrackVa])
        {
            T Va_err = Va_ref - x(0);
            non_control_cost += W_Va_err * Va_err * Va_err;
        }
        return non_control_cost;
    }
    template<typename T>
    T get_control_cost(const Eigen::Ref<const input_t<T>> &u)
    {
        T control_cost(0);
        if (objectives[MinimizeControl])
        {
            control_cost += u.dot(R * u);
        }
        return control_cost;
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const tau_t& tau)
    {
        return get_non_control_cost<T>(x) + get_control_cost<T>(u);
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        T mayer = mayer_multiplier * get_non_control_cost<T>(xf);
//        mayer = mayer_multiplier * non_control_cost + p(0); // Time-optimal ocp

        // LQR terminal weight
//        Model::SystemMat Q_N;
//        Q_N << 0.5519, -0.8989, -0.0082, -1.1563,
//                -0.8989, 5.0973, 0.0408, 5.9988,
//                -0.0082, 0.0408, 0.0084, 0.0554,
//                -1.1563, 5.9988, 0.0554, 7.6771;
//        mayer = x.transpose() * Q_N.template cast<T>() * x;
        return mayer;
    }

    template<typename x_t, typename u_t, typename p_t,
            typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                             const Eigen::MatrixBase<u_t>& u,
                             const Eigen::MatrixBase<p_t>& p)
    {
//         Linear(ized) dynamics
//        xdot = A.template cast<T>() * (x - x_trim.template cast<T>()) +
//               B.template cast<T>() * (u - u_trim.template cast<T>());

        // Nonlinear dynamics
        Model::output_t <T> y;
        state_t<T> xdot;
        model.dynamics<T>(xdot, x, u, dyn_params, y);
        return xdot;
    }
};

} // namespace fixed_wing_ocp

#endif //SRC_FIXEDWINGOCPEIGEN_HPP
