#ifndef LAOPT_MULTIPLESHOOTING_HPP
#define LAOPT_MULTIPLESHOOTING_HPP

// Advanced user (level 2)

#include <Eigen/Dense>
#include "laopt/laopt.hpp"

#define PRINT(x) \
//std::cout << __FUNCTION__ << ": " << x << std::endl // Comment this line in to activate PRINT function in the code

namespace transcription {

/*
 * Multiple Shooting
 * |     |     |     |    ...     |     |
 * 0     1     2     3    ...    N-1    N     Decision variable indices (initial condition + number of segments)
 * 0     1     2     3         N_segs-1       Segment indices of N_segs segments
 * */
template<typename ControlProblem, unsigned N_segs>
class MultipleShooting : public laopt::Differentiable<MultipleShooting<ControlProblem, N_segs>>
{
protected: // TODO ino1
    /* Mirror scalar type (from ControlProblem), define variable template with scalar type */
    using Scalar = typename ControlProblem::Scalar;
    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    /* Instance of end user's ControlProblem */
    ControlProblem &controlProblem;

    /* Create discrete problem variables (define U_var with same length than X_var for easier data handling,
     * although last u will not be used */
    static const unsigned N = N_segs; // Last index of decision variables
    const double h{1.0 / N};
    Eigen::Vector<Scalar, N + 1> T; // Normalized time grid (0 ... 1)
    std::array<variable_t<ControlProblem::NX>, N + 1> X_var;
    std::array<variable_t<ControlProblem::NU>, N + 1> U_var;

//    template<int DerivedNT1, int DerivedNT2, int DerivedNX>
//    Eigen::Matrix<Scalar, DerivedNX + 1, -1> resample_trajectory(const Eigen::Vector<Scalar, DerivedNT1> &T_opt,
//                                                                 const Eigen::Matrix<Scalar, DerivedNX, DerivedNT2> &X_opt,
//                                                                 Scalar Ts_max) const
//    {
//        static_assert(DerivedNT1 == DerivedNT2, "T and X must be of same length.");
//
//        const Scalar dT = (T_opt(D_poly) - T_opt(0));
//        unsigned n_per_seg = std::floor(dT / Ts_max);
//        if (n_per_seg * Ts_max < dT) { ++n_per_seg; };
//        const unsigned n = N_segs * n_per_seg;
//        PRINT("T_opt(D_poly): " << T_opt(D_poly) << ", n_per_seg: " << n_per_seg << ", n (total): " << n);
//
//        Eigen::Matrix<Scalar, DerivedNX + 1, -1> TXn(DerivedNX + 1, n + 1);
//        TXn.setZero();
//
//        using namespace Eigen;
//        for (unsigned i_seg = 0; i_seg < N_segs; i_seg++)
//        {
//            const unsigned i_seg_start = i_seg * D_poly;
//            const unsigned k_seg_start = i_seg * n_per_seg;
//            PRINT("-------------------------- \n"
//                  "i_seg: " << i_seg << ", i_seg_start: " << i_seg_start << ", k_seg_start: " << k_seg_start);
//
//            const auto X_seg = X_opt(all, seqN(i_seg_start, D_poly + 1));
//            PRINT("X_seg:\n" << X_seg);
//
//            for (unsigned j = 0; j < n_per_seg; j++)
//            {
//                const unsigned k = k_seg_start + j;
//                const Scalar T_eval = j * 1.0 / n_per_seg; // Time on [0 ... 1]
//                const Scalar tau_eval = 2.0 * T_eval - 1;  // Time on [-1 ... 1]
//                PRINT("j: " << j << ", k: " << k << ", tau: " << tau_eval);
//
//                TXn(0, k) = (i_seg * h_seg + h_seg * T_eval);
//                TXn(seqN(1, DerivedNX), k) << interpolate<DerivedNX>(X_seg.template reshaped<ColMajor>(), tau_eval);
//            }
//
//            /* In last segment, write last point */
//            if (i_seg == N_segs - 1)
//            {
//                TXn(0, n) = controlProblem.tf;
//                /* Copy or extrapolate */
//                TXn(seqN(1, DerivedNX), n) << X_opt(all, last);
////                TXn(seqN(1, DerivedNX), n) << interpolate<DerivedNX>(X_seg.template reshaped<ColMajor>(), 1);
//            }
//
//            /* Transform time by absolute horizon range (except  */
//            TXn(0, seqN(k_seg_start, n_per_seg)) =
//                    Eigen::MatrixX<Scalar>::Constant(1, n_per_seg, controlProblem.t0) +
//                    (controlProblem.tf - controlProblem.t0) * TXn(0, seqN(k_seg_start, n_per_seg));
//            PRINT("\n" << TXn << "\n");
//        }
//        return TXn;
//    }

public: //protected: // TODO ino1 (would like to make this protected)
    /* Dynamic constraints */
    struct DiscreteDynamics {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NX>
    function_impl(DiscreteDynamics,
                  const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        using state_t = typename x_t::PlainObject;
        state_t k1 = controlProblem.template dynamics_impl<scalar_t>(x, u);
        state_t k2 = controlProblem.template dynamics_impl<scalar_t>(x + h * 0.5 * k1, u);
        state_t k3 = controlProblem.template dynamics_impl<scalar_t>(x + h * 0.5 * k2, u);
        state_t k4 = controlProblem.template dynamics_impl<scalar_t>(x + h * k3, u);
        return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    /* Objective */
    struct StageCost {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(StageCost,
                  const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        return h * controlProblem.template lagrange_term_impl<scalar_t>(x, u);
    }

    struct MayerCost {};
    template<typename x_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost,
                  const Eigen::MatrixBase<x_t> &x)
    {
        return controlProblem.template mayer_term_impl<scalar_t>(x);
    }

    template<typename OptProblem>
    void define_problem(OptProblem &optProblem)
    {
        /* Register variables */
        for (unsigned i = 0; i <= N; i++)
        {
            optProblem.add_variable(X_var[i]); // TODO eno1: Loop through array in add_variable() -> Do not want that
            optProblem.add_variable(U_var[i]);
        }

        /* Loop through grid points */
        for (unsigned i = 0; i < N; i++)
        {
            optProblem.add_obj(this->function(StageCost{}, X_var[i], U_var[i])); // TODO eno4: Allow expressions here
            optProblem.add_constr(X_var[i + 1] == this->function(DiscreteDynamics{}, X_var[i], U_var[i]));
        }

        /* Last grid point */
        optProblem.add_obj(this->function(MayerCost{}, X_var[N]));

        /* Box constraints */
        for (unsigned i = 0; i <= N; i++)
        {
            optProblem.add_constr(controlProblem.lbx <= X_var[i] <= controlProblem.ubx);
            optProblem.add_constr(controlProblem.lbu <= U_var[i] <= controlProblem.ubu);
        }

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= X_var[0] <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= X_var[N] <= controlProblem.xf_ub);

        /* Set last control equal second last */
        optProblem.add_constr(U_var[N] == U_var[N - 1]);
    }

public:
    explicit MultipleShooting(ControlProblem &ctrlProblem_) :
            controlProblem(ctrlProblem_)
    {
        /* Construct trajectory time grid on [0, 1] */
        for (unsigned i = 0; i <= N; i++) { T(i) = i * h; }
    }

    using scalar_t = typename ControlProblem::Scalar; // TODO: Change in laOPT to accept Scalar
    using TimeTrajectory = Eigen::Vector<Scalar, N + 1>;
    using StateTrajectory = Eigen::Matrix<Scalar, ControlProblem::NX, N + 1>;
    using InputTrajectory = Eigen::Matrix<Scalar, ControlProblem::NU, N + 1>;

    /* Set functions */
    template<int rows, typename Scalar = double>
    void set_X_guess(const Eigen::Matrix<Scalar, rows, 1> &x_guess)
    {
        for (unsigned i = 0; i < X_var.size(); i++) { X_var.at(i) << x_guess; }
    }
    template<int rows, int cols, typename Scalar = double>
    void set_X_guess(const Eigen::Matrix<Scalar, rows, cols> &X_guess)
    {
        for (unsigned i = 0; i < X_var.size(); i++) { X_var.at(i) << X_guess.col(i); }
    }

    /* Get functions */
    TimeTrajectory get_T_opt() const
    {
        return TimeTrajectory::Constant(controlProblem.t0) + (controlProblem.tf - controlProblem.t0) * T;
    }
    StateTrajectory get_X_opt() const
    {
        StateTrajectory X_opt;
        X_opt.setZero();
        for (unsigned i = 0; i < X_var.size(); i++) { X_opt.col(i) << X_var.at(i); }
        return X_opt;
    }
    InputTrajectory get_U_opt() const
    {
        InputTrajectory U_opt;
        U_opt.setZero();
        for (unsigned i = 0; i < U_var.size(); i++) { U_opt.col(i) << U_var.at(i); }
        return U_opt;
    }
//
//    Eigen::Vector<Scalar, ControlProblem::NX> get_x_at(const Scalar &t) const
//    {
//        const Scalar T_eval = (t - controlProblem.t0) / (controlProblem.tf - controlProblem.t0); // on [0 ... 1]|traj;
//
//        /* Find segment to sample from */
//        const unsigned i_seg = std::floor(T_eval / h_seg);
//        const unsigned i_seg_start = i_seg * D_poly;
//        PRINT("i_seg: " << i_seg << ", i_seg_start: " << i_seg_start);
//
//        const Eigen::Vector<Scalar, NX * (D_poly + 1)> X_seg = get_x<D_poly + 1>(XU_var, i_seg_start);
//        const Scalar t_eval = T_eval - i_seg * h_seg;     // Time in segment [0 ... 1]|seg
//        const Scalar tau_eval = 2.0 * t_eval / h_seg - 1; // Time on [-1 ... 1]|
//        PRINT("X_seg:\n" << X_seg);
//        PRINT("t_eval: " << t_eval);
//        PRINT("tau_eval: " << tau_eval);
//        return interpolate<NX>(X_seg, tau_eval);
//    }
//    Eigen::Vector<Scalar, ControlProblem::NU> get_u_at(const Scalar &t) const
//    {
//        const Scalar T_eval = t / (controlProblem.tf - controlProblem.t0) - controlProblem.t0; // on [0 ... 1];
//
//        /* Find segment to sample from */
//        const unsigned i_seg = std::floor(T_eval / h_seg);
//        const unsigned i_seg_start = i_seg * D_poly;
//
//        /* Sample from segment */
//        const Eigen::Vector<Scalar, NU * (D_poly + 1)> X_seg = get_u<D_poly + 1>(XU_var, i_seg_start);
//        const Scalar t_eval = T_eval - i_seg * h_seg;     // Time in segment [0 ... 1]|seg
//        const Scalar tau_eval = 2.0 * t_eval / h_seg - 1; // Time on [-1 ... 1]|
//        return interpolate<NU>(X_seg, tau_eval);
//    }
//
//    Eigen::MatrixX<Scalar> get_TX_resampled(const Scalar &Ts_max) const
//    {
//        return resample_trajectory(get_T_opt(), get_X_opt(), Ts_max);
//    }
//    Eigen::MatrixX<Scalar> get_TU_resampled(const Scalar &Ts_max) const
//    {
//        return resample_trajectory(get_T_opt(), get_U_opt(), Ts_max);
//    }
};

} // namespace transcription

#endif //LAOPT_MULTIPLESHOOTING_HPP