#ifndef LAOPT_MULTIPLE_SHOOTING_HPP
#define LAOPT_MULTIPLE_SHOOTING_HPP

// Advanced user (level 2)

#include <Eigen/Dense>
#include "laopt/laopt.hpp"
#include "constants.hpp"

namespace laopt_tools {

#define PRINT(x) \
//std::cout << __FUNCTION__ << ": " << x << std::endl // Comment this line in to activate PRINT function in the code

/*
 * Multiple Shooting
 * |     |     |     |    ...     |     |
 * 0     1     2     3    ...    N-1    N     Decision variable indices (initial condition + number of segments)
 * 0     1     2     3         N_segs-1       Segment indices of N_segs segments
 * */
template<typename Derived, typename ControlProblem, unsigned N_segs>
class MultipleShootingBase : public laopt::Differentiable<MultipleShootingBase<Derived, ControlProblem, N_segs>>
{
    friend laopt::Differentiable<MultipleShootingBase<Derived, ControlProblem, N_segs>>;

    template<typename, typename, typename, typename>
    friend class laopt::ProblemBase;

protected:
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
    std::array<variable_t<ControlProblem::NU>, N> U_var;
    variable_t<ControlProblem::NP> p_var;

    /* Dynamic constraints */
    struct DiscreteDynamics {};
    template<typename x_t, typename u_t, typename p_t, typename tf_t,
            typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NX>
    function_impl(DiscreteDynamics,
                  const Eigen::MatrixBase<x_t> &x,
                  const Eigen::MatrixBase<u_t> &u,
                  const Eigen::MatrixBase<p_t> &p,
                  const Eigen::MatrixBase<tf_t> &tf)
    {
        using state_t = typename x_t::PlainObject;
        state_t k1 = tf(0) * controlProblem.template dynamics_impl<scalar_t>(x, u, p);
        state_t k2 = tf(0) * controlProblem.template dynamics_impl<scalar_t>(x + h * 0.5 * k1, u, p);
        state_t k3 = tf(0) * controlProblem.template dynamics_impl<scalar_t>(x + h * 0.5 * k2, u, p);
        state_t k4 = tf(0) * controlProblem.template dynamics_impl<scalar_t>(x + h * k3, u, p);
        return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    /* Objective */
    struct StageCost {};
    template<typename x_t, typename u_t, typename p_t,
            typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(StageCost,
                  const Eigen::MatrixBase<x_t> &x,
                  const Eigen::MatrixBase<u_t> &u,
                  const Eigen::MatrixBase<p_t> &p)
    {
        return h * controlProblem.template lagrange_term_impl<scalar_t>(x, u, p);
    }

    struct MayerCost {};
    template<typename x_t, typename p_t, typename tf_t,
            typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost,
                  const Eigen::MatrixBase<x_t> &xf,
                  const Eigen::MatrixBase<p_t> &p,
                  const Eigen::MatrixBase<tf_t> &tf)
    {
        return controlProblem.template mayer_term_impl<scalar_t>(xf, p, tf(0));
    }

    template<typename OptProblem>
    void define_problem(OptProblem &optProblem)
    {
        /* Register variables */
        for (unsigned i = 0; i < N; i++)
        {
            optProblem.add_variable(X_var[i]); // TODO eno1: Loop through array in add_variable() -> Do not want that
            optProblem.add_variable(U_var[i]);
        }
        optProblem.add_variable(X_var[N]);
        static_cast<Derived*>(this)->register_tf_var(optProblem);
        optProblem.add_variable(p_var);

        /* Loop through grid points */
        for (unsigned i = 0; i < N; i++)
        {
            optProblem.add_obj(this->function(StageCost{}, X_var[i], U_var[i], p_var)); // TODO eno4: Allow expressions here
            optProblem.add_constr(X_var[i + 1] == this->function(DiscreteDynamics{}, X_var[i], U_var[i], p_var, static_cast<Derived*>(this)->get_tf_var()));
        }

        /* Last grid point */
        optProblem.add_obj(this->function(MayerCost{}, X_var[N], p_var, static_cast<Derived*>(this)->get_tf_var()));

        /* Box constraints */
        for (unsigned i = 0; i < N; i++)
        {
            optProblem.add_constr(controlProblem.x_lb <= X_var[i] <= controlProblem.x_ub);
            optProblem.add_constr(controlProblem.u_lb <= U_var[i] <= controlProblem.u_ub);
        }
        optProblem.add_constr(controlProblem.x_lb <= X_var[N] <= controlProblem.x_ub);

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= X_var[0] <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= X_var[N] <= controlProblem.xf_ub);
        static_cast<Derived*>(this)->add_tf_var_contr(optProblem);
        optProblem.add_constr(controlProblem.opt_params_lb.vector() <= p_var <= controlProblem.opt_params_ub.vector());
    }

public:
    explicit MultipleShootingBase(ControlProblem &ctrlProblem_) :
            controlProblem(ctrlProblem_)
    {
        /* Construct trajectory time grid on [0, 1] */
        for (unsigned i = 0; i <= N; i++) { T(i) = i * h; }
    }

    using scalar_t = typename ControlProblem::Scalar; // TODO: Change in laOPT to accept Scalar
    using State = typename ControlProblem::State;
    using Input = typename ControlProblem::Input;
    using Param = typename ControlProblem::Param;
    using TimeTrajectory = Eigen::Vector<Scalar, N + 1>;
    using StateTrajectory = Eigen::Matrix<Scalar, ControlProblem::NX, N + 1>;
    using InputTrajectory = Eigen::Matrix<Scalar, ControlProblem::NU, N>;

    /* Set functions */
    void set_X_guess(const State &x_guess)
    {
        for (unsigned i = 0; i < X_var.size(); i++) { X_var.at(i) << x_guess; }
    }
    void set_X_guess(const StateTrajectory &X_guess)
    {
        for (unsigned i = 0; i < X_var.size(); i++) { X_var.at(i) << X_guess.col(i); }
    }
    void set_U_guess(const Input &u_guess)
    {
        for (unsigned i = 0; i < U_var.size(); i++) { U_var.at(i) << u_guess; }
    }
    void set_U_guess(const InputTrajectory &U_guess)
    {
        for (unsigned i = 0; i < U_var.size(); i++) { U_var.at(i) << U_guess.col(i); }
    }
    void set_p_guess(const Param &p_guess) { p_var = p_guess; }

    /* Get functions */
    double get_tf_opt() const
    {
        return static_cast<const Derived*>(this)->get_tf_var()(0);
    }
    TimeTrajectory get_T_opt() const
    {
        return TimeTrajectory::Constant(controlProblem.t0) + (get_tf_opt() - controlProblem.t0) * T;
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
    Param get_p_opt() const { return Param(p_var); }
    typename ControlProblem::OptParam get_opt_params() const
    {
        typename ControlProblem::OptParam opt_param;
        opt_param.set_vector(get_p_opt());
        return opt_param;
    }

    Eigen::Vector<Scalar, ControlProblem::NX> get_x_at(const Scalar &t) const
    {
        const double tf = get_tf_opt();
        if (t == tf) { return X_var[N]; }
        else
        {
            /* Interpolate between discrete states */
            const Scalar T_eval = (t - controlProblem.t0) / (tf - controlProblem.t0);
            // on [0 ... 1]|traj;

            /* Find segment to sample from */
            const unsigned iL = std::floor(T_eval / h);
            const Scalar tau_eval = (T_eval - iL * h) / h;
            PRINT("i_lower: " << iL << ", tau_eval: " << tau_eval);
            const Eigen::Vector<Scalar, ControlProblem::NX> xL = X_var[iL];
            const Eigen::Vector<Scalar, ControlProblem::NX> xU = X_var[iL + 1];

            return xL + tau_eval * (xU - xL);
        }
    }
    Eigen::Vector<Scalar, ControlProblem::NU> get_u_at(const Scalar &t) const
    {
        const double tf = get_tf_opt();
        if (t == tf) { return U_var[N]; }
        else
        {
            const Scalar T_eval = (t - controlProblem.t0) / (tf - controlProblem.t0);
            return U_var[std::floor(T_eval / h)];
        }
    }

    Eigen::MatrixX<Scalar> get_TX_resampled(const Scalar &Ts_max) const
    {
        return resample_trajectory_linear(get_T_opt(), get_X_opt(), Ts_max);
    }
    Eigen::MatrixX<Scalar> get_TU_resampled(const Scalar &Ts_max) const
    {
        /* Duplicate last input to match time grid, then resample */
        Eigen::Matrix<Scalar, ControlProblem::NU, N + 1> U_opt;
        U_opt.template block<ControlProblem::NU, N>(0, 0) = get_U_opt();
        U_opt.col(N) = U_opt.col(N-1);
        return resample_trajectory_hold(get_T_opt(), U_opt, Ts_max);
    }

    /* Diagnosis */
    void print_diagnostics() const
    {
        std::cout << std::setprecision(4) << std::defaultfloat;
        std::cout << "Diagnostics: Multiple Shooting with N_segs = " << N_segs << "\n";
        controlProblem.print_diagnostics();
        const Eigen::VectorXd T_opt = get_T_opt();
        const Eigen::MatrixXd X_opt = get_X_opt();
        const Eigen::MatrixXd U_opt = get_U_opt();
        const Eigen::MatrixXd p_opt = get_p_opt();
        std::cout << "T_opt = [\n" << T_opt.transpose() << "];\n";
        std::cout << "X_opt = [\n" << X_opt << "];\n";
        std::cout << "U_opt = [\n" << U_opt << "];\n";
        std::cout << "p_opt = [" << p_opt.transpose() << "];\n";
    }

protected: /* Helpers for resampling */
    template<int DerivedNT1, int DerivedNT2, int DerivedNX>
    Eigen::Matrix<Scalar, DerivedNX + 1, -1> resample_trajectory_linear(const Eigen::Vector<Scalar, DerivedNT1> &T_opt,
                                                                        const Eigen::Matrix<Scalar, DerivedNX, DerivedNT2> &X_opt,
                                                                        Scalar Ts_max) const
    {
        static_assert(DerivedNT1 == DerivedNT2, "T and X must be of same length.");

        const Scalar dT = (T_opt(1) - T_opt(0));
        unsigned n_per_seg = std::floor(dT / Ts_max);
        if (n_per_seg * Ts_max < dT) { ++n_per_seg; };
        const unsigned n = N_segs * n_per_seg;
        PRINT("T_opt(1): " << T_opt(1) << ", n_per_seg: " << n_per_seg << ", n (total): " << n);

        Eigen::Matrix<Scalar, DerivedNX + 1, -1> TXn(DerivedNX + 1, n + 1);
        TXn.setZero();

        using namespace Eigen;
        for (unsigned iL = 0; iL < N; iL++)
        {
            const unsigned k_seg_start = iL * n_per_seg;
            PRINT("iL: " << iL << ", k_seg_start: " << k_seg_start);

            const Eigen::Vector<Scalar, DerivedNX> xL = X_opt.col(iL);
            const Eigen::Vector<Scalar, DerivedNX> xU = X_opt.col(iL + 1);

            for (unsigned j = 0; j < n_per_seg; j++)
            {
                const unsigned k = k_seg_start + j;
                const Scalar tau_eval = j * 1.0 / n_per_seg; // Time on [0 ... 1]
                PRINT("j: " << j << ", k: " << k << ", tau: " << tau_eval);

                TXn(0, k) = (iL * h + h * tau_eval);
                TXn(seqN(1, DerivedNX), k) << xL + tau_eval * (xU - xL);
            }

            /* In last segment, write last point */
            if (iL == N_segs - 1)
            {
                TXn(0, n) = get_tf_opt();
                TXn(seqN(1, DerivedNX), n) << X_opt.col(N);
            }

            /* Transform time by absolute horizon range (except  */
            TXn(0, seqN(k_seg_start, n_per_seg)) =
                    Eigen::MatrixX<Scalar>::Constant(1, n_per_seg, controlProblem.t0) +
                    (get_tf_opt() - controlProblem.t0) * TXn(0, seqN(k_seg_start, n_per_seg));
        }
        return TXn;
    }
    template<int DerivedNT1, int DerivedNT2, int DerivedNX>
    Eigen::Matrix<Scalar, DerivedNX + 1, -1> resample_trajectory_hold(const Eigen::Vector<Scalar, DerivedNT1> &T_opt,
                                                                      const Eigen::Matrix<Scalar, DerivedNX, DerivedNT2> &X_opt,
                                                                      Scalar Ts_max) const
    {
        static_assert(DerivedNT1 == DerivedNT2, "T and X must be of same length.");

        const Scalar dT = (T_opt(1) - T_opt(0));
        unsigned n_per_seg = std::floor(dT / Ts_max);
        if (n_per_seg * Ts_max < dT) { ++n_per_seg; };
        const unsigned n = N_segs * n_per_seg;
        PRINT("T_opt(1): " << T_opt(1) << ", n_per_seg: " << n_per_seg << ", n (total): " << n);

        Eigen::Matrix<Scalar, DerivedNX + 1, -1> TXn(DerivedNX + 1, n + 1);
        TXn.setZero();

        using namespace Eigen;
        for (unsigned i = 0; i < N; i++)
        {
            const unsigned k_seg_start = i * n_per_seg;
            PRINT("i: " << i << ", k_seg_start: " << k_seg_start);

            for (unsigned j = 0; j < n_per_seg; j++)
            {
                const unsigned k = k_seg_start + j;
                const Scalar tau_eval = j * 1.0 / n_per_seg; // Time on [0 ... 1]
                PRINT("j: " << j << ", k: " << k << ", tau: " << tau_eval);

                TXn(0, k) = (i * h + h * tau_eval);
                TXn(seqN(1, DerivedNX), k) << X_opt.col(i);
            }

            /* In last segment, write last point */
            if (i == N_segs - 1)
            {
                TXn(0, n) = get_tf_opt();
                TXn(seqN(1, DerivedNX), n) << X_opt.col(N);
            }

            /* Transform time by absolute horizon range (except  */
            TXn(0, seqN(k_seg_start, n_per_seg)) =
                    Eigen::MatrixX<Scalar>::Constant(1, n_per_seg, controlProblem.t0) +
                    (get_tf_opt() - controlProblem.t0) * TXn(0, seqN(k_seg_start, n_per_seg));
        }
        return TXn;
    }
};

template<typename ControlProblem, unsigned N_segs, int Options = FixedEndTime>
class MultipleShooting;

template<typename ControlProblem, unsigned N_segs>
class MultipleShooting<ControlProblem, N_segs, FixedEndTime> : public MultipleShootingBase<MultipleShooting<ControlProblem, N_segs, FixedEndTime>, ControlProblem, N_segs>
{
protected:
    using Base = MultipleShootingBase<MultipleShooting<ControlProblem, N_segs, FixedEndTime>, ControlProblem, N_segs>;
    friend Base;

    using Scalar = typename Base::Scalar;

    Eigen::Vector<Scalar, 1> tf_var;

    inline Eigen::Vector<Scalar, 1>& get_tf_var() { return tf_var; }
    inline const Eigen::Vector<Scalar, 1>& get_tf_var() const { return tf_var; }

    template<typename OptProblem>
    inline void register_tf_var(OptProblem &optProblem)
    {
        assert(this->controlProblem.tf_lb == this->controlProblem.tf_lb && "tf upper and lower bound have to be the same");
        tf_var(0) = this->controlProblem.tf_lb;
    }

    template<typename OptProblem>
    inline void add_tf_var_contr(OptProblem &optProblem) {}
public:
    using Base::Base;
};

template<typename ControlProblem, unsigned N_segs>
class MultipleShooting<ControlProblem, N_segs, FreeEndTime> : public MultipleShootingBase<MultipleShooting<ControlProblem, N_segs, FreeEndTime>, ControlProblem, N_segs>
{
protected:
    using Base = MultipleShootingBase<MultipleShooting<ControlProblem, N_segs, FreeEndTime>, ControlProblem, N_segs>;
    friend Base;

    using Scalar = typename Base::Scalar;
    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    variable_t<1> tf_var;

    inline variable_t<1>& get_tf_var() { return tf_var; }
    inline const variable_t<1>& get_tf_var() const { return tf_var; }

    template<typename OptProblem>
    inline void register_tf_var(OptProblem &optProblem)
    {
        optProblem.add_variable(tf_var);
    }

    template<typename OptProblem>
    inline void add_tf_var_contr(OptProblem &optProblem)
    {
        optProblem.add_constr(this->controlProblem.tf_lb <= tf_var <= this->controlProblem.tf_ub);
    }
public:
    using Base::Base;
};

} // namespace laopt_tools

#endif //LAOPT_MULTIPLE_SHOOTING_HPP
