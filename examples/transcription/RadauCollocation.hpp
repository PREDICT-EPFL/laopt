#ifndef LAOPT_RADAUCOLLOCATION_HPP
#define LAOPT_RADAUCOLLOCATION_HPP

// Advanced user (level 2)

#include <Eigen/Dense>
#include "laopt/laopt.hpp"

#define PRINT(x) \
//std::cout << __FUNCTION__ << ": " << x << std::endl // Comment this line in to activate PRINT function in the code

namespace transcription {

/*
 * Radau Collocation
 * |||    |    |    |||    |    |    |||   ...    |||    |    |    |||
 *  0     1    2     3     4    5     6    ...    N-3   N-2  N-1    N       Decision variable indices (initial condition + number of segments)
 *  0     1    2   D_poly                                                   Node indices of D_poly+1 nodes within segment
 *  0                1                2         N_segs-1                    Segment indices of N_segs segments
 * */
template<typename ControlProblem, unsigned N_segs, unsigned D_poly>
class RadauCollocation : public laopt::Differentiable<RadauCollocation<ControlProblem, N_segs, D_poly>>
{
protected:
    /* Mirror scalar type (from ControlProblem), define variable template with scalar type */
    using Scalar = typename ControlProblem::Scalar;
    static constexpr unsigned NX = ControlProblem::NX;
    static constexpr unsigned NU = ControlProblem::NU;
    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    /* Instance of end user's ControlProblem */
    ControlProblem &controlProblem;

    /* Create discrete problem variables (define U_var with same length than X_var for easier data handling,
     * although last u will not be used */
    static constexpr unsigned N = D_poly * N_segs; // Last index of decision variables
    const double h_seg{1.0 / N_segs};
    Eigen::Vector<Scalar, N + 1> T; // Normalized time grid (0 ... 1)
//    std::array<variable_t<ControlProblem::NX>, N + 1> X_var;
//    std::array<variable_t<NU>, N + 1> U_var;
    variable_t<(N + 1) * NX> X_var;
    variable_t<(N + 1) * NU> U_var;

    template<unsigned mat_rows, typename X_t>
    auto get_col(const X_t &X, unsigned col_index)
    {
        /* Pretend X_t is a column-wise reshaped matrix */
        const unsigned i0 = col_index * mat_rows;
        return X.template segment<mat_rows>(i0);
    }

    /* Collocation basis and grid */
    using CollocationPoints = Eigen::Vector<Scalar, D_poly + 1>;
    CollocationPoints get_collocation_points() const
    {
        Eigen::Vector<Scalar, D_poly + 1> nodes;
        nodes << -1.0000, -0.5753, 0.1811, 0.8228, 1.0000;
        if (nodes(0) >= 0) { nodes = (nodes + CollocationPoints::Constant(1.0)) / 2.0; }
        return nodes;
    }
    Eigen::Vector<Scalar, D_poly + 1> Tau = get_collocation_points(); // collocation_points::get();
    /* Lagrange polynomials */
    Scalar L(unsigned j, Scalar tau_eval) const
    {
        /* First create indexing for more convenient looping afterwards */
        std::vector<unsigned> loop_range;
        for (unsigned i = 0; i < D_poly + 1; i++)
        {
            if (i != j) loop_range.push_back(i);
        }

        /* Loop */
        Scalar L = 1;
        for (const unsigned l: loop_range)
        {
//            L *= static_cast<Scalar>((tau_eval - Tau(l)) / (Tau(j) - Tau(l)));
            L *= (tau_eval - Tau(l)) / (Tau(j) - Tau(l));
        }
        return L;
    }
    Scalar dL(unsigned j, Scalar tau_eval) const
    {
        /* First create indexing for more convenient looping afterwards */
        std::vector<unsigned> loop_range;
        for (unsigned i = 0; i < D_poly + 1; i++)
        {
            if (i != j) loop_range.push_back(i);
        }

        Scalar dL = 0;
        for (const unsigned i: loop_range)
        {
            Scalar L = 1;
            for (const unsigned l: loop_range)
            {
//                if (l != i) { L *= static_cast<Scalar>((tau_eval - Tau(l)) / (Tau(j) - Tau(l))); }
                if (l != i) { L *= (tau_eval - Tau(l)) / (Tau(j) - Tau(l)); }
            }
//            dL += 1.0 / static_cast<Scalar>(Tau(j) - Tau(i)) * L;
            dL += 1.0 / (Tau(j) - Tau(i)) * L;
        }
        return dL;
    }

    using DiffMat = Eigen::Matrix<Scalar, D_poly, D_poly + 1>;
    DiffMat get_diff_mat() const
    {
        const auto tau_eval = Tau.template head<D_poly>();
        DiffMat D = DiffMat::Zero();
        for (unsigned i = 0; i < tau_eval.size(); i++)
        {
            for (unsigned j = 0; j < D_poly + 1; j++)
            {
                D(i, j) = dL(j, tau_eval(i));
            }
        }
        return D;
    }
    const DiffMat diff_mat = get_diff_mat();
    using IntMat = Eigen::Matrix<Scalar, D_poly, D_poly>;

public:
    struct StateAt {};
    template<typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(StateAt,
                  const Eigen::MatrixBase<X_t> &X_vec, unsigned k)
    {
        return get_col<NX>(X_vec, k);
    }
    struct InputAt {};
    template<typename U_t, typename scalar_t = typename Eigen::MatrixBase<U_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NU>
    function_impl(InputAt,
                  const Eigen::MatrixBase<U_t> &U_vec, unsigned k)
    {
        return get_col<NU>(U_vec, k);
    }

    struct DifferentialApproximationAt {};
    template<typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(DifferentialApproximationAt,
                  const Eigen::MatrixBase<X_t> &X_vec, unsigned seg_start, unsigned j_node)
    {
        /* Construct differential approximation at node j_node */
        Eigen::Vector<scalar_t, NX> x_apr; // NX x 1
        {
            for (unsigned l = 0; l <= D_poly; l++)
            {
                x_apr += get_col<NX>(X_vec, seg_start + l) * dL(l, Tau(j_node));
            }
        }
        return 2.0 / h_seg * x_apr;
    }

    /* Objective */
    struct StageCost {};
    template<typename X_t, typename U_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(StageCost,
                  const Eigen::MatrixBase<X_t> &X_vec, const Eigen::MatrixBase<U_t> &U_vec, unsigned k)
    {
        return controlProblem.template lagrange_term_impl<scalar_t>(get_col<NX>(X_vec, k), get_col<NU>(U_vec, k));
    }

    struct MayerCost {};
    template<typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost,
                  const Eigen::MatrixBase<X_t> &X_vec, unsigned k)
    {
        return controlProblem.template mayer_term_impl<scalar_t>(get_col<NX>(X_vec, k));
    }

    /* Dynamic constraints */
    struct ContinuousDynamicsAt {};
    template<typename X_t, typename U_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(ContinuousDynamicsAt,
                  const Eigen::MatrixBase<X_t> &X_vec, const Eigen::MatrixBase<U_t> &U_vec, unsigned k)
    {
        return controlProblem.template dynamics_impl<scalar_t>(get_col<NX>(X_vec, k), get_col<NU>(U_vec, k));
    }

public:
    explicit RadauCollocation(ControlProblem &ctrlProblem_) :
            controlProblem(ctrlProblem_)
    {
        PRINT("nodes: " << get_collocation_points().transpose() << "\n");
        PRINT("D    :\n" << get_diff_mat() << "\n");
    }

    using scalar_t = typename ControlProblem::Scalar; // TODO: Change in laOPT to accept Scalar
    using TimeTrajectory = Eigen::Vector<Scalar, N + 1>;
    using StateTrajectory = Eigen::Matrix<Scalar, NX, N + 1>;
    using InputTrajectory = Eigen::Matrix<Scalar, NU, N + 1>;

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
    const TimeTrajectory &get_T_opt() const { return T; }
    StateTrajectory get_X_opt()
    {
        Eigen::Vector<Scalar, (N + 1) * NX> X_opt_vec;
        X_opt_vec.setZero();
        X_opt_vec << X_var;

        StateTrajectory X_opt = StateTrajectory::Zero();
        for (unsigned i = 0; i <= N; i++)
        {
            using namespace Eigen;
            X_opt.col(i) << X_opt_vec.template segment<NX>(i * NX);
        }
        return X_opt;
    }
    InputTrajectory get_U_opt()
    {
        Eigen::Vector<Scalar, (N + 1) * NU> U_opt_vec;
        U_opt_vec.setZero();
        U_opt_vec << U_var;

        InputTrajectory U_opt;
        for (unsigned i = 0; i <= N; i++)
        {
            U_opt.col(i) << U_opt_vec.template segment<NU>(i * NU);
        }
        return U_opt;
    }

//protected: // TODO ino1 (would like to make this protected)
    template<typename OptProblem>
    void define_problem(OptProblem &optProblem)
    {
        const unsigned N_ = N;

        /* Register variables */
        optProblem.add_variable(X_var);
        optProblem.add_variable(U_var);

        for (unsigned i_seg = 0; i_seg < N_segs; i_seg++) // i_seg loops through all segments
        {
            const unsigned id_seg_start = i_seg * D_poly;

            /* For each node on the segment, add differential constraint */
            for (unsigned j_node = 0; j_node < D_poly; j_node++)
            {
                const unsigned k = id_seg_start + j_node; // Index of this node in the trajectory
                T(k) = i_seg * h_seg + h_seg * (Tau(j_node) + 1) / 2.0;

                optProblem.add_obj(this->function(StageCost{}, X_var, U_var, k));
                optProblem.add_constr(this->function(DifferentialApproximationAt{}, X_var, id_seg_start, j_node) ==
                                      this->function(ContinuousDynamicsAt{}, X_var, U_var, k));
            }
        }

        /* Last grid point */
        T(N) = controlProblem.tf;
        PRINT("T(" << N << ") = " << T(N));
        optProblem.add_obj(this->function(MayerCost{}, X_var, N_)); // Can't use N directly -> linking error, undefined reference to N

        /* Box constraints */
        const Eigen::Vector<Scalar, (N + 1) * NX> LBX = controlProblem.lbx.template replicate<N + 1, 1>();
        const Eigen::Vector<Scalar, (N + 1) * NX> UBX = controlProblem.ubx.template replicate<N + 1, 1>();
        const Eigen::Vector<Scalar, (N + 1) * NU> LBU = controlProblem.lbu.template replicate<N + 1, 1>();
        const Eigen::Vector<Scalar, (N + 1) * NU> UBU = controlProblem.ubu.template replicate<N + 1, 1>();
        optProblem.add_constr(LBX <= X_var <= UBX);
        optProblem.add_constr(LBU <= U_var <= UBU);

//        optProblem.add_constr(controlProblem.lbx.template replicate<N + 1, 1>()
//                              <= X_var <=
//                              controlProblem.ubx.template replicate<N + 1, 1>());
        optProblem.add_constr(controlProblem.lbu.template replicate<N + 1, 1>()
                              <= U_var <=
                              controlProblem.ubu.template replicate<N + 1, 1>());

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= this->function(StateAt{}, X_var, 0) <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= this->function(StateAt{}, X_var, N_) <= controlProblem.xf_ub);

        /* Set last control equal second last for easier data handling */
        optProblem.add_constr(this->function(InputAt{}, U_var, N_) == this->function(InputAt{}, U_var, N_ - 1));
    }
};

} // namespace transcription

#endif //LAOPT_RADAUCOLLOCATION_HPP
