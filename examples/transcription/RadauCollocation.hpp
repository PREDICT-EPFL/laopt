#ifndef LAOPT_RADAUCOLLOCATION_HPP
#define LAOPT_RADAUCOLLOCATION_HPP

// Advanced user (level 2)

#include <Eigen/Dense>
#include "unsupported/Eigen/Polynomials"
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
    /* Mirror types from ControlProblem, define variable template with scalar type */
    using Scalar = typename ControlProblem::Scalar;
    static const unsigned NX = ControlProblem::NX;
    static const unsigned NU = ControlProblem::NU;
    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    /* Instance of end user's ControlProblem */
    ControlProblem &controlProblem;

    /* Create discrete problem variables (define U_var with same length than X_var for easier data handling,
     * although last u will not be used */
    static const unsigned N = D_poly * N_segs; // Last index of decision variables
    const double h_seg{1.0 / N_segs};
    Eigen::Vector<Scalar, N + 1> T; // Normalized time grid (0 ... 1)
    variable_t<(N + 1) * NX> X_var;
    variable_t<(N + 1) * NU> U_var;

    /*
     * Static functions
     */
    template<unsigned mat_rows, typename X_t>
    static auto get_col(const X_t &X, unsigned col_index_start)
    {
        /* Call more generic function with one column */
        return get_cols<mat_rows, 1>(X, col_index_start);
    }
    template<unsigned mat_rows, unsigned N_cols, typename X_t>
    static auto get_cols(const X_t &X, unsigned col_index_start)
    {
        /* Pretend X_t is a column-wise reshaped matrix with mat_rows */
        const unsigned i0 = col_index_start * mat_rows;
        return X(Eigen::seqN(i0, Eigen::fix<N_cols * mat_rows>));
    }

    /* Collocation basis and grid */
    template<typename Derived, typename DerivedB>
    static typename Derived::PlainObject
    poly_mul(const Eigen::MatrixBase<Derived> &p1, const Eigen::MatrixBase<DerivedB> &p2)
    {
        /*
         * This function was copied together from PolyMPC
         */
        typename Derived::PlainObject product = Derived::PlainObject::Zero();
        const int p1_size = Derived::RowsAtCompileTime;
        const int p2_size = Derived::RowsAtCompileTime;
        static_assert(p1_size == p2_size, "poly_mul: Polynomials must be of the same order!");

        using Scalar_ = typename Derived::Scalar;
        Scalar_ eps = std::numeric_limits<Scalar_>::epsilon();

        /* Detect nonzeros */
        int nnz_p1, nnz_p2;
        for (int i = 0; i < p1_size; ++i)
        {
            if (std::fabs(p1[i]) >= eps) { nnz_p1 = i; }
            if (std::fabs(p2[i]) >= eps) { nnz_p2 = i; }
        }

        for (int i = 0; i <= nnz_p2; ++i)
        {
            for (int j = 0; j <= nnz_p1; ++j)
            {
                /* Truncate higher orders if necessary */
                if ((i + j) == p1_size) { break; }
                product[i + j] = p1[j] * p2[i];
            }
        }

        return product;
    }
    /*
     * End of Static functions
     */

protected:
    using CollocationPoints = Eigen::Vector<Scalar, D_poly + 1>;
    CollocationPoints get_collocation_points() const
    {
        /*
         * This function was copied together from PolyMPC
         */
        using Basis = Eigen::Matrix<Scalar, D_poly + 1, D_poly + 1>;

        /* Compute basis --------------------------------------------------------- */
        Basis Ln = Basis::Zero();
        /* The first basis polynomial is L0(x) = 1 */
        Ln(0, 0) = 1;
        /* The second basis polynomial is L1(x) = x */
        Ln(1, 1) = 1;

        /* Compute recurrent coefficients */
        CollocationPoints a = CollocationPoints::Zero();
        CollocationPoints c = CollocationPoints::Zero();
        CollocationPoints x = CollocationPoints::Zero(); // p(x) = x
        x[1] = 1;
        for (unsigned n = 0; n <= D_poly; ++n)
        {
            a(n) = scalar_t(2 * n + 1) / (n + 1);
            c(n) = scalar_t(n) / (n + 1);
        }

        /* Create polynomial basis */
        for (unsigned n = 1; n < D_poly; ++n)
        {
            Ln.col(n + 1) = a(n) * poly_mul(Ln.col(n), x) - c(n) * Ln.col(n - 1);
        }

        /* Compute collocation points --------------------------------------------------------- */
        /* Legendre Gauss Radau (LGR) collocation points for the interval [-1, 1]*/
        /* Compute roots of LN-1 + LN */
        CollocationPoints Ln_sum = Ln.col(D_poly - 1) + Ln.col(D_poly);
        Scalar eps = std::numeric_limits<Scalar>::epsilon();

        /* prepare the polynomial for the solver */
        for (unsigned i = 0; i < D_poly; ++i)
        {
            if (std::fabs(Ln_sum[i]) <= eps) { Ln_sum[i] = Scalar(0); }
        }

        Eigen::PolynomialSolver<Scalar, D_poly> root_finder;
        root_finder.compute(Ln_sum);

        CollocationPoints nodes = CollocationPoints::Zero();
        nodes[D_poly] = 1;

        nodes.template segment<D_poly>(0) = root_finder.roots().real();

        /* Sort the nodes in ascending order */
        std::sort(nodes.data(), nodes.data() + nodes.size());
        return nodes;
    }
    const Eigen::Vector<Scalar, D_poly + 1> Tau = get_collocation_points(); // collocation_points::get();
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
                if (l != i) { L *= (tau_eval - Tau(l)) / (Tau(j) - Tau(l)); }
            }
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
    IntMat get_int_mat() const
    {
        DiffMat D = get_diff_mat();
        return D.template block<D_poly, D_poly>(0, 1).inverse();
    }
    const IntMat int_mat = get_int_mat();

public:
    struct DifferentialApproximation {};
    template<typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(DifferentialApproximation, const Eigen::MatrixBase<X_t> &X_vec, unsigned j_node)
    {
        /* Loop through nodes in segment and add contribution to differential approximation at node j_node */
        Eigen::Vector<scalar_t, NX> x_apr; // NX x 1
        x_apr.setZero();
        for (unsigned l = 0; l <= D_poly; l++)
        {
            x_apr += diff_mat(j_node, l) * get_col<NX>(X_vec, l);
        }
        return 2.0 / h_seg * x_apr;
    }

    /* Objective */
    struct SegmentCost {};
    template<typename X_t, typename U_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(SegmentCost, const Eigen::MatrixBase<X_t> &X_vec, const Eigen::MatrixBase<U_t> &U_vec)
    {
        /* Loop through nodes in segment and add contribution to integral approximation at node j_node */
        scalar_t cost{0};
        for (unsigned l = 0; l < D_poly; l++)
        {
            cost += int_mat(int_mat.rows() - 1, l) *
                    controlProblem.template lagrange_term_impl<scalar_t>(get_col<NX>(X_vec, l), get_col<NU>(U_vec, l));
        }
        return h_seg / 2.0 * cost;
    }

    struct MayerCost {};
    template<typename x_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost, const Eigen::MatrixBase<x_t> &x)
    {
        return controlProblem.template mayer_term_impl<scalar_t>(x);
    }

    /* Dynamic constraints */
    struct ContinuousDynamics {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(ContinuousDynamics, const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        return controlProblem.template dynamics_impl<scalar_t>(x, u);
    }

public:
    explicit RadauCollocation(ControlProblem &ctrlProblem_) :
            controlProblem(ctrlProblem_)
    {
        /* Construct trajectory time grid on [0, 1] */
        for (unsigned i_seg = 0; i_seg < N_segs; i_seg++)
        {
            const unsigned id_seg_start = i_seg * D_poly;
            for (unsigned j_node = 0; j_node < D_poly; j_node++)
            {
                const unsigned k = id_seg_start + j_node;
                T(k) = i_seg * h_seg + h_seg * (Tau(j_node) + 1) / 2.0;
            }
        }
        T(N) = 1;

        PRINT("nodes: " << get_collocation_points().transpose() << "\n");
        PRINT("D    :\n" << get_diff_mat() << "\n");
        PRINT("I    :\n" << get_int_mat() << "\n");
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
    TimeTrajectory get_T_opt()
    {
        return TimeTrajectory::Constant(controlProblem.t0) + (controlProblem.tf - controlProblem.t0) * T;
    }
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
        /* Register variables */
        optProblem.add_variable(X_var);
        optProblem.add_variable(U_var);

        for (unsigned i_seg = 0; i_seg < N_segs; i_seg++) // i_seg loops through all segments
        {
            const unsigned id_seg_start = i_seg * D_poly;
            const auto X_seg = get_cols<NX, D_poly + 1>(X_var, id_seg_start);
            const auto U_seg = get_cols<NU, D_poly + 1>(U_var, id_seg_start);

            optProblem.add_obj(this->function(SegmentCost{}, X_seg, U_seg));

            /* Loop through nodes in segment and add differential constraint */
            for (unsigned j_node = 0; j_node < D_poly; j_node++)
            {
                const unsigned k = id_seg_start + j_node; // Index of this node in the trajectory

                optProblem.add_constr(this->function(DifferentialApproximation{}, X_seg, j_node) ==
                                      this->function(ContinuousDynamics{},
                                                     get_col<NX>(X_var, k), get_col<NU>(U_var, k)));
            }
        }

        /* Last grid point */
        optProblem.add_obj(this->function(MayerCost{}, get_col<NX>(X_var, N)));

        /* Box constraints */
        optProblem.add_constr(controlProblem.lbx.template replicate<N + 1, 1>() <= X_var <=
                              controlProblem.ubx.template replicate<N + 1, 1>());
        optProblem.add_constr(controlProblem.lbu.template replicate<N + 1, 1>() <= U_var <=
                              controlProblem.ubu.template replicate<N + 1, 1>());

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= get_col<NX>(X_var, 0) <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= get_col<NX>(X_var, N) <= controlProblem.xf_ub);

        /* Set last control equal second last for easier data handling */
        optProblem.add_constr(get_col<NU>(U_var, N) == get_col<NU>(U_var, N - 1));
    }
};

} // namespace transcription

#endif //LAOPT_RADAUCOLLOCATION_HPP