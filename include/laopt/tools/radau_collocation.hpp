#ifndef LAOPT_RADAU_COLLOCATION_HPP
#define LAOPT_RADAU_COLLOCATION_HPP

// Advanced user (level 2)
#include <iostream>
#include <iomanip>

#include <Eigen/Dense>
#include "unsupported/Eigen/Polynomials"
#include "laopt/laopt.hpp"
#include "constants.hpp"

namespace laopt_tools {

#define PRINT(x) \
//std::cout << __FUNCTION__ << ": " << x << std::endl // Comment this line in to activate PRINT function in the code

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
    friend laopt::Differentiable<RadauCollocation<ControlProblem, N_segs, D_poly>>;

    template<typename, typename, typename, typename>
    friend class laopt::ProblemBase;

private: // Static functions
    template<unsigned mat_rows, unsigned row_start, unsigned N_rows, unsigned N_cols, typename Vec_t>
    static auto get_slice(Vec_t &vec, unsigned col_index)
    {
        std::array<unsigned, N_cols * N_rows> ind{};
        unsigned k = 0;
        for (unsigned iCol = col_index; iCol < (col_index + N_cols); iCol++)
        {
            for (unsigned jRow = row_start; jRow < (row_start + N_rows); jRow++)
            {
                ind[k++] = iCol * mat_rows + jRow;
            }
        }
        /* Pretend the long vector vec is a reshaped matrix of (NX + NU) x (N + 1) */
        return vec(ind);
    }
    template<unsigned mat_rows, typename Vec_t>
    static auto get_col(Vec_t &vec, unsigned col_index_start)
    {
        /* Call more generic function with one column */
        return get_slice<mat_rows, 0, mat_rows, 1>(vec, col_index_start);
    }

    template<typename DerivedA, typename DerivedB>
    static typename DerivedA::PlainObject
    poly_mul(const Eigen::MatrixBase<DerivedA> &p1, const Eigen::MatrixBase<DerivedB> &p2)
    {
        /*
         * This function was copied together from PolyMPC
         */
        typename DerivedA::PlainObject product = DerivedA::PlainObject::Zero();
        const int p1_size = DerivedA::RowsAtCompileTime;
        const int p2_size = DerivedA::RowsAtCompileTime;
        static_assert(p1_size == p2_size, "poly_mul: Polynomials must be of the same order!");

        using Scalar_ = typename DerivedA::Scalar;
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

protected:
    /* Mirror types from ControlProblem, define variable template with scalar type */
    using Scalar = typename ControlProblem::Scalar;
    static const unsigned NX = ControlProblem::NX;
    static const unsigned NU = ControlProblem::NU;
    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    /* Instance of end user's ControlProblem */
    ControlProblem &controlProblem;

    /* Decision variables (same number of inputs as states for easier data handling) */
    static const unsigned N = D_poly * N_segs; // Last index of decision variables
    const double h_seg{1.0 / N_segs};
    Eigen::Vector<Scalar, N + 1> T; // Normalized time grid (0 ... 1)
    variable_t<(N + 1) * (NX + NU)> XU_var;
    variable_t<1> tf_var;
    variable_t<ControlProblem::NP> p_var;

    /* Helper functions to extract single states and inputs from long decision variable vector */
    template<typename XU_t>
    auto get_x(XU_t &XU_Vec, unsigned k) const { return get_slice<NX + NU, 0, NX, 1>(XU_Vec, k); }
    template<unsigned N_cols, typename XU_t>
    auto get_x(XU_t &XU_Vec, unsigned k) const { return get_slice<NX + NU, 0, NX, N_cols>(XU_Vec, k); }

    template<typename XU_t>
    auto get_u(XU_t &XU_Vec, unsigned k) const { return get_slice<NX + NU, NX, NU, 1>(XU_Vec, k); }
    template<unsigned N_cols, typename XU_t>
    auto get_u(XU_t &XU_Vec, unsigned k) const { return get_slice<NX + NU, NX, NU, N_cols>(XU_Vec, k); }

    /* Collocation grid, differentiation and integration matrices */
    using CollocationPoints = Eigen::Vector<Scalar, D_poly + 1>;
    CollocationPoints get_collocation_points() const
    {
        if (0 < D_poly && D_poly < 10)
        {
            std::vector<std::vector<Scalar>> Radau_static{
                    {0.0, 1.0},
                    {0.0, 0.33333333333333337,  1.0},
                    {0.0, 0.15505102572168222,  0.6449489742783179,  1.0},
                    {0.0, 0.0885879595127042,   0.40946686444073466, 0.787659461760847,   1.0},
                    {0.0, 0.057104196114518224, 0.2768430136381237,  0.5835904323689168,  0.8602401356562193, 1.0},
                    {0.0, 0.039809857051469055, 0.19801341787360788, 0.4379748102473862,  0.695464273353636,  0.9014649142011735, 1.0},
                    {0.0, 0.02931642715978522,  0.14807859966848436, 0.3369846902811542,  0.5586715187715502, 0.7692338620300545, 0.926945671319741,  1.0},
                    {0.0, 0.022479386438713056, 0.11467905316090415, 0.2657898227845895,  0.4528463736694446, 0.6473752828868304, 0.8197593082631076, 0.9437374394630773, 1.0},
                    {0.0, 0.017779915147363934, 0.09132360789979432, 0.21430847939563036, 0.3719321645832724, 0.5451866848034266, 0.7131752428555695, 0.8556337429578544, 0.9553660447100301, 1.0},
            };
            return 2.0 *
                   (CollocationPoints::Constant(1.0) -
                    Eigen::Map<const CollocationPoints>(Radau_static.at(D_poly - 1).data())
                   ).reverse()
                   - CollocationPoints::Constant(1.0);
        }

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
    const CollocationPoints Tau = get_collocation_points();

    /* Lagrange polynomials */
    template<unsigned n, typename scalar_t = Scalar>
    Eigen::Vector<Scalar, n> interpolate(const Eigen::Vector<Scalar, n * (D_poly + 1)> &X_seg, Scalar tau_eval) const
    {
        Eigen::Vector<scalar_t, n> x_apr; // n x 1
        x_apr.setZero();
        for (unsigned l = 0; l <= D_poly; l++)
        {
            x_apr += L(l, tau_eval) * get_col<n>(X_seg, l);
        }
        return x_apr;
    }
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

protected:
    /* Dynamic constraints */
    struct ContinuousDynamics {};
    template<typename x_t, typename u_t, typename p_t, typename tf_t,
            typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(ContinuousDynamics,
                  const Eigen::MatrixBase<x_t> &x,
                  const Eigen::MatrixBase<u_t> &u,
                  const Eigen::MatrixBase<p_t> &p,
                  const Eigen::MatrixBase<tf_t> &tf)
    {
        return tf(0) * controlProblem.template dynamics_impl<scalar_t>(x, u, p);
    }

    struct DifferentialApproximation {};
    template<typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, NX>
    function_impl(DifferentialApproximation,
                  const Eigen::MatrixBase<X_t> &X_vec, unsigned j_node)
    {
        /* Loop through nodes in segment and add contribution to differential approximation at node j_node */
        Eigen::Vector<scalar_t, NX> dx_apr; // NX x 1
        dx_apr.setZero();
        for (unsigned l = 0; l <= D_poly; l++)
        {
            dx_apr += diff_mat(j_node, l) * get_col<NX>(X_vec, l);
        }
        return 2.0 / h_seg * dx_apr;
    }

    template<typename OutJacobian, typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE void
    jacobian_impl(DifferentialApproximation, OutJacobian& out_jacobian,
                  const Eigen::MatrixBase<X_t> &X_vec, unsigned j_node)
    {
        for (unsigned l = 0; l <= D_poly; l++)
        {
            Scalar diag = 2.0 / h_seg * diff_mat(j_node, l);
            // assign diagonal values
            for (unsigned i = 0; i < NX; i++) {
                out_jacobian(i, l * NX + i) += diag;
            }
        }
    }

    template <typename Weight, typename OutGradient,
              typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE void
    gradient_impl(DifferentialApproximation, OutGradient& out_gradient, const Eigen::MatrixBase<Weight>& weight,
                  const Eigen::MatrixBase<X_t> &X_vec, unsigned j_node)
    {
        for (unsigned l = 0; l <= D_poly; l++)
        {
            out_gradient(Eigen::seqN(l * NX, Eigen::fix<NX>)) += 2.0 / h_seg * diff_mat(j_node, l) * weight;
        }
    }

    template<typename Weight, typename OutHessian,
             typename X_t, typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE void
    hessian_impl(DifferentialApproximation, OutHessian&, const Eigen::MatrixBase<Weight>& weight,
                 const Eigen::MatrixBase<X_t> &X_vec, unsigned j_node)
    {
        // Hessian is zero, i.e., we don't set any values
    }

    /* Inequality constraints */
    struct InequalityConstraints {};
    template<typename x_t, typename u_t, typename p_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NG>
    function_impl(InequalityConstraints,
                  const Eigen::MatrixBase<x_t> &x,
                  const Eigen::MatrixBase<u_t> &u,
                  const Eigen::MatrixBase<p_t> &p)
    {
        return controlProblem.template inequality_constraints_impl<scalar_t>(x, u, p);
    }

    struct InitialInequalityConstraints {};
    template<typename x_t, typename u_t, typename p_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NG0>
    function_impl(InitialInequalityConstraints,
                  const Eigen::MatrixBase<x_t> &x0,
                  const Eigen::MatrixBase<u_t> &u0,
                  const Eigen::MatrixBase<p_t> &p)
    {
        return controlProblem.template inequality_constraints0_impl<scalar_t>(x0, u0, p);
    }

    struct FinalInequalityConstraints {};
    template<typename x_t, typename p_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NGF>
    function_impl(FinalInequalityConstraints,
                  const Eigen::MatrixBase<x_t> &xf,
                  const Eigen::MatrixBase<p_t> &p)
    {
        return controlProblem.template inequality_constraintsf_impl<scalar_t>(xf, p);
    }

    /* Objective */
    struct NodeCost {};
    template<typename X_t, typename U_t, typename p_t,
            typename scalar_t = typename Eigen::MatrixBase<X_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(NodeCost,
                  const Eigen::MatrixBase<X_t> &x,
                  const Eigen::MatrixBase<U_t> &u,
                  const Eigen::MatrixBase<p_t> &p,
                  const unsigned j_node)
    {
        /* Contribution of this node to integral approximation of the segment the node is in */
        return h_seg / 2.0 * int_mat(int_mat.rows() - 1, j_node) *
            controlProblem.template lagrange_term_impl<scalar_t>(x, u, p);
    }

    struct MayerCost {};
    template<typename x_t, typename p_t, typename tf_t,
            typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost,
                  const Eigen::MatrixBase<x_t> &x,
                  const Eigen::MatrixBase<p_t> &p,
                  const Eigen::MatrixBase<tf_t> &tf)
    {
        return controlProblem.template mayer_term_impl<scalar_t>(x, p, tf(0));
    }

    template<int Option = ControlProblem::Options>
    inline typename std::enable_if<(Option & FreeEndTime) == 0, Eigen::Vector<Scalar, 1>>::type
    get_tf_var() const
    {
        if (controlProblem.tf_lb != controlProblem.tf_ub)
        {
            std::cerr << "RadauCollocation<FixedEndTime>: final time bounds need to be identical (tf_lb == tf_ub)\n";
            exit(EXIT_FAILURE);
        }
        Eigen::Vector<Scalar, 1> tf;
        tf(0) = controlProblem.tf_lb;
        return tf;
    }

    template<int Option = ControlProblem::Options>
    inline typename std::enable_if<(Option & FreeEndTime) != 0, const variable_t<1>&>::type
    get_tf_var() const
    {
        return tf_var;
    }

    template<typename OptProblem>
    void define_problem(OptProblem &optProblem)
    {
        /* Register variables */
        optProblem.add_variable(XU_var);
        if (ControlProblem::Options & FreeEndTime)
        {
            optProblem.add_variable(tf_var);
        }
        optProblem.add_variable(p_var);

        /* Loop through segments */
        for (unsigned i_seg = 0; i_seg < N_segs; i_seg++)
        {
            const unsigned id_seg_start = i_seg * D_poly;
            const auto X_seg_diff = get_x<D_poly + 1>(XU_var, id_seg_start); // Diff. approx depends on all points

            /* Loop through nodes in segment */
            for (unsigned j_node = 0; j_node < D_poly; j_node++)
            {
                const unsigned k = id_seg_start + j_node; // Index of this node in the trajectory

                /* Add contribution of this node to integral approximation of this segment */
                optProblem.add_obj(this->function(NodeCost{}, get_x(XU_var, k), get_u(XU_var, k), p_var, j_node));

                /* Add differential constraint at each node */
                optProblem.add_constr(this->function(DifferentialApproximation{}, X_seg_diff, j_node) ==
                                      this->function(ContinuousDynamics{}, get_x(XU_var, k), get_u(XU_var, k), p_var, get_tf_var()));
            }
        }

        /* Last grid point */
        optProblem.add_obj(this->function(MayerCost{}, get_x(XU_var, N), p_var, get_tf_var()));

        /* Box constraints */
        for (unsigned k = 0; k <= N; k++)
        {
            optProblem.add_constr(controlProblem.x_lb <= get_x(XU_var, k) <= controlProblem.x_ub);
            optProblem.add_constr(controlProblem.u_lb <= get_u(XU_var, k) <= controlProblem.u_ub);
        }

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= get_x(XU_var, 0) <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= get_x(XU_var, N) <= controlProblem.xf_ub);
        if (ControlProblem::Options & FreeEndTime)
        {
            optProblem.add_constr(controlProblem.tf_lb <= tf_var <= controlProblem.tf_ub);
        }
        optProblem.add_constr(controlProblem.opt_params_lb.vector() <= p_var <= controlProblem.opt_params_ub.vector());

        /* Inequality constraints */
        optProblem.add_constr(this->function(InitialInequalityConstraints{},  get_x(XU_var, 0), get_u(XU_var, 0), p_var) <= 0);

        for (unsigned k = 1; k < N; k++)
        {
            optProblem.add_constr(this->function(InequalityConstraints{}, get_x(XU_var, k), get_u(XU_var, k), p_var) <= 0);
        }
        optProblem.add_constr(this->function(FinalInequalityConstraints{}, get_x(XU_var, N), p_var) <= 0);
        // TODO: FinalInequalityConstraints could also be on input for collocation scheme

        /* Set last control equal second last for easier data handling */
        optProblem.add_constr(get_u(XU_var, N) == get_u(XU_var, N - 1));
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
                T(k) = i_seg * h_seg + h_seg * (Tau(j_node) + 1.0) / 2.0;
            }
        }
        T(N) = 1;

        PRINT("nodes: " << get_collocation_points().transpose() << "\n");
        PRINT("D    :\n" << get_diff_mat() << "\n");
        PRINT("I    :\n" << get_int_mat() << "\n");
    }

    using scalar_t = typename ControlProblem::Scalar; // TODO: Change in laOPT to accept Scalar
    using State = typename ControlProblem::State;
    using Input = typename ControlProblem::Input;
    using Param = typename ControlProblem::Param;
    using TimeTrajectory = Eigen::Vector<Scalar, N + 1>;
    using StateTrajectory = Eigen::Matrix<Scalar, ControlProblem::NX, N + 1>;
    using InputTrajectory = Eigen::Matrix<Scalar, ControlProblem::NU, N + 1>;

    /* Set functions */
    void set_X_guess(const State &x_guess)
    {
        for (unsigned k = 0; k <= N; k++) { get_x(XU_var, k) << x_guess; }
    }
    void set_X_guess(const StateTrajectory &X_guess)
    {
        for (unsigned k = 0; k <= N; k++) { get_x(XU_var, k) << X_guess.col(k); }
    }
    void set_U_guess(const Input &u_guess)
    {
        for (unsigned k = 0; k <= N; k++) { get_u(XU_var, k) << u_guess; }
    }
    void set_U_guess(const InputTrajectory &U_guess)
    {
        for (unsigned k = 0; k <= N; k++) { get_u(XU_var, k) << U_guess.col(k); }
    }
    void set_p_guess(const Param &p_guess) { p_var = p_guess; }

    /* Get functions */
    double get_tf_opt() const
    {
        return get_tf_var()(0);
    }
    TimeTrajectory get_T_opt() const
    {
        return TimeTrajectory::Constant(controlProblem.t0) + (get_tf_opt() - controlProblem.t0) * T;
    }
    StateTrajectory get_X_opt() const
    {
        StateTrajectory X_opt;
        X_opt.setZero();
        for (unsigned i = 0; i <= N; i++) { X_opt.col(i) << get_x(XU_var, i); }
        return X_opt;
    }
    InputTrajectory get_U_opt() const
    {
        InputTrajectory U_opt;
        U_opt.setZero();
        for (unsigned i = 0; i <= N; i++) { U_opt.col(i) << get_u(XU_var, i); }
        return U_opt;
    }
    Param get_p_opt() const { return Param(p_var); }
    typename ControlProblem::OptParam get_opt_params() const
    {
        typename ControlProblem::OptParam opt_param;
        opt_param.set_vector(get_p_opt());
        return opt_param;
    }

    Eigen::Vector<Scalar, NX> get_x_at(const Scalar &t) const
    {
        const Scalar T_eval = (t - controlProblem.t0) / (get_tf_opt() - controlProblem.t0); // on [0 ... 1]|traj;

        /* Find segment to sample from */
        const unsigned i_seg = std::floor(T_eval / h_seg);
        const unsigned i_seg_start = i_seg * D_poly;
        PRINT("i_seg: " << i_seg << ", i_seg_start: " << i_seg_start);

        const Eigen::Vector<Scalar, NX * (D_poly + 1)> X_seg = get_x<D_poly + 1>(XU_var, i_seg_start);
        const Scalar t_eval = T_eval - i_seg * h_seg;     // Time in segment [0 ... 1]|seg
        const Scalar tau_eval = 2.0 * t_eval / h_seg - 1; // Time on [-1 ... 1]|
        PRINT("X_seg:\n" << X_seg);
        PRINT("t_eval: " << t_eval);
        PRINT("tau_eval: " << tau_eval);
        return interpolate<NX>(X_seg, tau_eval);
    }
    Eigen::Vector<Scalar, NU> get_u_at(const Scalar &t) const
    {
        const Scalar T_eval = t / (get_tf_opt() - controlProblem.t0) - controlProblem.t0; // on [0 ... 1];

        /* Find segment to sample from */
        const unsigned i_seg = std::floor(T_eval / h_seg);
        const unsigned i_seg_start = i_seg * D_poly;

        /* Sample from segment */
        const Eigen::Vector<Scalar, NU * (D_poly + 1)> X_seg = get_u<D_poly + 1>(XU_var, i_seg_start);
        const Scalar t_eval = T_eval - i_seg * h_seg;     // Time in segment [0 ... 1]|seg
        const Scalar tau_eval = 2.0 * t_eval / h_seg - 1; // Time on [-1 ... 1]|
        return interpolate<NU>(X_seg, tau_eval);
    }

    Eigen::MatrixX<Scalar> get_TX_resampled(const Scalar &Ts_max) const
    {
        return resample_trajectory(get_T_opt(), get_X_opt(), Ts_max);
    }
    Eigen::MatrixX<Scalar> get_TU_resampled(const Scalar &Ts_max) const
    {
        return resample_trajectory(get_T_opt(), get_U_opt(), Ts_max);
    }

    /* Diagnosis */
    void print_diagnostics() const
    {
        std::cout << std::setprecision(4) << std::defaultfloat;
        std::cout << "Diagnostics: Radau Collocation with N_segs = " << N_segs << ", D_poly = " << D_poly << "\n";
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
    Eigen::Matrix<Scalar, DerivedNX + 1, -1> resample_trajectory(const Eigen::Vector<Scalar, DerivedNT1> &T_opt,
                                                                 const Eigen::Matrix<Scalar, DerivedNX, DerivedNT2> &X_opt,
                                                                 Scalar Ts_max) const
    {
        static_assert(DerivedNT1 == DerivedNT2, "T and X must be of same length.");

        const Scalar dT = (T_opt(D_poly) - T_opt(0));
        unsigned n_per_seg = std::floor(dT / Ts_max);
        if (n_per_seg * Ts_max < dT) { ++n_per_seg; };
        const unsigned n = N_segs * n_per_seg;
        PRINT("T_opt(D_poly): " << T_opt(D_poly) << ", n_per_seg: " << n_per_seg << ", n (total): " << n);

        Eigen::Matrix<Scalar, DerivedNX + 1, -1> TXn(DerivedNX + 1, n + 1);
        TXn.setZero();

        using namespace Eigen;
        for (unsigned i_seg = 0; i_seg < N_segs; i_seg++)
        {
            const unsigned i_seg_start = i_seg * D_poly;
            const unsigned k_seg_start = i_seg * n_per_seg;
            PRINT("-------------------------- \n"
                  "i_seg: " << i_seg << ", i_seg_start: " << i_seg_start << ", k_seg_start: " << k_seg_start);

            const auto X_seg = X_opt(all, seqN(i_seg_start, D_poly + 1));
            PRINT("X_seg:\n" << X_seg);

            for (unsigned j = 0; j < n_per_seg; j++)
            {
                const unsigned k = k_seg_start + j;
                const Scalar T_eval = j * 1.0 / n_per_seg; // Time on [0 ... 1]
                const Scalar tau_eval = 2.0 * T_eval - 1;  // Time on [-1 ... 1]
                PRINT("j: " << j << ", k: " << k << ", tau: " << tau_eval);

                TXn(0, k) = (i_seg * h_seg + h_seg * T_eval);
                TXn(seqN(1, DerivedNX), k) << interpolate<DerivedNX>(X_seg.template reshaped<ColMajor>(), tau_eval);
            }

            /* In last segment, write last point */
            if (i_seg == N_segs - 1)
            {
                TXn(0, n) = get_tf_opt();
                /* Copy or extrapolate */
                TXn(seqN(1, DerivedNX), n) << X_opt(all, last);
//                TXn(seqN(1, DerivedNX), n) << interpolate<DerivedNX>(X_seg.template reshaped<ColMajor>(), 1);
            }

            /* Transform time by absolute horizon range (except  */
            TXn(0, seqN(k_seg_start, n_per_seg)) =
                    Eigen::MatrixX<Scalar>::Constant(1, n_per_seg, controlProblem.t0) +
                    (get_tf_opt() - controlProblem.t0) * TXn(0, seqN(k_seg_start, n_per_seg));
            PRINT("\n" << TXn << "\n");
        }
        return TXn;
    }
};

} // namespace laopt_tools

#endif //LAOPT_RADAU_COLLOCATION_HPP
