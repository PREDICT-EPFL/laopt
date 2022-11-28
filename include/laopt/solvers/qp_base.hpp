#ifndef LAOPT_QP_BASE_HPP
#define LAOPT_QP_BASE_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace laopt
{

template <typename Scalar>
struct qp_solver_settings_t {
    /** Common settings */
    Scalar eps_rel       = 1e-6;  /** Relative tolerance for termination, 0 < eps_rel */
    Scalar eps_abs       = 1e-6;  /** Absolute tolerance for termination, 0 < eps_abs */
    int    max_iter      = 4000;  /** Maximal number of iteration, 0 < max_iter */
    bool   reuse_pattern = false; /** Assume that problem size and sparsity pattern have not changed since last 'solve call' */
    bool   verbose       = false;
};

enum struct qp_status_t {
    SOLVED = 1,
    MAX_ITER_REACHED = -1,
    INFEASIBLE = -2,
    NON_CONVEX = -3,
    UNSOLVED = -9,
    INVALID_SETTINGS = -10
};

struct qp_solver_info_t {
    qp_status_t status = qp_status_t::UNSOLVED;
    int iter = 0;
};

template<typename Derived, typename Scalar = double>
class QPBase
{
public:
    using scalar_t = Scalar;

protected:
    int m_n; // number of decision variables
    int m_m; // number of constraints

    Eigen::VectorX<scalar_t> m_x;          // primal decision variable
    Eigen::VectorX<scalar_t> m_lam;        // dual variable for constraints
    Eigen::VectorX<scalar_t> m_lam_bounds; // dual variable for simple bounds

    qp_solver_settings_t<scalar_t> m_settings;
    qp_solver_info_t m_info;

    enum struct constraint_t {
        EQ_CONSTR,
        INEQ_CONSTR,
        INEQ_LB_ONLY_CONSTR,
        INEQ_UB_ONLY_CONSTR,
        UNBOUNDED_CONSTR
    };

    std::vector<constraint_t> m_box_constraint_type; // box constraint classification
    std::vector<constraint_t> m_constraint_type;     // general constraint classification

public:
    static constexpr scalar_t UNBOUNDED_THRESHOLD = 1e+10;
    static constexpr scalar_t EQ_TOL = 1e-6;

    QPBase(int n, int m) :
        m_n(n), m_m(m),
        m_x(n), m_lam(m), m_lam_bounds(n),
        m_box_constraint_type(n), m_constraint_type(m)
    {
        m_x.setZero();
        m_lam.setZero();
        m_lam_bounds.setZero();
    };

    /** getters  / setters */
    EIGEN_STRONG_INLINE const Eigen::VectorX<scalar_t>& primal_solution() const noexcept { return m_x; }
    EIGEN_STRONG_INLINE Eigen::VectorX<scalar_t>& primal_solution() noexcept { return m_x; }

    EIGEN_STRONG_INLINE const Eigen::VectorX<scalar_t>& dual_solution() const { return m_lam; }
    EIGEN_STRONG_INLINE Eigen::VectorX<scalar_t>& dual_solution() noexcept { return m_lam; }

    EIGEN_STRONG_INLINE const Eigen::VectorX<scalar_t>& dual_bounds_solution() const { return m_lam_bounds; }
    EIGEN_STRONG_INLINE Eigen::VectorX<scalar_t>& dual_bounds_solution() noexcept { return m_lam_bounds; }

    EIGEN_STRONG_INLINE const qp_solver_settings_t<scalar_t>& settings() const noexcept { return m_settings; }
    EIGEN_STRONG_INLINE qp_solver_settings_t<scalar_t>& settings() noexcept { return m_settings; }

    EIGEN_STRONG_INLINE const qp_solver_info_t& info() const noexcept { return m_info; }
    EIGEN_STRONG_INLINE qp_solver_info_t& info() noexcept { return m_info; }

    /** solve with generic and box constraints*/
    qp_solver_info_t solve(const Eigen::SparseMatrix<scalar_t>& H,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                           const Eigen::SparseMatrix<scalar_t>& A,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert(H.isCompressed() && H.rows() == m_n && H.cols() == m_n);
        eigen_assert(f.rows() == m_n);
        eigen_assert(xlb.rows() == m_n && xub.rows() == m_n);
        eigen_assert(A.isCompressed() && A.rows() == m_m && A.cols() == m_n);
        eigen_assert(Alb.rows() == m_m && Aub.rows() == m_m);
        return static_cast<Derived*>(this)->solve_impl(H, f, xlb, xub, A, Alb, Aub);
    }

protected:

    EIGEN_STRONG_INLINE void
    parse_constraints_bounds(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert((xlb.array() <= xub.array()).any());
        eigen_assert((Alb.array() <= Aub.array()).any());

        // parse box constraints
        for (int i = 0; i < m_n; i++)
        {
            if (xlb(i) < -UNBOUNDED_THRESHOLD && xub(i) > UNBOUNDED_THRESHOLD)
            {
                m_box_constraint_type[i] = constraint_t::UNBOUNDED_CONSTR;
            }
            else if (xub(i) - xlb(i) < EQ_TOL)
            {
                m_box_constraint_type[i] = constraint_t::EQ_CONSTR;
            }
            else if (xub(i) > UNBOUNDED_THRESHOLD)
            {
                m_box_constraint_type[i] = constraint_t::INEQ_LB_ONLY_CONSTR;
            }
            else if (xlb(i) < -UNBOUNDED_THRESHOLD)
            {
                m_box_constraint_type[i] = constraint_t::INEQ_UB_ONLY_CONSTR;
            }
            else
            {
                m_box_constraint_type[i] = constraint_t::INEQ_CONSTR;
            }
        }

        // parse general constraints
        for (int i = 0; i < m_m; i++)
        {
            if (Alb(i) < -UNBOUNDED_THRESHOLD && Aub(i) > UNBOUNDED_THRESHOLD)
            {
                m_constraint_type[i] = constraint_t::UNBOUNDED_CONSTR;
            }
            else if (Aub(i) - Alb(i) < EQ_TOL)
            {
                m_constraint_type[i] = constraint_t::EQ_CONSTR;
            }
            else if (Aub(i) > UNBOUNDED_THRESHOLD)
            {
                m_constraint_type[i] = constraint_t::INEQ_LB_ONLY_CONSTR;
            }
            else if (Alb(i) < -UNBOUNDED_THRESHOLD)
            {
                m_constraint_type[i] = constraint_t::INEQ_UB_ONLY_CONSTR;
            }
            else
            {
                m_constraint_type[i] = constraint_t::INEQ_CONSTR;
            }
        }
    }
};

} // namespace laopt

#endif // LAOPT_QP_BASE_HPP
