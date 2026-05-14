#ifndef LAOPT_OSQP_INTERFACE_HPP
#define LAOPT_OSQP_INTERFACE_HPP

#include <type_traits>

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "osqp.h"

namespace laopt
{

template<typename Scalar = double>
class OSQPSolver : public QPBase<OSQPSolver<Scalar>, Scalar>
{
public:
    using Base = QPBase<OSQPSolver<Scalar>, Scalar>;
    using scalar_t = typename Base::scalar_t;

    static_assert(std::is_same<scalar_t, OSQPFloat>::value,
                  "laopt::OSQPSolver scalar type must match OSQPFloat");

private:
    using constraint_t = typename Base::constraint_t;
    using constraint_changed_t = typename Base::constraint_changed_t;

    ::OSQPSolver* m_osqp_solver;
    OSQPSettings m_osqp_settings;
    OSQPCscMatrix m_P_osqp_view;
    OSQPCscMatrix m_A_osqp_view;
    bool m_osqp_initialized;

    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, OSQPInt> m_P_osqp;
    Eigen::VectorX<scalar_t> m_q_osqp;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, OSQPInt> m_A_osqp;
    Eigen::VectorX<scalar_t> m_Alb_osqp;
    Eigen::VectorX<scalar_t> m_Aub_osqp;
    Eigen::VectorX<scalar_t> m_x_osqp;
    Eigen::VectorX<scalar_t> m_lam_osqp;

public:
    OSQPSolver(int n, int m) : Base(n, m)
    {
        this->m_settings.max_iter = 10000;

        m_osqp_solver = nullptr;
        m_P_osqp_view = {};
        m_A_osqp_view = {};
        m_osqp_initialized = false;

        osqp_set_default_settings(&m_osqp_settings);

        set_osqp_settings();
    }

    ~OSQPSolver()
    {
        if (m_osqp_solver != nullptr) {
            osqp_cleanup(m_osqp_solver);
        }
    }

    qp_solver_info_t solve_impl(const Eigen::SparseMatrix<scalar_t>& H,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                const Eigen::SparseMatrix<scalar_t>& A,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        set_osqp_settings();

        construct_osqp_data(H, f, xlb, xub, A, Alb, Aub);

        OSQPCscMatrix_set_data(&m_P_osqp_view,
                               static_cast<OSQPInt>(m_P_osqp.rows()),
                               static_cast<OSQPInt>(m_P_osqp.cols()),
                               static_cast<OSQPInt>(m_P_osqp.nonZeros()),
                               m_P_osqp.valuePtr(),
                               m_P_osqp.innerIndexPtr(),
                               m_P_osqp.outerIndexPtr());
        OSQPCscMatrix_set_data(&m_A_osqp_view,
                               static_cast<OSQPInt>(m_A_osqp.rows()),
                               static_cast<OSQPInt>(m_A_osqp.cols()),
                               static_cast<OSQPInt>(m_A_osqp.nonZeros()),
                               m_A_osqp.valuePtr(),
                               m_A_osqp.innerIndexPtr(),
                               m_A_osqp.outerIndexPtr());

        if (!this->m_settings.reuse_pattern || !m_osqp_initialized)
        {
            if (m_osqp_solver != nullptr) {
                osqp_cleanup(m_osqp_solver);
                m_osqp_solver = nullptr;
            }

            const OSQPInt exitflag = osqp_setup(&m_osqp_solver,
                                                &m_P_osqp_view,
                                                m_q_osqp.data(),
                                                &m_A_osqp_view,
                                                m_Alb_osqp.data(),
                                                m_Aub_osqp.data(),
                                                static_cast<OSQPInt>(m_A_osqp.rows()),
                                                static_cast<OSQPInt>(m_P_osqp.rows()),
                                                &m_osqp_settings);
            if (exitflag != OSQP_NO_ERROR)
            {
                m_osqp_initialized = false;
                this->m_info.status = qp_status_t::UNSOLVED;
                return this->m_info;
            }
            m_osqp_initialized = true;
        }
        else
        {
            eigen_assert(m_osqp_solver != nullptr);

            osqp_update_data_mat(m_osqp_solver,
                                 m_P_osqp.valuePtr(),
                                 OSQP_NULL,
                                 static_cast<OSQPInt>(m_P_osqp.nonZeros()),
                                 m_A_osqp.valuePtr(),
                                 OSQP_NULL,
                                 static_cast<OSQPInt>(m_A_osqp.nonZeros()));
            osqp_update_data_vec(m_osqp_solver, m_q_osqp.data(), m_Alb_osqp.data(), m_Aub_osqp.data());
        }

        osqp_warm_start(m_osqp_solver, this->m_x_osqp.data(), m_lam_osqp.data());
        osqp_solve(m_osqp_solver);

        Eigen::Map<Eigen::Vector<OSQPFloat, -1>> primal_solution(m_osqp_solver->solution->x, m_P_osqp.rows());
        Eigen::Map<Eigen::Vector<OSQPFloat, -1>> dual_solution(m_osqp_solver->solution->y, m_A_osqp.rows());

        this->m_x = primal_solution(Eigen::seqN(0, this->m_n));
        if (this->m_settings.elastic_mode) {
            this->m_elastic_var = primal_solution(this->m_n);
        }
        this->m_lam = dual_solution(Eigen::seqN(0, this->m_m));
        // copy box constraints dual variables
        int bound_i = 0;
        for (int i = 0; i < this->m_n; i++)
        {
            if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
            {
                this->m_lam_bounds(i) = dual_solution(this->m_m + bound_i);
                bound_i++;
            } else {
                this->m_lam_bounds(i) = 0;
            }
        }

        this->m_info.iter = static_cast<int>(m_osqp_solver->info->iter);

        // update status
        switch (m_osqp_solver->info->status_val)
        {
            case OSQP_SOLVED:
            case OSQP_SOLVED_INACCURATE:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case OSQP_MAX_ITER_REACHED:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            case OSQP_PRIMAL_INFEASIBLE:
            case OSQP_DUAL_INFEASIBLE:
            case OSQP_PRIMAL_INFEASIBLE_INACCURATE:
            case OSQP_DUAL_INFEASIBLE_INACCURATE:
                this->m_info.status = qp_status_t::INFEASIBLE;
                break;
            case OSQP_UNSOLVED:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
            case OSQP_NON_CVX:
                this->m_info.status = qp_status_t::NON_CONVEX;
                break;
            default:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
        }

        return this->m_info;
    }

private:
    /** construct the data matrices accepted by OSQP */
    EIGEN_STRONG_INLINE void
    construct_osqp_data(const Eigen::SparseMatrix<scalar_t>& H,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                        const Eigen::SparseMatrix<scalar_t>& A,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        constraint_changed_t constraints_type_change = this->parse_constraints_bounds(xlb, xub, Alb, Aub);
        if (constraints_type_change == constraint_changed_t::BOX_ONLY_CHANGE ||
            constraints_type_change == constraint_changed_t::CHANGE)
        {
            // if the types of the box constraints changed
            // we have to reinitialize solver because of sparsity pattern change
            m_osqp_initialized = false;
        }

        construct_osqp_cost(H, f);
        construct_osqp_constraints(xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void
    construct_osqp_cost(const Eigen::SparseMatrix<scalar_t>& H,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f) noexcept
    {
        eigen_assert(H.isCompressed());

        if (!this->settings().reuse_pattern || !m_osqp_initialized)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            m_x_osqp.resize(n_vars);
            m_P_osqp.resize(n_vars, n_vars);
            m_q_osqp.resize(n_vars);

            m_x_osqp.setZero();
            m_x_osqp(Eigen::seqN(0, this->m_n)) = this->m_x;

            // copy upper triangular matrix
            for (int i = 0; i < H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(H, i); it; ++it)
                {
                    if (it.row() <= it.col())
                    {
                        m_P_osqp.coeffRef(it.row(), it.col()) = it.value();
                    }
                }
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_P_osqp.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_q_osqp(this->m_n) = this->m_settings.elastic_weight_l1;
            }

            m_P_osqp.makeCompressed();
        }
        else
        {
            m_x_osqp(Eigen::seqN(0, this->m_n)) = this->m_x;

            eigen_assert(m_P_osqp.isCompressed());

            // copy H to the upper left block of m_P_osqp
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_H_tri = m_P_osqp.outerIndexPtr()[col + 1] - m_P_osqp.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(H.valuePtr() + H.outerIndexPtr()[col], inner_nnz_H_tri, m_P_osqp, col, 0);
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_P_osqp.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_q_osqp(this->m_n) = this->m_settings.elastic_weight_l1;
            }
        }

        m_q_osqp(Eigen::seqN(0, this->m_n)) = f;

        eigen_assert(m_P_osqp.isCompressed());
    }

    EIGEN_STRONG_INLINE void
    construct_osqp_constraints(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                               const Eigen::SparseMatrix<scalar_t>& A,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert(A.isCompressed());

        if (!this->settings().reuse_pattern || !m_osqp_initialized)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            // keep track of how many nnz we need per column
            // we start by copying the nnz's of A
            Eigen::VectorXi A_osqp_nnz(n_vars);
            A_osqp_nnz.setZero();
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                A_osqp_nnz(col) = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
            }

            // add nnz's for box constraints
            int num_box_constraints = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    num_box_constraints++;
                    // add one nnz for the box constraint i
                    A_osqp_nnz(i) += 1;
                }
            }

            // add nnz's for slack constraints
            if (this->m_settings.elastic_mode)
            {
                // need box constraint on slack in [0, 1]
                num_box_constraints++;
                // for each non-linear constraint we have to add slacks + 1 for slack box constraints
                A_osqp_nnz(this->m_n) = this->m_m + 1;
            }

            m_Alb_osqp.resize(this->m_m + num_box_constraints);
            m_Aub_osqp.resize(this->m_m + num_box_constraints);
            m_lam_osqp.resize(this->m_m + num_box_constraints);

            // copy A bounds
            m_Alb_osqp(Eigen::seqN(0, this->m_m)) = Alb;
            m_Aub_osqp(Eigen::seqN(0, this->m_m)) = Aub;
            // copy A dual variables
            m_lam_osqp(Eigen::seqN(0, this->m_m)) = this->m_lam;

            // copy box bounds
            int bound_i = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Alb_osqp(this->m_m + bound_i) = xlb(i);
                    m_Aub_osqp(this->m_m + bound_i) = xub(i);
                    m_lam_osqp(this->m_m + bound_i) = this->m_lam_bounds(i);
                    bound_i++;
                }
            }
            // set slack constraints
            if (this->m_settings.elastic_mode)
            {
                m_Alb_osqp(this->m_m + bound_i) = scalar_t(0);
                m_Aub_osqp(this->m_m + bound_i) = scalar_t(1);
                m_lam_osqp(this->m_m + bound_i) = scalar_t(0);
            }

            m_A_osqp.resize(this->m_m + num_box_constraints, n_vars);
            m_A_osqp.reserve(A_osqp_nnz);

            // copy A to the upper left block of m_A_osqp
            for (int i = 0; i < A.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    m_A_osqp.coeffRef(it.row(), it.col()) = it.value();
                }
            }

            // create entries for the box constraints
            bound_i = 0;
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                if (this->m_box_constraint_type[col] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_A_osqp.coeffRef(this->m_m + bound_i++, col) = 1;
                }
            }

            // create entries for the slack constraints
            if (this->m_settings.elastic_mode)
            {
                for (int i = 0; i < this->m_m; i++)
                {
                    scalar_t slack_coeff = 0;

                    if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR || Alb(i) > scalar_t(0))
                    {
                        slack_coeff = Alb(i);
                    }
                    else if (Aub(i) < scalar_t(0))
                    {
                        slack_coeff = Aub(i);
                    }

                    m_A_osqp.coeffRef(i, this->m_n) = slack_coeff;
                }

                // entry for slack box constraints
                m_A_osqp.coeffRef(this->m_m + bound_i++, this->m_n) = 1;
            }

            m_A_osqp.makeCompressed();
        }
        else
        {
            // copy A bounds
            m_Alb_osqp(Eigen::seqN(0, this->m_m)) = Alb;
            m_Aub_osqp(Eigen::seqN(0, this->m_m)) = Aub;
            // copy A dual variables
            m_lam_osqp(Eigen::seqN(0, this->m_m)) = this->m_lam;

            // copy variable bounds
            int bound_i = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Alb_osqp(this->m_m + bound_i) = xlb(i);
                    m_Aub_osqp(this->m_m + bound_i) = xub(i);
                    m_lam_osqp(this->m_m + bound_i) = this->m_lam_bounds(i);
                    bound_i++;
                }
            }

            // copy A to the upper left block of m_A_osqp
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_A = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(A.valuePtr() + A.outerIndexPtr()[col], inner_nnz_A, m_A_osqp, col, 0);
            }

            // copy slack coefficients
            if (this->m_settings.elastic_mode)
            {
                for (int i = 0; i < this->m_m; i++)
                {
                    scalar_t slack_coeff = 0;

                    if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR || Alb(i) > scalar_t(0))
                    {
                        slack_coeff = Alb(i);
                    }
                    else if (Aub(i) < scalar_t(0))
                    {
                        slack_coeff = Aub(i);
                    }

                    *(m_A_osqp.valuePtr() + m_A_osqp.outerIndexPtr()[this->m_n] + i) = slack_coeff;
                }
            }

            // all other entries of m_A_osqp are unchanged
        }

        eigen_assert(m_A_osqp.isCompressed());
    }

    void set_osqp_settings() noexcept
    {
        m_osqp_settings.eps_rel = this->m_settings.eps_rel;
        m_osqp_settings.eps_abs = this->m_settings.eps_abs;
        m_osqp_settings.max_iter = static_cast<OSQPInt>(this->m_settings.max_iter);
        m_osqp_settings.warm_starting = true;
        m_osqp_settings.verbose = this->m_settings.verbose;

        if (m_osqp_solver != nullptr) {
            osqp_update_settings(m_osqp_solver, &m_osqp_settings);
        }
    }

};

} // laopt namespace

#endif // LAOPT_OSQP_INTERFACE_HPP
