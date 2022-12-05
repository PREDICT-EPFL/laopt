#ifndef LAOPT_OSQP_INTERFACE_HPP
#define LAOPT_OSQP_INTERFACE_HPP

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

private:
    using constraint_t = typename Base::constraint_t;

    OSQPWorkspace* m_osqp_workspace;
    OSQPSettings* m_osqp_settings;
    OSQPData* m_osqp_data;
    bool m_osqp_initialized;

    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, c_int> m_H_osqp;
    Eigen::VectorX<scalar_t> m_f_osqp;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, c_int> m_A_osqp;
    Eigen::VectorX<scalar_t> m_Alb_osqp;
    Eigen::VectorX<scalar_t> m_Aub_osqp;
    Eigen::VectorX<scalar_t> m_lam_osqp;

public:
    OSQPSolver(int n, int m) : Base(n, m), m_f_osqp(n)
    {
        m_osqp_workspace = nullptr;
        m_osqp_settings = (OSQPSettings*) c_malloc(sizeof(OSQPSettings));
        m_osqp_data = (OSQPData*) c_malloc(sizeof(OSQPData));
        m_osqp_initialized = false;

        if (m_osqp_data) {
            m_osqp_data->P = (csc*) c_malloc(sizeof(csc));
            m_osqp_data->P->nz = -1;
            m_osqp_data->A = (csc*) c_malloc(sizeof(csc));
            m_osqp_data->A->nz = -1;
        }
        if (m_osqp_settings) {
            osqp_set_default_settings(m_osqp_settings);
        }

        set_osqp_settings();
    }

    ~OSQPSolver()
    {
        if (m_osqp_initialized) {
            osqp_cleanup(m_osqp_workspace);
        }
        if (m_osqp_data) {
            if (m_osqp_data->P) c_free(m_osqp_data->P);
            if (m_osqp_data->A) c_free(m_osqp_data->A);
            c_free(m_osqp_data);
        }
        if (m_osqp_settings) {
            c_free(m_osqp_settings);
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

        m_osqp_data->n = this->m_n;
        m_osqp_data->m = m_A_osqp.rows();

        m_osqp_data->P->m = m_H_osqp.rows();
        m_osqp_data->P->n = m_H_osqp.cols();
        m_osqp_data->P->nzmax = m_H_osqp.nonZeros();
        m_osqp_data->P->x = m_H_osqp.valuePtr();
        m_osqp_data->P->i = m_H_osqp.innerIndexPtr();
        m_osqp_data->P->p = m_H_osqp.outerIndexPtr();

        m_osqp_data->q = m_f_osqp.data();

        m_osqp_data->A->m = m_A_osqp.rows();
        m_osqp_data->A->n = m_A_osqp.cols();
        m_osqp_data->A->nzmax = m_A_osqp.nonZeros();
        m_osqp_data->A->x = m_A_osqp.valuePtr();
        m_osqp_data->A->i = m_A_osqp.innerIndexPtr();
        m_osqp_data->A->p = m_A_osqp.outerIndexPtr();

        m_osqp_data->l = m_Alb_osqp.data();
        m_osqp_data->u = m_Aub_osqp.data();

        if (!this->m_settings.reuse_pattern)
        {
            if (m_osqp_initialized) {
                osqp_cleanup(m_osqp_workspace);
            }

            osqp_setup(&m_osqp_workspace, m_osqp_data, m_osqp_settings);
            m_osqp_initialized = true;

            osqp_warm_start(m_osqp_workspace, this->m_x.data(), m_lam_osqp.data());
            osqp_solve(m_osqp_workspace);
        }
        else
        {
            eigen_assert(m_osqp_initialized);

            osqp_update_P_A(m_osqp_workspace, m_osqp_data->P->x, OSQP_NULL, m_osqp_data->P->nzmax,
                                              m_osqp_data->A->x, OSQP_NULL, m_osqp_data->A->nzmax);
            osqp_update_lin_cost(m_osqp_workspace, m_osqp_data->q);
            osqp_update_bounds(m_osqp_workspace, m_osqp_data->l, m_osqp_data->u);

            osqp_warm_start(m_osqp_workspace, this->m_x.data(), m_lam_osqp.data());
            osqp_solve(m_osqp_workspace);
        }

        Eigen::Map<Eigen::Vector<c_float, -1>> primal_solution(m_osqp_workspace->solution->x, m_osqp_workspace->data->n);
        Eigen::Map<Eigen::Vector<c_float, -1>> dual_solution(m_osqp_workspace->solution->y, m_osqp_workspace->data->m);

        this->m_x = primal_solution;
        this->m_lam = dual_solution(Eigen::seqN(0, this->m_m));
        // copy box constraints dual variables
        int bound_i = 0;
        for (int i = 0; i < this->m_n; i++)
        {
            if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
            {
                this->m_lam_bounds(i) = dual_solution(this->m_m + bound_i);
                bound_i++;
            }
        }

        this->m_info.iter = m_osqp_workspace->info->iter;

        // update status
        switch (m_osqp_workspace->info->status_val)
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
        construct_osqp_cost(H, f);
        construct_osqp_constraints(xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void
    construct_osqp_cost(const Eigen::SparseMatrix<scalar_t>& H,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f) noexcept
    {
        m_f_osqp = f;

        eigen_assert(H.isCompressed());

        if (!this->settings().reuse_pattern)
        {
            m_H_osqp.resize(this->m_n, this->m_n);

            // copy upper triangular matrix
            for (int i = 0; i < H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(H, i); it; ++it)
                {
                    if (it.row() <= it.col())
                    {
                        m_H_osqp.coeffRef(it.row(), it.col()) = it.value();
                    }
                }
            }
            m_H_osqp.makeCompressed();
        }
        else
        {
            eigen_assert(m_H_osqp.isCompressed());

            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_H_tri = m_H_osqp.outerIndexPtr()[col + 1] - m_H_osqp.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(H.valuePtr() + H.outerIndexPtr()[col], inner_nnz_H_tri, m_H_osqp, col, 0);
            }
        }

        eigen_assert(m_H_osqp.isCompressed());
    }

    EIGEN_STRONG_INLINE void
    construct_osqp_constraints(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                               const Eigen::SparseMatrix<scalar_t>& A,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert(A.isCompressed());

        if (!this->settings().reuse_pattern)
        {
            this->parse_constraints_bounds(xlb, xub, Alb, Aub);

            // keep track of how many nnz we need per column
            // we start by copying the nnz's of A
            Eigen::VectorXi A_osqp_nnz(this->m_n);
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                A_osqp_nnz(col) = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
            }

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

            m_Alb_osqp.resize(this->m_m + num_box_constraints);
            m_Aub_osqp.resize(this->m_m + num_box_constraints);
            m_lam_osqp.resize(this->m_m + num_box_constraints);

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

            m_A_osqp.resize(this->m_m + num_box_constraints, this->m_n);
            m_A_osqp.reserve(A_osqp_nnz);

            // copy A to the head of m_A_osqp
            for (int i = 0; i < A.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    m_A_osqp.coeffRef(it.row(), it.col()) = it.value();
                }
            }

            // copy entries for the box constraints
            bound_i = 0;
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                if (this->m_box_constraint_type[col] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_A_osqp.coeffRef(this->m_m + bound_i, col) = 1;
                    bound_i++;
                }
            }

            m_A_osqp.makeCompressed();
        }
        else
        {
            // we assume constraint bounds did not change

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

            // copy A to the head of m_A_osqp
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_A = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(A.valuePtr() + A.outerIndexPtr()[col], inner_nnz_A, m_A_osqp, col, 0);
            }

            // entries for box constraints of m_A_osqp are unchanged
        }

        eigen_assert(m_A_osqp.isCompressed());
    }

    void set_osqp_settings() noexcept
    {
        m_osqp_settings->eps_rel = this->m_settings.eps_rel;
        m_osqp_settings->eps_abs = this->m_settings.eps_abs;
        m_osqp_settings->max_iter = this->m_settings.max_iter;
        m_osqp_settings->warm_start = true;
        m_osqp_settings->verbose = this->m_settings.verbose;

        if (m_osqp_initialized) {
            osqp_update_eps_rel(m_osqp_workspace, this->m_settings.eps_rel);
            osqp_update_eps_abs(m_osqp_workspace, this->m_settings.eps_abs);
            osqp_update_max_iter(m_osqp_workspace, this->m_settings.max_iter);
            osqp_update_verbose(m_osqp_workspace, this->m_settings.verbose);
        }
    }

};

} // laopt namespace

#endif // LAOPT_OSQP_INTERFACE_HPP
