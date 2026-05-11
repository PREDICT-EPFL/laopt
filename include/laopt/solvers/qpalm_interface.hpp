#ifndef LAOPT_QPALM_INTERFACE_HPP
#define LAOPT_QPALM_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "qpalm.h"

namespace laopt
{

template<typename Scalar = double>
class QPALMSolver : public QPBase<QPALMSolver<Scalar>, Scalar>
{
public:
    using Base = QPBase<QPALMSolver<Scalar>, Scalar>;
    using scalar_t = typename Base::scalar_t;

private:
    using constraint_t = typename Base::constraint_t;
    using constraint_changed_t = typename Base::constraint_changed_t;

    QPALMWorkspace * m_qpalm_workspace;
    QPALMSettings * m_qpalm_settings;
    QPALMData * m_qpalm_data;
    bool m_qpalm_initialized;

    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, c_int> m_Q_qpalm;
    Eigen::VectorX<scalar_t> m_q_qpalm;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, c_int> m_A_qpalm;
    Eigen::VectorX<scalar_t> m_Alb_qpalm;
    Eigen::VectorX<scalar_t> m_Aub_qpalm;
    Eigen::VectorX<scalar_t> m_x_qpalm;
    Eigen::VectorX<scalar_t> m_lam_qpalm;

public:
    QPALMSolver(int n, int m) : Base(n, m)
    {
        this->m_settings.max_iter = 10000;

        m_qpalm_workspace = nullptr;
        m_qpalm_settings = (QPALMSettings*) qpalm_malloc(sizeof(QPALMSettings));
        m_qpalm_data = (QPALMData*) qpalm_malloc(sizeof(QPALMData));
        m_qpalm_initialized = false;

        if (m_qpalm_data) {
            m_qpalm_data->Q = (solver_sparse*) qpalm_malloc(sizeof(solver_sparse));
            m_qpalm_data->Q->values = 1;
            m_qpalm_data->Q->symmetry = UPPER;
            m_qpalm_data->Q->nz = nullptr;
            m_qpalm_data->A = (solver_sparse*) qpalm_malloc(sizeof(solver_sparse));
            m_qpalm_data->A->values = 1;
            m_qpalm_data->A->symmetry = UNSYMMETRIC;
            m_qpalm_data->A->nz = nullptr;
        }
        if (m_qpalm_settings) {
            qpalm_set_default_settings(m_qpalm_settings);
        }

        set_qpalm_settings();
    }

    ~QPALMSolver()
    {
        if (m_qpalm_workspace != nullptr) {
            qpalm_cleanup(m_qpalm_workspace);
        }
        if (m_qpalm_data) {
            if (m_qpalm_data->Q) qpalm_free(m_qpalm_data->Q);
            if (m_qpalm_data->A) qpalm_free(m_qpalm_data->A);
            qpalm_free(m_qpalm_data);
        }
        if (m_qpalm_settings) {
            qpalm_free(m_qpalm_settings);
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
        set_qpalm_settings();

        construct_qpalm_data(H, f, xlb, xub, A, Alb, Aub);

        m_qpalm_data->n = m_Q_qpalm.rows();
        m_qpalm_data->m = m_A_qpalm.rows();

        m_qpalm_data->Q->nrow = m_Q_qpalm.rows();
        m_qpalm_data->Q->ncol = m_Q_qpalm.cols();
        m_qpalm_data->Q->nzmax = m_Q_qpalm.nonZeros();
        m_qpalm_data->Q->x = m_Q_qpalm.valuePtr();
        m_qpalm_data->Q->i = m_Q_qpalm.innerIndexPtr();
        m_qpalm_data->Q->p = m_Q_qpalm.outerIndexPtr();

        m_qpalm_data->q = m_q_qpalm.data();
        m_qpalm_data->c = 0;

        m_qpalm_data->A->nrow = m_A_qpalm.rows();
        m_qpalm_data->A->ncol = m_A_qpalm.cols();
        m_qpalm_data->A->nzmax = m_A_qpalm.nonZeros();
        m_qpalm_data->A->x = m_A_qpalm.valuePtr();
        m_qpalm_data->A->i = m_A_qpalm.innerIndexPtr();
        m_qpalm_data->A->p = m_A_qpalm.outerIndexPtr();

        m_qpalm_data->bmin = m_Alb_qpalm.data();
        m_qpalm_data->bmax = m_Aub_qpalm.data();

        if (!this->m_settings.reuse_pattern || !m_qpalm_initialized)
        {
            if (m_qpalm_workspace != nullptr) {
                qpalm_cleanup(m_qpalm_workspace);
            }

            m_qpalm_workspace = qpalm_setup(m_qpalm_data, m_qpalm_settings);
            m_qpalm_initialized = true;
        }
        else
        {
            eigen_assert(m_qpalm_workspace != nullptr);

            qpalm_update_Q_A(m_qpalm_workspace, m_qpalm_data->Q->x, m_qpalm_data->A->x);
            qpalm_update_q(m_qpalm_workspace, m_qpalm_data->q);
            qpalm_update_bounds(m_qpalm_workspace, m_qpalm_data->bmin, m_qpalm_data->bmax);
        }

        qpalm_warm_start(m_qpalm_workspace, this->m_x_qpalm.data(), m_lam_qpalm.data());
        qpalm_solve(m_qpalm_workspace);

        Eigen::Map<Eigen::Vector<c_float, -1>> primal_solution(m_qpalm_workspace->solution->x, m_qpalm_workspace->data->n);
        Eigen::Map<Eigen::Vector<c_float, -1>> dual_solution(m_qpalm_workspace->solution->y, m_qpalm_workspace->data->m);

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

        this->m_info.iter = m_qpalm_workspace->info->iter;

        // update status
        switch (m_qpalm_workspace->info->status_val)
        {
            case QPALM_SOLVED:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case QPALM_MAX_ITER_REACHED:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            case QPALM_PRIMAL_INFEASIBLE:
            case QPALM_DUAL_INFEASIBLE:
                this->m_info.status = qp_status_t::INFEASIBLE;
                break;
            case QPALM_UNSOLVED:
            default:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
        }

        return this->m_info;
    }

private:
    /** construct the data matrices accepted by OSQP */
    EIGEN_STRONG_INLINE void
    construct_qpalm_data(const Eigen::SparseMatrix<scalar_t>& H,
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
            m_qpalm_initialized = false;
        }

        construct_qpalm_cost(H, f);
        construct_qpalm_constraints(xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void
    construct_qpalm_cost(const Eigen::SparseMatrix<scalar_t>& H,
                        const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f) noexcept
    {
        eigen_assert(H.isCompressed());

        if (!this->settings().reuse_pattern || !m_qpalm_initialized)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            m_x_qpalm.resize(n_vars);
            m_Q_qpalm.resize(n_vars, n_vars);
            m_q_qpalm.resize(n_vars);

            m_x_qpalm.setZero();
            m_x_qpalm(Eigen::seqN(0, this->m_n)) = this->m_x;

            // copy upper triangular matrix
            for (int i = 0; i < H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(H, i); it; ++it)
                {
                    if (it.row() <= it.col())
                    {
                        m_Q_qpalm.coeffRef(it.row(), it.col()) = it.value();
                    }
                }
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_Q_qpalm.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_q_qpalm(this->m_n) = this->m_settings.elastic_weight_l1;
            }

            m_Q_qpalm.makeCompressed();
        }
        else
        {
            m_x_qpalm(Eigen::seqN(0, this->m_n)) = this->m_x;

            eigen_assert(m_Q_qpalm.isCompressed());

            // copy H to the upper left block of m_Q_qpalm
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_H_tri = m_Q_qpalm.outerIndexPtr()[col + 1] - m_Q_qpalm.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(H.valuePtr() + H.outerIndexPtr()[col], inner_nnz_H_tri, m_Q_qpalm, col, 0);
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_Q_qpalm.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_q_qpalm(this->m_n) = this->m_settings.elastic_weight_l1;
            }
        }

        m_q_qpalm(Eigen::seqN(0, this->m_n)) = f;

        eigen_assert(m_Q_qpalm.isCompressed());
    }

    EIGEN_STRONG_INLINE void
    construct_qpalm_constraints(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                               const Eigen::SparseMatrix<scalar_t>& A,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert(A.isCompressed());

        if (!this->settings().reuse_pattern || !m_qpalm_initialized)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            // keep track of how many nnz we need per column
            // we start by copying the nnz's of A
            Eigen::VectorXi A_qpalm_nnz(n_vars);
            A_qpalm_nnz.setZero();
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                A_qpalm_nnz(col) = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
            }

            // add nnz's for box constraints
            int num_box_constraints = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    num_box_constraints++;
                    // add one nnz for the box constraint i
                    A_qpalm_nnz(i) += 1;
                }
            }

            // add nnz's for slack constraints
            if (this->m_settings.elastic_mode)
            {
                // need box constraint on slack in [0, 1]
                num_box_constraints++;
                // for each non-linear constraint we have to add slacks + 1 for slack box constraints
                A_qpalm_nnz(this->m_n) = this->m_m + 1;
            }

            m_Alb_qpalm.resize(this->m_m + num_box_constraints);
            m_Aub_qpalm.resize(this->m_m + num_box_constraints);
            m_lam_qpalm.resize(this->m_m + num_box_constraints);

            // copy A bounds
            m_Alb_qpalm(Eigen::seqN(0, this->m_m)) = Alb;
            m_Aub_qpalm(Eigen::seqN(0, this->m_m)) = Aub;
            // copy A dual variables
            m_lam_qpalm(Eigen::seqN(0, this->m_m)) = this->m_lam;

            // copy box bounds
            int bound_i = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Alb_qpalm(this->m_m + bound_i) = xlb(i);
                    m_Aub_qpalm(this->m_m + bound_i) = xub(i);
                    m_lam_qpalm(this->m_m + bound_i) = this->m_lam_bounds(i);
                    bound_i++;
                }
            }
            // set slack constraints
            if (this->m_settings.elastic_mode)
            {
                m_Alb_qpalm(this->m_m + bound_i) = scalar_t(0);
                m_Aub_qpalm(this->m_m + bound_i) = scalar_t(1);
                m_lam_qpalm(this->m_m + bound_i) = scalar_t(0);
            }

            m_A_qpalm.resize(this->m_m + num_box_constraints, n_vars);
            m_A_qpalm.reserve(A_qpalm_nnz);

            // copy A to the upper left block of m_A_qpalm
            for (int i = 0; i < A.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    m_A_qpalm.coeffRef(it.row(), it.col()) = it.value();
                }
            }

            // create entries for the box constraints
            bound_i = 0;
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                if (this->m_box_constraint_type[col] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_A_qpalm.coeffRef(this->m_m + bound_i++, col) = 1;
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

                    m_A_qpalm.coeffRef(i, this->m_n) = slack_coeff;
                }

                // entry for slack box constraints
                m_A_qpalm.coeffRef(this->m_m + bound_i++, this->m_n) = 1;
            }

            m_A_qpalm.makeCompressed();
        }
        else
        {
            // copy A bounds
            m_Alb_qpalm(Eigen::seqN(0, this->m_m)) = Alb;
            m_Aub_qpalm(Eigen::seqN(0, this->m_m)) = Aub;
            // copy A dual variables
            m_lam_qpalm(Eigen::seqN(0, this->m_m)) = this->m_lam;

            // copy variable bounds
            int bound_i = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Alb_qpalm(this->m_m + bound_i) = xlb(i);
                    m_Aub_qpalm(this->m_m + bound_i) = xub(i);
                    m_lam_qpalm(this->m_m + bound_i) = this->m_lam_bounds(i);
                    bound_i++;
                }
            }

            // copy A to the upper left block of m_A_qpalm
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_A = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(A.valuePtr() + A.outerIndexPtr()[col], inner_nnz_A, m_A_qpalm, col, 0);
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

                    *(m_A_qpalm.valuePtr() + m_A_qpalm.outerIndexPtr()[this->m_n] + i) = slack_coeff;
                }
            }

            // all other entries of m_A_qpalm are unchanged
        }

        eigen_assert(m_A_qpalm.isCompressed());
    }

    void set_qpalm_settings() noexcept
    {
        m_qpalm_settings->eps_rel = this->m_settings.eps_rel;
        m_qpalm_settings->eps_abs = this->m_settings.eps_abs;
        m_qpalm_settings->max_iter = this->m_settings.max_iter;
        m_qpalm_settings->warm_start = true;
        m_qpalm_settings->verbose = this->m_settings.verbose;

        if (m_qpalm_workspace != nullptr) {
            qpalm_update_settings(m_qpalm_workspace, m_qpalm_settings);
        }
    }

};

} // laopt namespace

#endif // LAOPT_QPALM_INTERFACE_HPP
