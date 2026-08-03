#ifndef LAOPT_QPSWIFT_INTERFACE_HPP
#define LAOPT_QPSWIFT_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "qpSWIFT.h"

// TODO: Handle conflicting macros better
#undef Int

namespace laopt
{

template<typename Scalar = double>
class QPSwiftSolver : public QPBase<QPSwiftSolver<Scalar>, Scalar>
{
public:
    using Base = QPBase<QPSwiftSolver<Scalar>, Scalar>;
    using scalar_t = typename Base::scalar_t;

private:
    using constraint_t = typename Base::constraint_t;
    using constraint_changed_t = typename Base::constraint_changed_t;

    QP* m_qpswift_solver;
    bool m_qpswift_initialized;

    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, qp_int> m_P_qpswift;
    Eigen::VectorX<scalar_t> m_c_qpswift;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, qp_int> m_A_qpswift;
    Eigen::VectorX<scalar_t> m_b_qpswift;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, qp_int> m_G_qpswift;
    Eigen::VectorX<scalar_t> m_h_qpswift;

    Eigen::VectorX<int> m_A_to_qpswift_map; // maps row in A to row in A_qpswift or C_qpswift

public:
    QPSwiftSolver(int n, int m) :
        Base(n, m),
        m_qpswift_solver(nullptr),
        m_qpswift_initialized(false),
        m_A_to_qpswift_map(m)
    {
        this->m_settings.max_iter = 50;
    }

    ~QPSwiftSolver()
    {
        if (m_qpswift_solver != nullptr) {
            QP_CLEANUP(m_qpswift_solver);
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
        construct_qpswift_data(H, f, xlb, xub, A, Alb, Aub);

        if (!this->m_settings.reuse_pattern || !m_qpswift_initialized)
        {
            if (m_qpswift_solver != nullptr)
            {
                QP_CLEANUP(m_qpswift_solver);
            }

            m_qpswift_solver = QP_SETUP(m_P_qpswift.rows(), m_G_qpswift.rows(), m_A_qpswift.rows(),
                                        m_P_qpswift.outerIndexPtr(), m_P_qpswift.innerIndexPtr(), m_P_qpswift.valuePtr(),
                                        m_A_qpswift.outerIndexPtr(), m_A_qpswift.innerIndexPtr(), m_A_qpswift.valuePtr(),
                                        m_G_qpswift.outerIndexPtr(), m_G_qpswift.innerIndexPtr(), m_G_qpswift.valuePtr(),
                                        m_c_qpswift.data(), m_h_qpswift.data(), m_b_qpswift.data(), 0.0, nullptr);
            m_qpswift_initialized = true;
        }
        else
        {
            eigen_assert(m_qpswift_solver != nullptr);

            // there is no update function provided in QPSwift, but we can imitate what is done in QP_SETUP
            // this is very hacky code and could break with future versions of QPSwift
            // this actually should be integrated into QPSwift directly
            qp_timer tsetup;
            tic(&tsetup);

            qp_int n = m_qpswift_solver->n;
            qp_int m = m_qpswift_solver->m;
            qp_int p = m_qpswift_solver->p;

            // Initialize Settings
            m_qpswift_solver->options->abstol = ABSTOL;
            m_qpswift_solver->options->reltol = RELTOL;
            m_qpswift_solver->options->maxit = MAXIT;
            m_qpswift_solver->options->sigma = SIGMA;
            m_qpswift_solver->options->verbose = VERBOSE;

            // Initialize Stats
            m_qpswift_solver->stats->Flag = QP_FATAL;
            m_qpswift_solver->stats->IterationCount = 0;
            m_qpswift_solver->stats->alpha_p = 0.0;
            m_qpswift_solver->stats->alpha_d = 0.0;

            // Initialize Matrices
            SparseMatrixSetup(n, n, m_P_qpswift.nonZeros(),
                              m_P_qpswift.outerIndexPtr(), m_P_qpswift.innerIndexPtr(), m_P_qpswift.valuePtr(),
                              m_qpswift_solver->P);
            SparseMatrixSetup(p, n, m_A_qpswift.nonZeros(),
                              m_A_qpswift.outerIndexPtr(), m_A_qpswift.innerIndexPtr(), m_A_qpswift.valuePtr(),
                              m_qpswift_solver->A);
            SparseMatrixSetup(m, n, m_G_qpswift.nonZeros(),
                              m_G_qpswift.outerIndexPtr(), m_G_qpswift.innerIndexPtr(), m_G_qpswift.valuePtr(),
                              m_qpswift_solver->G);
            m_qpswift_solver->c = m_c_qpswift.data();
            m_qpswift_solver->h = m_h_qpswift.data();
            m_qpswift_solver->b = m_b_qpswift.data();

            // Transpose of Matrices
            SparseMatrixTranspose(m_qpswift_solver->A, m_qpswift_solver->At);
            SparseMatrixTranspose(m_qpswift_solver->G, m_qpswift_solver->Gt);

            // Form full KKT Matrix
            formkktmatrix_full(m_qpswift_solver->P, m_qpswift_solver->G, m_qpswift_solver->A,
                               m_qpswift_solver->Gt, m_qpswift_solver->At, m_qpswift_solver->kkt->kktmatrix);

            // Initialize KKT
            kkt* kkt = m_qpswift_solver->kkt;
            LDL_numeric(kkt->kktmatrix->n, kkt->kktmatrix->jc, kkt->kktmatrix->ir, kkt->kktmatrix->pr, kkt->Lp, kkt->Parent, kkt->Lnz, kkt->Li, kkt->Lx, kkt->D, kkt->Y, kkt->Pattern, kkt->Flag, kkt->P, kkt->Pinv);
            // Transpose of L
            Transpose_Row_Count(kkt->kktmatrix->n, kkt->kktmatrix->n, kkt->Li, kkt->Lp, kkt->Lti, kkt->Ltp);

            // From KKT right hand side
            for (qp_int i = 0; i < n; i++)
            {
                kkt->b[i] = -m_qpswift_solver->c[i];
            }
            for (qp_int i = n; i < n + p; i++)
            {
                kkt->b[i] = m_qpswift_solver->b[i - n];
            }
            for (qp_int i = n + p; i < n + p + m; i++)
            {
                kkt->b[i] = m_qpswift_solver->h[i - n - p];
            }

            // Solve Ax=b, overwriting b with the solution x
            LDL_perm(kkt->kktmatrix->n, m_qpswift_solver->delta, kkt->b, kkt->P);
            LDL_lsolve(kkt->kktmatrix->n, m_qpswift_solver->delta, kkt->Lp, kkt->Li, kkt->Lx);
            LDL_dsolve(kkt->kktmatrix->n, m_qpswift_solver->delta, kkt->D);
            LDL_ltsolve(kkt->kktmatrix->n, m_qpswift_solver->delta, kkt->Lp, kkt->Li, kkt->Lx);
            LDL_permt(kkt->kktmatrix->n, kkt->b, m_qpswift_solver->delta, kkt->P);

            for (qp_int i = 0; i < n; i++)
            {
                m_qpswift_solver->x[i] = kkt->b[i];
            }

            for (qp_int i = n; i < n + p; i++)
            {
                m_qpswift_solver->y[i - n] = kkt->b[i];
            }

            qp_real* z_inter = (qp_real*) MALLOC(m_qpswift_solver->m * sizeof(qp_real));

            // Calculate z_inter = Gx - h in two steps
            // Calculate z_inter = -Gx
            SparseMatrixMultiply(m_qpswift_solver->G, m_qpswift_solver->x, z_inter, 1);
            // Add h to z
            updatevariables(z_inter, m_qpswift_solver->h, 1, m);

            /* find alpha_p and alpha_d */
            qp_real alpha_p, alpha_d;
            findminmax(z_inter, m_qpswift_solver->m, &alpha_p, &alpha_d);
            alpha_p = -alpha_p;
            if (alpha_p < 0)
            {
                for (qp_int i = 0; i < m; i++)
                {
                    m_qpswift_solver->s[i] = z_inter[i];
                }
            }
            else
            {
                for (qp_int i = 0; i < m; i++)
                {
                    m_qpswift_solver->s[i] = z_inter[i] + (1 + alpha_p);
                }
            }

            if (alpha_d < 0)
            {
                for (qp_int i = 0; i < m; i++)
                {
                    m_qpswift_solver->z[i] = -z_inter[i];
                }
            }
            else
            {
                for (qp_int i = 0; i < m; i++)
                {
                    m_qpswift_solver->z[i] = -z_inter[i] + (1 + alpha_d);
                }
            }

            FREE(z_inter);

            m_qpswift_solver->stats->ldl_numeric = 0.0;
            m_qpswift_solver->stats->tsetup = toc(&tsetup);
        }

        set_qpswift_settings();
        QP_SOLVE(m_qpswift_solver);

        Eigen::Map<Eigen::Vector<qp_real, -1>> primal_solution(m_qpswift_solver->x, m_qpswift_solver->n);
        Eigen::Map<Eigen::Vector<qp_real, -1>> dual_eq_solution(m_qpswift_solver->y, m_qpswift_solver->p);
        Eigen::Map<Eigen::Vector<qp_real, -1>> dual_ineq_solution(m_qpswift_solver->z, m_qpswift_solver->m);

        this->m_x = primal_solution(Eigen::seqN(0, this->m_n));

        if (this->m_settings.elastic_mode) {
            this->m_elastic_var = primal_solution(this->m_n);
        }

        // copy eq and ineq dual variables
        int eq_bound_i = 0;
        int ineq_bound_i = 0;
        for (int i = 0; i < this->m_m; i++)
        {
            if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
            {
                this->m_lam(i) = dual_eq_solution(eq_bound_i++);
            }
            else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam(i) = -dual_ineq_solution(ineq_bound_i++);
            }
            else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam(i) = dual_ineq_solution(ineq_bound_i++);
            }
            else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam(i) = -dual_ineq_solution(ineq_bound_i++);
                this->m_lam(i) += dual_ineq_solution(ineq_bound_i++);
            }
            else
            {
                this->m_lam(i) = 0;
            }
        }
        // copy box constraints dual variables
        for (int i = 0; i < this->m_n; i++)
        {
            if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
            {
                this->m_lam_bounds(i) = dual_eq_solution(eq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam_bounds(i) = -dual_ineq_solution(ineq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam_bounds(i) = dual_ineq_solution(ineq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam_bounds(i) = -dual_ineq_solution(ineq_bound_i++);
                this->m_lam_bounds(i) += dual_ineq_solution(ineq_bound_i++);
            }
            else
            {
                this->m_lam_bounds(i) = 0;
            }
        }

        this->m_info.iter = m_qpswift_solver->stats->IterationCount;

        // update status
        switch (m_qpswift_solver->stats->Flag)
        {
            case QP_OPTIMAL:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case QP_MAXIT:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            case QP_KKTFAIL:
                this->m_info.status = qp_status_t::INFEASIBLE;
                break;
            default:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
        }

        return this->m_info;
    }

private:
    /** Copy from findminmax in Auxilary.c */
    void findminmax(qp_real *z, long n, qp_real *min, qp_real *max)
    {
        min[0] = z[0];
        max[0] = z[0];
        qp_int i;
        for (i = 1; i < n; i++)
        {
            if (z[i] < min[0])
            {
                min[0] = z[i];
            }
            if (z[i] > max[0])
            {
                max[0] = z[i];
            }
        }
    }

    /** construct the data matrices accepted by OSQP */
    EIGEN_STRONG_INLINE void
    construct_qpswift_data(const Eigen::SparseMatrix<scalar_t>& H,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                          const Eigen::SparseMatrix<scalar_t>& A,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        constraint_changed_t constraints_type_change = this->parse_constraints_bounds(xlb, xub, Alb, Aub);
        if (constraints_type_change != constraint_changed_t::NO_CHANGE)
        {
            // if the types of the constraints changed
            // we have to reinitialize solver because of sparsity pattern change
            m_qpswift_initialized = false;
        }

        construct_qpswift_cost(H, f);
        construct_qpswift_constraints(xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void
    construct_qpswift_cost(const Eigen::SparseMatrix<scalar_t>& H,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f) noexcept
    {
        eigen_assert(H.isCompressed());

        if (!this->settings().reuse_pattern || !m_qpswift_initialized)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            Eigen::VectorXi P_qpswift_nnz(n_vars);
            for (int col = 0; col < H.outerSize(); col++)
            {
                P_qpswift_nnz(col) = H.outerIndexPtr()[col + 1] - H.outerIndexPtr()[col];
            }
            if (this->m_settings.elastic_mode)
            {
                P_qpswift_nnz(this->m_n) = 1;
            }

            m_P_qpswift.resize(n_vars, n_vars);
            m_P_qpswift.reserve(P_qpswift_nnz);
            m_c_qpswift.resize(n_vars);

            // copy H
            for (int i = 0; i < H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(H, i); it; ++it)
                {
                    m_P_qpswift.coeffRef(it.row(), it.col()) = it.value();
                }
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_P_qpswift.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_c_qpswift(this->m_n) = this->m_settings.elastic_weight_l1;
            }

            m_P_qpswift.makeCompressed();
        }
        else
        {
            eigen_assert(m_P_qpswift.isCompressed());

            // copy H
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_H_tri = m_P_qpswift.outerIndexPtr()[col + 1] - m_P_qpswift.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(H.valuePtr() + H.outerIndexPtr()[col], inner_nnz_H_tri, m_P_qpswift, col, 0);
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_P_qpswift.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_c_qpswift(this->m_n) = this->m_settings.elastic_weight_l1;
            }
        }

        m_c_qpswift(Eigen::seqN(0, this->m_n)) = f;

        eigen_assert(m_P_qpswift.isCompressed());
    }

    EIGEN_STRONG_INLINE void
    construct_qpswift_constraints(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                 const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                 const Eigen::SparseMatrix<scalar_t>& A,
                                 const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                 const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert(A.isCompressed());

        if (!this->settings().reuse_pattern || !m_qpswift_initialized)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            // keep track of how many nnz we need per column for A (eq) and G (ineq)
            Eigen::VectorXi A_qpswift_nnz(n_vars);
            Eigen::VectorXi G_qpswift_nnz(n_vars);
            A_qpswift_nnz.setZero();
            G_qpswift_nnz.setZero();

            // count nnz's for A (eq) and G (ineq)
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        A_qpswift_nnz(it.col()) += 1;
                        if (this->m_settings.elastic_mode)
                        {
                            A_qpswift_nnz(this->m_n) += 1;
                        }
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_LB_ONLY_CONSTR ||
                             this->m_constraint_type[it.row()] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        G_qpswift_nnz(it.col()) += 1;
                        if (this->m_settings.elastic_mode)
                        {
                            G_qpswift_nnz(this->m_n) += 1;
                        }
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_CONSTR)
                    {
                        G_qpswift_nnz(it.col()) += 2;
                        if (this->m_settings.elastic_mode)
                        {
                            G_qpswift_nnz(this->m_n) += 2;
                        }
                    }
                }
            }

            // calculate the number of equality and inequality constraints
            int num_eq_constraints = 0;
            int num_ineq_constraints = 0;
            for (int i = 0; i < this->m_m; i++)
            {
                if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    num_eq_constraints++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR ||
                         this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    num_ineq_constraints++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    num_ineq_constraints += 2;
                }
            }

            // count and add nnz's for box constraints
            int num_box_eq_constraints = 0;
            int num_box_ineq_constraints = 0;
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    num_box_eq_constraints++;
                    A_qpswift_nnz(i) += 1;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR ||
                         this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    num_box_ineq_constraints++;
                    G_qpswift_nnz(i) += 1;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    num_box_ineq_constraints += 2;
                    G_qpswift_nnz(i) += 2;
                }
            }
            // lower and upper bound for slack variables
            if (this->m_settings.elastic_mode)
            {
                num_box_ineq_constraints += 2;
                G_qpswift_nnz(this->m_n) += 2;
            }

            m_b_qpswift.resize(num_eq_constraints + num_box_eq_constraints);
            m_h_qpswift.resize(num_ineq_constraints + num_box_ineq_constraints);

            // copy bounds
            int eq_bound_i = 0;
            int ineq_bound_i = 0;
            for (int i = 0; i < this->m_m; i++)
            {
                if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_A_to_qpswift_map(i) = eq_bound_i;
                    m_b_qpswift(eq_bound_i) = Alb(i);
                    eq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_A_to_qpswift_map(i) = ineq_bound_i;
                    m_h_qpswift(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_A_to_qpswift_map(i) = ineq_bound_i;
                    m_h_qpswift(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_A_to_qpswift_map(i) = ineq_bound_i;
                    m_h_qpswift(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                    m_h_qpswift(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
            }

            // copy box bounds
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_b_qpswift(eq_bound_i) = xlb(i);
                    eq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_h_qpswift(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_h_qpswift(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_h_qpswift(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                    m_h_qpswift(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
            }

            // set slack constraints
            if (this->m_settings.elastic_mode)
            {
                m_h_qpswift(ineq_bound_i) = -scalar_t(0);
                ineq_bound_i++;
                m_h_qpswift(ineq_bound_i) = scalar_t(1);
                ineq_bound_i++;
            }

            m_A_qpswift.resize(num_eq_constraints + num_box_eq_constraints, n_vars);
            m_A_qpswift.reserve(A_qpswift_nnz);
            m_G_qpswift.resize(num_ineq_constraints + num_box_ineq_constraints, n_vars);
            m_G_qpswift.reserve(G_qpswift_nnz);

            // copy eq and ineq matrix values
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        m_A_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = -it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_CONSTR)
                    {
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = -it.value();
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()) + 1, it.col()) = it.value();
                    }
                }
            }

            // create entries for the box constraints
            eq_bound_i = num_eq_constraints;
            ineq_bound_i = num_ineq_constraints;
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                if (this->m_box_constraint_type[col] == constraint_t::EQ_CONSTR)
                {
                    m_A_qpswift.coeffRef(eq_bound_i++, col) = 1;
                }
                else if (this->m_box_constraint_type[col] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_G_qpswift.coeffRef(ineq_bound_i++, col) = -1;
                }
                else if (this->m_box_constraint_type[col] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_G_qpswift.coeffRef(ineq_bound_i++, col) = 1;
                }
                else if (this->m_box_constraint_type[col] == constraint_t::INEQ_CONSTR)
                {
                    m_G_qpswift.coeffRef(ineq_bound_i++, col) = -1;
                    m_G_qpswift.coeffRef(ineq_bound_i++, col) = 1;
                }
            }

            // create entries for the slack constraints
            if (this->m_settings.elastic_mode)
            {
                eq_bound_i = 0;
                ineq_bound_i = 0;
                for (int i = 0; i < this->m_m; i++)
                {
                    scalar_t slack_coeff = 0;

                    if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                    {
                        slack_coeff = Alb(i);
                        m_A_qpswift.coeffRef(eq_bound_i, this->m_n) = slack_coeff;
                        eq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;

                        slack_coeff = 0;
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                }

                // entry for slack box constraints
                ineq_bound_i = num_ineq_constraints + num_box_ineq_constraints - 2;
                m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = -1;
                ineq_bound_i++;
                m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = 1;
                ineq_bound_i++;
            }

            m_A_qpswift.makeCompressed();
            m_G_qpswift.makeCompressed();
        }
        else
        {
            // copy bounds
            int eq_bound_i = 0;
            int ineq_bound_i = 0;
            for (int i = 0; i < this->m_m; i++)
            {
                if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_A_to_qpswift_map(i) = eq_bound_i;
                    m_b_qpswift(eq_bound_i) = Alb(i);
                    eq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_A_to_qpswift_map(i) = ineq_bound_i;
                    m_h_qpswift(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_A_to_qpswift_map(i) = ineq_bound_i;
                    m_h_qpswift(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_A_to_qpswift_map(i) = ineq_bound_i;
                    m_h_qpswift(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                    m_h_qpswift(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
            }

            // copy box bounds
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_b_qpswift(eq_bound_i) = xlb(i);
                    eq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_h_qpswift(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_h_qpswift(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_h_qpswift(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                    m_h_qpswift(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
            }

            // copy eq and ineq matrix values
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        m_A_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = -it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_CONSTR)
                    {
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()), it.col()) = -it.value();
                        m_G_qpswift.coeffRef(m_A_to_qpswift_map(it.row()) + 1, it.col()) = it.value();
                    }
                }
            }

            // copy slack coefficients
            if (this->m_settings.elastic_mode)
            {
                eq_bound_i = 0;
                ineq_bound_i = 0;
                for (int i = 0; i < this->m_m; i++)
                {
                    scalar_t slack_coeff = 0;

                    if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                    {
                        slack_coeff = Alb(i);
                        m_A_qpswift.coeffRef(eq_bound_i, this->m_n) = slack_coeff;
                        eq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;

                        slack_coeff = 0;
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_qpswift.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                }
            }

            // all other entries are unchanged
        }

        eigen_assert(m_A_qpswift.isCompressed());
        eigen_assert(m_G_qpswift.isCompressed());
    }

    void set_qpswift_settings() noexcept
    {
        if (m_qpswift_solver != nullptr)
        {
            m_qpswift_solver->options->reltol = this->m_settings.eps_rel;
            m_qpswift_solver->options->abstol = this->m_settings.eps_abs;
            m_qpswift_solver->options->maxit = this->m_settings.max_iter;
            m_qpswift_solver->options->verbose = this->m_settings.verbose;
        }
    }
};

} // namespace laopt

#endif // LAOPT_QPSWIFT_INTERFACE_HPP
