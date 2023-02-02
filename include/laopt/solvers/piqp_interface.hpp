#ifndef LAOPT_PIQP_INTERFACE_HPP
#define LAOPT_PIQP_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "piqp/piqp.hpp"

namespace laopt
{

template<typename Scalar = double, int Mode = piqp::KKTMode::KKT_FULL>
class PIQPSolver : public QPBase<PIQPSolver<Scalar, Mode>, Scalar>
{
public:
    using Base = QPBase<PIQPSolver<Scalar, Mode>, Scalar>;
    using scalar_t = typename Base::scalar_t;

private:
    using constraint_t = typename Base::constraint_t;

    piqp::Solver<Scalar, int, Mode> m_piqp_solver;
    bool m_piqp_initialized;

    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, int> m_P_piqp;
    Eigen::VectorX<scalar_t> m_c_piqp;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, int> m_A_piqp;
    Eigen::VectorX<scalar_t> m_b_piqp;
    Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, int> m_G_piqp;
    Eigen::VectorX<scalar_t> m_h_piqp;

    Eigen::VectorX<int> m_A_to_piqp_map; // maps row in A to row in A_piqp or C_piqp

public:
    PIQPSolver(int n, int m) :
        Base(n, m),
        m_piqp_initialized(false),
        m_A_to_piqp_map(m) {}

    qp_solver_info_t solve_impl(const Eigen::SparseMatrix<scalar_t>& H,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                const Eigen::SparseMatrix<scalar_t>& A,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        construct_piqp_data(H, f, xlb, xub, A, Alb, Aub);

        if (!this->m_settings.reuse_pattern)
        {
            m_piqp_solver.setup(m_P_piqp, m_A_piqp, m_G_piqp, m_c_piqp, m_b_piqp, m_h_piqp);
            m_piqp_initialized = true;
        }
        else
        {
            eigen_assert(m_piqp_initialized);

            m_piqp_solver.update(m_P_piqp, m_A_piqp, m_G_piqp, m_c_piqp, m_b_piqp, m_h_piqp);
        }

        set_piqp_settings();
        m_piqp_solver.solve();

        this->m_x = m_piqp_solver.result().x(Eigen::seqN(0, this->m_n));

        if (this->m_settings.elastic_mode) {
            this->m_elastic_var = m_piqp_solver.result().x(this->m_n);
        }

        // copy eq and ineq dual variables
        int eq_bound_i = 0;
        int ineq_bound_i = 0;
        for (int i = 0; i < this->m_m; i++)
        {
            if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
            {
                this->m_lam(i) = m_piqp_solver.result().y(eq_bound_i++);
            }
            else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam(i) = -m_piqp_solver.result().z(ineq_bound_i++);
            }
            else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam(i) = m_piqp_solver.result().z(ineq_bound_i++);
            }
            else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam(i) = -m_piqp_solver.result().z(ineq_bound_i++);
                this->m_lam(i) += m_piqp_solver.result().z(ineq_bound_i++);
            }
        }
        // copy box constraints dual variables
        for (int i = 0; i < this->m_n; i++)
        {
            if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
            {
                this->m_lam_bounds(i) = m_piqp_solver.result().y(eq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam_bounds(i) = -m_piqp_solver.result().z(ineq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam_bounds(i) = m_piqp_solver.result().z(ineq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam_bounds(i) = -m_piqp_solver.result().z(ineq_bound_i++);
                this->m_lam_bounds(i) += m_piqp_solver.result().z(ineq_bound_i++);
            }
        }

        this->m_info.iter = m_piqp_solver.result().info.iter;

        // update status
        switch (m_piqp_solver.result().info.status)
        {
            case piqp::Status::PIQP_SOLVED:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case piqp::Status::PIQP_MAX_ITER_REACHED:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            case piqp::Status::PIQP_PRIMAL_INFEASIBLE:
            case piqp::Status::PIQP_DUAL_INFEASIBLE:
                this->m_info.status = qp_status_t::INFEASIBLE;
                break;
            case piqp::Status::PIQP_INVALID_SETTINGS:
                this->m_info.status = qp_status_t::INVALID_SETTINGS;
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
    construct_piqp_data(const Eigen::SparseMatrix<scalar_t>& H,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                          const Eigen::SparseMatrix<scalar_t>& A,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                          const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        bool constraints_type_changed = this->parse_constraints_bounds(xlb, xub, Alb, Aub);
        if (constraints_type_changed)
        {
            // if the types of constraints have changed we can't reuse pattern
            this->settings().reuse_pattern = false;
        }

        construct_piqp_cost(H, f);
        construct_piqp_constraints(xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void
    construct_piqp_cost(const Eigen::SparseMatrix<scalar_t>& H,
                           const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f) noexcept
    {
        eigen_assert(H.isCompressed());

        if (!this->settings().reuse_pattern)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            Eigen::VectorXi P_piqp_nnz(n_vars);
            for (int col = 0; col < H.outerSize(); col++)
            {
                P_piqp_nnz(col) = H.outerIndexPtr()[col + 1] - H.outerIndexPtr()[col];
            }
            if (this->m_settings.elastic_mode)
            {
                P_piqp_nnz(this->m_n) = 1;
            }

            m_P_piqp.resize(n_vars, n_vars);
            m_P_piqp.reserve(P_piqp_nnz);
            m_c_piqp.resize(n_vars);

            // copy H
            for (int i = 0; i < H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(H, i); it; ++it)
                {
                    m_P_piqp.coeffRef(it.row(), it.col()) = it.value();
                }
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_P_piqp.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_c_piqp(this->m_n) = this->m_settings.elastic_weight_l1;
            }

            m_P_piqp.makeCompressed();
        }
        else
        {
            eigen_assert(m_P_piqp.isCompressed());

            // copy H
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_H_tri = m_P_piqp.outerIndexPtr()[col + 1] - m_P_piqp.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(H.valuePtr() + H.outerIndexPtr()[col], inner_nnz_H_tri, m_P_piqp, col, 0);
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_P_piqp.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_c_piqp(this->m_n) = this->m_settings.elastic_weight_l1;
            }
        }

        m_c_piqp(Eigen::seqN(0, this->m_n)) = f;

        eigen_assert(m_P_piqp.isCompressed());
    }

    EIGEN_STRONG_INLINE void
    construct_piqp_constraints(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                 const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                 const Eigen::SparseMatrix<scalar_t>& A,
                                 const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                 const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        eigen_assert(A.isCompressed());

        if (!this->settings().reuse_pattern)
        {
            int n_vars = this->m_n;
            if (this->m_settings.elastic_mode)
            {
                // we have to add a variable for the slack
                n_vars += 1;
            }

            // keep track of how many nnz we need per column for A (eq) and G (ineq)
            Eigen::VectorXi A_piqp_nnz(n_vars);
            Eigen::VectorXi G_piqp_nnz(n_vars);
            A_piqp_nnz.setZero();
            G_piqp_nnz.setZero();

            // count nnz's for A (eq) and G (ineq)
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        A_piqp_nnz(it.col()) += 1;
                        if (this->m_settings.elastic_mode)
                        {
                            A_piqp_nnz(this->m_n) += 1;
                        }
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_LB_ONLY_CONSTR ||
                             this->m_constraint_type[it.row()] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        G_piqp_nnz(it.col()) += 1;
                        if (this->m_settings.elastic_mode)
                        {
                            G_piqp_nnz(this->m_n) += 1;
                        }
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_CONSTR)
                    {
                        G_piqp_nnz(it.col()) += 2;
                        if (this->m_settings.elastic_mode)
                        {
                            G_piqp_nnz(this->m_n) += 2;
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
                    A_piqp_nnz(i) += 1;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR ||
                         this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    num_box_ineq_constraints++;
                    G_piqp_nnz(i) += 1;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    num_box_ineq_constraints += 2;
                    G_piqp_nnz(i) += 2;
                }
            }
            // lower and upper bound for slack variables
            if (this->m_settings.elastic_mode)
            {
                num_box_ineq_constraints += 2;
                G_piqp_nnz(this->m_n) += 2;
            }

            m_b_piqp.resize(num_eq_constraints + num_box_eq_constraints);
            m_h_piqp.resize(num_ineq_constraints + num_box_ineq_constraints);

            // copy bounds
            int eq_bound_i = 0;
            int ineq_bound_i = 0;
            for (int i = 0; i < this->m_m; i++)
            {
                if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_A_to_piqp_map(i) = eq_bound_i;
                    m_b_piqp(eq_bound_i) = Alb(i);
                    eq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_A_to_piqp_map(i) = ineq_bound_i;
                    m_h_piqp(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_A_to_piqp_map(i) = ineq_bound_i;
                    m_h_piqp(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_A_to_piqp_map(i) = ineq_bound_i;
                    m_h_piqp(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                    m_h_piqp(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
            }

            // copy box bounds
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_b_piqp(eq_bound_i) = xlb(i);
                    eq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_h_piqp(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_h_piqp(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_h_piqp(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                    m_h_piqp(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
            }

            // set slack constraints
            if (this->m_settings.elastic_mode)
            {
                m_h_piqp(ineq_bound_i) = -scalar_t(0);
                ineq_bound_i++;
                m_h_piqp(ineq_bound_i) = scalar_t(1);
                ineq_bound_i++;
            }

            m_A_piqp.resize(num_eq_constraints + num_box_eq_constraints, n_vars);
            m_A_piqp.reserve(A_piqp_nnz);
            m_G_piqp.resize(num_ineq_constraints + num_box_ineq_constraints, n_vars);
            m_G_piqp.reserve(G_piqp_nnz);

            // copy eq and ineq matrix values
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        m_A_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = -it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_CONSTR)
                    {
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = -it.value();
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()) + 1, it.col()) = it.value();
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
                    m_A_piqp.coeffRef(eq_bound_i++, col) = 1;
                }
                else if (this->m_box_constraint_type[col] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_G_piqp.coeffRef(ineq_bound_i++, col) = -1;
                }
                else if (this->m_box_constraint_type[col] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_G_piqp.coeffRef(ineq_bound_i++, col) = 1;
                }
                else if (this->m_box_constraint_type[col] == constraint_t::INEQ_CONSTR)
                {
                    m_G_piqp.coeffRef(ineq_bound_i++, col) = -1;
                    m_G_piqp.coeffRef(ineq_bound_i++, col) = 1;
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
                        m_A_piqp.coeffRef(eq_bound_i, this->m_n) = slack_coeff;
                        eq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;

                        slack_coeff = 0;
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                }

                // entry for slack box constraints
                ineq_bound_i = num_ineq_constraints + num_box_ineq_constraints - 2;
                m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = -1;
                ineq_bound_i++;
                m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = 1;
                ineq_bound_i++;
            }

            m_A_piqp.makeCompressed();
            m_G_piqp.makeCompressed();
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
                    m_A_to_piqp_map(i) = eq_bound_i;
                    m_b_piqp(eq_bound_i) = Alb(i);
                    eq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_A_to_piqp_map(i) = ineq_bound_i;
                    m_h_piqp(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_A_to_piqp_map(i) = ineq_bound_i;
                    m_h_piqp(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
                else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_A_to_piqp_map(i) = ineq_bound_i;
                    m_h_piqp(ineq_bound_i) = -Alb(i);
                    ineq_bound_i++;
                    m_h_piqp(ineq_bound_i) = Aub(i);
                    ineq_bound_i++;
                }
            }

            // copy box bounds
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_b_piqp(eq_bound_i) = xlb(i);
                    eq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    m_h_piqp(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    m_h_piqp(ineq_bound_i) = xub(i);
                    ineq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] == constraint_t::INEQ_CONSTR)
                {
                    m_h_piqp(ineq_bound_i) = -xlb(i);
                    ineq_bound_i++;
                    m_h_piqp(ineq_bound_i) = xub(i);
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
                        m_A_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = -it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] == constraint_t::INEQ_CONSTR)
                    {
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()), it.col()) = -it.value();
                        m_G_piqp.coeffRef(m_A_to_piqp_map(it.row()) + 1, it.col()) = it.value();
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
                        m_A_piqp.coeffRef(eq_bound_i, this->m_n) = slack_coeff;
                        eq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                    else if (this->m_constraint_type[i] == constraint_t::INEQ_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = -Alb(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;

                        slack_coeff = 0;
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_G_piqp.coeffRef(ineq_bound_i, this->m_n) = slack_coeff;
                        ineq_bound_i++;
                    }
                }
            }

            // all other entries are unchanged
        }

        eigen_assert(m_A_piqp.isCompressed());
        eigen_assert(m_G_piqp.isCompressed());
    }

    void set_piqp_settings() noexcept
    {
        m_piqp_solver.settings().feas_tol_rel = this->m_settings.eps_rel;
        m_piqp_solver.settings().feas_tol_abs = this->m_settings.eps_abs;
        m_piqp_solver.settings().dual_tol = this->m_settings.eps_abs;
        m_piqp_solver.settings().max_iter = this->m_settings.max_iter;
        m_piqp_solver.settings().verbose = this->m_settings.verbose;
    }
};

} // namespace laopt

#endif // LAOPT_PIQP_INTERFACE_HPP
