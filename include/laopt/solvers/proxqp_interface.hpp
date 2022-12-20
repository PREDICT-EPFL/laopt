#ifndef LAOPT_PROXQP_INTERFACE_HPP
#define LAOPT_PROXQP_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include <proxsuite/proxqp/sparse/sparse.hpp>

namespace laopt
{

template<typename Scalar = double>
class ProxQPSolver : public QPBase<ProxQPSolver<Scalar>, Scalar>
{
public:
    using Base = QPBase<ProxQPSolver<Scalar>, Scalar>;
    using scalar_t = typename Base::scalar_t;

private:
    using constraint_t = typename Base::constraint_t;

    proxsuite::proxqp::sparse::QP<scalar_t, int> m_proxqp_solver;
    bool m_proxqp_initialized;

    Eigen::SparseMatrix<scalar_t> m_H_proxqp;
    Eigen::VectorX<scalar_t> m_g_proxqp;
    Eigen::SparseMatrix<scalar_t> m_A_proxqp;
    Eigen::VectorX<scalar_t> m_b_proxqp;
    Eigen::SparseMatrix<scalar_t> m_C_proxqp;
    Eigen::VectorX<scalar_t> m_Clb_proxqp;
    Eigen::VectorX<scalar_t> m_Cub_proxqp;
    Eigen::VectorX<scalar_t> m_x_proxqp;
    Eigen::VectorX<scalar_t> m_y_proxqp;
    Eigen::VectorX<scalar_t> m_z_proxqp;

    Eigen::VectorX<int> m_A_to_proxqp_map; // maps row in A to row in A_proxqp or C_proxqp

public:
    ProxQPSolver(int n, int m) :
        Base(n, m),
        m_proxqp_solver(1, 0, 0),
        m_proxqp_initialized(false),
        m_A_to_proxqp_map(m) {}

    qp_solver_info_t solve_impl(const Eigen::SparseMatrix<scalar_t>& H,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                const Eigen::SparseMatrix<scalar_t>& A,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        construct_proxqp_data(H, f, xlb, xub, A, Alb, Aub);

        if (!this->m_settings.reuse_pattern)
        {
            using SolverType = proxsuite::proxqp::sparse::QP<scalar_t, int>;
            m_proxqp_solver.~SolverType();
            new (&m_proxqp_solver) proxsuite::proxqp::sparse::QP<scalar_t, int>(m_H_proxqp.rows(), m_A_proxqp.rows(), m_C_proxqp.rows());

            m_proxqp_solver.init(m_H_proxqp, m_g_proxqp, m_A_proxqp, m_b_proxqp, m_C_proxqp, m_Clb_proxqp, m_Cub_proxqp);
            m_proxqp_initialized = true;
        }
        else
        {
            eigen_assert(m_proxqp_initialized);

            m_proxqp_solver.update(m_H_proxqp, m_g_proxqp, m_A_proxqp, m_b_proxqp, m_C_proxqp, m_Clb_proxqp, m_Cub_proxqp);
        }

        set_proxqp_settings();
        m_proxqp_solver.solve(m_x_proxqp, m_y_proxqp, m_z_proxqp);

        this->m_x = m_proxqp_solver.results.x(Eigen::seqN(0, this->m_n));
        // copy eq and ineq dual variables
        int eq_bound_i = 0;
        int ineq_bound_i = 0;
        for (int i = 0; i < this->m_m; i++)
        {
            if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
            {
                this->m_lam(i) = m_proxqp_solver.results.y(eq_bound_i++);
            }
            else if (this->m_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
            {
                this->m_lam(i) = m_proxqp_solver.results.z(ineq_bound_i++);
            }
        }
        // copy box constraints dual variables
        for (int i = 0; i < this->m_n; i++)
        {
            if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
            {
                this->m_lam_bounds(i) = m_proxqp_solver.results.y(eq_bound_i++);
            }
            else if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
            {
                this->m_lam_bounds(i) = m_proxqp_solver.results.z(ineq_bound_i++);
            }
        }

        this->m_info.iter = m_proxqp_solver.results.info.iter;

        // update status
        switch (m_proxqp_solver.results.info.status)
        {
            case proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            case proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE:
            case proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE:
                this->m_info.status = qp_status_t::INFEASIBLE;
                break;
            case proxsuite::proxqp::QPSolverOutput::PROXQP_NOT_RUN:
            default:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
        }

        return this->m_info;
    }

private:
    /** construct the data matrices accepted by OSQP */
    EIGEN_STRONG_INLINE void
    construct_proxqp_data(const Eigen::SparseMatrix<scalar_t>& H,
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

        construct_proxqp_cost(H, f);
        construct_proxqp_constraints(xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void
    construct_proxqp_cost(const Eigen::SparseMatrix<scalar_t>& H,
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

            Eigen::VectorXi H_proxqp_nnz(n_vars);
            for (int col = 0; col < H.outerSize(); col++)
            {
                H_proxqp_nnz(col) = H.outerIndexPtr()[col + 1] - H.outerIndexPtr()[col];
            }
            if (this->m_settings.elastic_mode)
            {
                H_proxqp_nnz(this->m_n) = 1;
            }

            m_x_proxqp.resize(n_vars);
            m_H_proxqp.resize(n_vars, n_vars);
            m_H_proxqp.reserve(H_proxqp_nnz);
            m_g_proxqp.resize(n_vars);

            m_x_proxqp.setZero();
            m_x_proxqp(Eigen::seqN(0, this->m_n)) = this->m_x;

            // copy H
            for (int i = 0; i < H.outerSize(); ++i)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(H, i); it; ++it)
                {
                    m_H_proxqp.coeffRef(it.row(), it.col()) = it.value();
                }
            }

            if (this->m_settings.elastic_mode)
            {
                // add l2 penalties to slack
                m_H_proxqp.coeffRef(this->m_n, this->m_n) = this->m_settings.elastic_weight_l2;
                // add l1 penalties to slack
                m_g_proxqp(this->m_n) = this->m_settings.elastic_weight_l1;
            }

            m_H_proxqp.makeCompressed();
        }
        else
        {
            m_x_proxqp(Eigen::seqN(0, this->m_n)) = this->m_x;

            eigen_assert(m_H_proxqp.isCompressed());

            // copy H
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_H_tri = m_H_proxqp.outerIndexPtr()[col + 1] - m_H_proxqp.outerIndexPtr()[col];
                copy_n_into_sparse_matrix(H.valuePtr() + H.outerIndexPtr()[col], inner_nnz_H_tri, m_H_proxqp, col, 0);
            }

            // entries for slacks are kept from last iteration
        }

        m_g_proxqp(Eigen::seqN(0, this->m_n)) = f;

        eigen_assert(m_H_proxqp.isCompressed());
    }

    EIGEN_STRONG_INLINE void
    construct_proxqp_constraints(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
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

            // keep track of how many nnz we need per column for A (eq) and C (ineq)
            Eigen::VectorXi A_proxqp_nnz(n_vars);
            Eigen::VectorXi C_proxqp_nnz(n_vars);
            A_proxqp_nnz.setZero();
            C_proxqp_nnz.setZero();

            // count nnz's for A (eq) and C (ineq)
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        A_proxqp_nnz(it.col()) += 1;
                        if (this->m_settings.elastic_mode)
                        {
                            A_proxqp_nnz(this->m_n) += 1;
                        }
                    }
                    else if (this->m_constraint_type[it.row()] != constraint_t::UNBOUNDED_CONSTR)
                    {
                        C_proxqp_nnz(it.col()) += 1;
                        if (this->m_settings.elastic_mode)
                        {
                            A_proxqp_nnz(this->m_n) += 1;
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
                else if (this->m_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    num_ineq_constraints++;
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
                    A_proxqp_nnz(i) += 1;
                }
                else if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    num_box_ineq_constraints++;
                    C_proxqp_nnz(i) += 1;
                }
            }
            // lower and upper bound for slack variables
            if (this->m_settings.elastic_mode)
            {
                num_box_ineq_constraints++;
                C_proxqp_nnz(this->m_n) += 1;
            }

            m_b_proxqp.resize(num_eq_constraints + num_box_eq_constraints);
            m_y_proxqp.resize(num_eq_constraints + num_box_eq_constraints);
            m_Clb_proxqp.resize(num_ineq_constraints + num_box_ineq_constraints);
            m_Cub_proxqp.resize(num_ineq_constraints + num_box_ineq_constraints);
            m_z_proxqp.resize(num_ineq_constraints + num_box_ineq_constraints);

            // copy bounds
            int eq_bound_i = 0;
            int ineq_bound_i = 0;
            for (int i = 0; i < this->m_m; i++)
            {
                if (this->m_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_A_to_proxqp_map(i) = eq_bound_i;
                    m_b_proxqp(eq_bound_i) = Alb(i);
                    m_y_proxqp(eq_bound_i) = this->m_lam(i);
                    eq_bound_i++;
                }
                else if (this->m_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_A_to_proxqp_map(i) = ineq_bound_i;
                    m_Clb_proxqp(ineq_bound_i) = Alb(i);
                    m_Cub_proxqp(ineq_bound_i) = Aub(i);
                    m_z_proxqp(ineq_bound_i) = this->m_lam(i);
                    ineq_bound_i++;
                }
            }

            // copy box bounds
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_b_proxqp(eq_bound_i) = xlb(i);
                    m_y_proxqp(eq_bound_i) = this->m_lam_bounds(i);
                    eq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Clb_proxqp(ineq_bound_i) = xlb(i);
                    m_Cub_proxqp(ineq_bound_i) = xub(i);
                    m_z_proxqp(ineq_bound_i) = this->m_lam_bounds(i);
                    ineq_bound_i++;
                }
            }

            // set slack constraints
            if (this->m_settings.elastic_mode)
            {
                m_Clb_proxqp(ineq_bound_i) = scalar_t(0);
                m_Cub_proxqp(ineq_bound_i) = scalar_t(1);
                ineq_bound_i++;
            }

            m_A_proxqp.resize(num_eq_constraints + num_box_eq_constraints, n_vars);
            m_A_proxqp.reserve(A_proxqp_nnz);
            m_C_proxqp.resize(num_ineq_constraints + num_box_ineq_constraints, n_vars);
            m_C_proxqp.reserve(C_proxqp_nnz);

            // copy eq and ineq matrix values
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        m_A_proxqp.coeffRef(m_A_to_proxqp_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] != constraint_t::UNBOUNDED_CONSTR)
                    {
                        m_C_proxqp.coeffRef(m_A_to_proxqp_map(it.row()), it.col()) = it.value();
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
                    m_A_proxqp.coeffRef(eq_bound_i++, col) = 1;
                }
                else if (this->m_box_constraint_type[col] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_C_proxqp.coeffRef(ineq_bound_i++, col) = 1;
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
                        m_A_proxqp.coeffRef(eq_bound_i++, this->m_n) = slack_coeff;
                    }
                    else if (this->m_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = Alb(i);
                        }
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_C_proxqp.coeffRef(ineq_bound_i++, this->m_n) = slack_coeff;
                    }
                }

                // entry for slack box constraints
                ineq_bound_i = num_ineq_constraints + num_box_ineq_constraints - 1;
                m_C_proxqp.coeffRef(ineq_bound_i++, this->m_n) = 1;
            }

            m_A_proxqp.makeCompressed();
            m_C_proxqp.makeCompressed();
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
                    m_b_proxqp(eq_bound_i) = Alb(i);
                    m_y_proxqp(eq_bound_i) = this->m_lam(i);
                    eq_bound_i++;
                }
                else if (this->m_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Clb_proxqp(ineq_bound_i) = Alb(i);
                    m_Cub_proxqp(ineq_bound_i) = Aub(i);
                    m_z_proxqp(ineq_bound_i) = this->m_lam(i);
                    ineq_bound_i++;
                }
            }

            // copy box bounds
            for (int i = 0; i < this->m_n; i++)
            {
                if (this->m_box_constraint_type[i] == constraint_t::EQ_CONSTR)
                {
                    m_b_proxqp(eq_bound_i) = xlb(i);
                    m_y_proxqp(eq_bound_i) = this->m_lam_bounds(i);
                    eq_bound_i++;
                }
                else if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                {
                    m_Clb_proxqp(ineq_bound_i) = xlb(i);
                    m_Cub_proxqp(ineq_bound_i) = xub(i);
                    m_z_proxqp(ineq_bound_i) = this->m_lam_bounds(i);
                    ineq_bound_i++;
                }
            }

            // copy A to the correct rows of m_A_proxqp and m_C_proxqp
            for (int i = 0; i < A.outerSize(); i++)
            {
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(A, i); it; ++it)
                {
                    if (this->m_constraint_type[it.row()] == constraint_t::EQ_CONSTR)
                    {
                        m_A_proxqp.coeffRef(m_A_to_proxqp_map(it.row()), it.col()) = it.value();
                    }
                    else if (this->m_constraint_type[it.row()] != constraint_t::UNBOUNDED_CONSTR)
                    {
                        m_C_proxqp.coeffRef(m_A_to_proxqp_map(it.row()), it.col()) = it.value();
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
                        m_A_proxqp.coeffRef(eq_bound_i++, this->m_n) = slack_coeff;
                    }
                    else if (this->m_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
                    {
                        if (Alb(i) > scalar_t(0)) {
                            slack_coeff = Alb(i);
                        }
                        if (Aub(i) < scalar_t(0)) {
                            slack_coeff = Aub(i);
                        }
                        m_C_proxqp.coeffRef(ineq_bound_i++, this->m_n) = slack_coeff;
                    }
                }
            }

            // all other entries of m_C_proxqp are unchanged
        }

        eigen_assert(m_A_proxqp.isCompressed());
        eigen_assert(m_C_proxqp.isCompressed());
    }

    void set_proxqp_settings() noexcept
    {
        m_proxqp_solver.settings.eps_rel = this->m_settings.eps_rel;
        m_proxqp_solver.settings.eps_abs = this->m_settings.eps_abs;
        m_proxqp_solver.settings.max_iter = this->m_settings.max_iter;
        m_proxqp_solver.settings.verbose = this->m_settings.verbose;
    }
};

} // namespace laopt

#endif //LAOPT_PROXQP_INTERFACE_HPP
