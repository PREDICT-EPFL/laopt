#ifndef LAOPT_OSQP_INTERFACE_HPP
#define LAOPT_OSQP_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "OsqpEigen/OsqpEigen.h"

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

    OsqpEigen::Solver osqp_solver;
    OsqpEigen::Settings osqp_settings;

    Eigen::VectorX<scalar_t> m_f_osqp; // internal copy of linear part of cost since memory has to be owned by solver
    Eigen::SparseMatrix<scalar_t> m_A_osqp;
    Eigen::VectorX<scalar_t> m_Alb_osqp;
    Eigen::VectorX<scalar_t> m_Aub_osqp;
    Eigen::VectorX<scalar_t> m_lam_osqp;

public:
    OSQPSolver(int n, int m) : Base(n, m), m_f_osqp(n)
    {
        set_osqp_settings();
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

        construct_osqp_constraints(xlb, xub, A, Alb, Aub);
        m_f_osqp = f;

        if (!this->m_settings.reuse_pattern)
        {
            osqp_solver.data()->setNumberOfVariables(this->m_n);
            osqp_solver.data()->setNumberOfConstraints(m_Alb_osqp.rows());

            osqp_solver.data()->setHessianMatrix(H);
            osqp_solver.data()->setGradient(m_f_osqp);
            osqp_solver.data()->setLinearConstraintsMatrix(m_A_osqp);
            osqp_solver.data()->setLowerBound(m_Alb_osqp);
            osqp_solver.data()->setUpperBound(m_Aub_osqp);

            osqp_solver.initSolver();
            osqp_solver.setWarmStart(this->m_x, m_lam_osqp);
            osqp_solver.solveProblem();
        }
        else
        {
            osqp_solver.updateHessianMatrix(H);
            osqp_solver.updateGradient(m_f_osqp);
            osqp_solver.updateLinearConstraintsMatrix(m_A_osqp);
            osqp_solver.updateBounds(m_Alb_osqp, m_Aub_osqp);

            osqp_solver.setWarmStart(this->m_x, m_lam_osqp);
            osqp_solver.solveProblem();
        }

        this->m_x = osqp_solver.getSolution();
        this->m_lam = osqp_solver.getDualSolution()(Eigen::seqN(0, this->m_m));
        // copy box constraints dual variables
        int bound_i = 0;
        for (int i = 0; i < this->m_n; i++)
        {
            if (this->m_box_constraint_type[i] != constraint_t::UNBOUNDED_CONSTR)
            {
                this->m_lam_bounds(i) = osqp_solver.getDualSolution()(this->m_m + bound_i);
                bound_i++;
            }
        }

        this->m_info.iter = osqp_solver.workspace()->info->iter;

        // update status
        switch (osqp_solver.workspace()->info->status_val)
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
    /** construct the Jacobian matrix accepted by OSQP */
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

            // assert m_A_osqp is not in compressed mode since we use innerNonZeros() later
            eigen_assert(m_A_osqp.innerNonZeroPtr() != nullptr);

            // copy A to the head of m_A_osqp
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                int inner_nnz_A = A.outerIndexPtr()[col + 1] - A.outerIndexPtr()[col];
                // assert that there is enough memory to copy inner indices
                eigen_assert(m_A_osqp.outerIndexPtr()[col + 1] - m_A_osqp.outerIndexPtr()[col] >= inner_nnz_A);
                // copy inner indices
                std::copy_n(A.innerIndexPtr() + A.outerIndexPtr()[col], inner_nnz_A, m_A_osqp.innerIndexPtr() + m_A_osqp.outerIndexPtr()[col]);
                // set inner non zeros to appropriate value
                m_A_osqp.innerNonZeroPtr()[col] = inner_nnz_A;
                // copy values
                copy_n_into_sparse_matrix(A.valuePtr() + A.outerIndexPtr()[col], inner_nnz_A, m_A_osqp, col, 0);
            }

            // copy entries for the box constraints
            bound_i = 0;
            for (Eigen::Index col = 0; col < this->m_n; col++)
            {
                if (this->m_box_constraint_type[col] != constraint_t::UNBOUNDED_CONSTR)
                {
                    // we assert that we have exactly one more free memory spot
                    eigen_assert(m_A_osqp.outerIndexPtr()[col + 1] - m_A_osqp.outerIndexPtr()[col] == m_A_osqp.innerNonZeroPtr()[col] + 1);
                    // set inner index
                    *(m_A_osqp.innerIndexPtr() + m_A_osqp.outerIndexPtr()[col] + m_A_osqp.innerNonZeroPtr()[col]) = this->m_m + bound_i;
                    bound_i++;
                    // set value
                    *(m_A_osqp.valuePtr() + m_A_osqp.outerIndexPtr()[col] + m_A_osqp.innerNonZeroPtr()[col]) = 1;
                    // increase non zeros for column
                    m_A_osqp.innerNonZeroPtr()[col] += 1;
                }
            }

//            m_A_osqp.makeCompressed();
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
    }

    void set_osqp_settings() noexcept
    {
        osqp_solver.settings()->setRelativeTolerance(this->m_settings.eps_rel);
        osqp_solver.settings()->setAbsoluteTolerance(this->m_settings.eps_abs);
        osqp_solver.settings()->setMaxIteration(this->m_settings.max_iter);
        osqp_solver.settings()->setWarmStart(true);
        osqp_solver.settings()->setVerbosity(this->m_settings.verbose);
    }

};

} // laopt namespace

#endif // LAOPT_OSQP_INTERFACE_HPP
