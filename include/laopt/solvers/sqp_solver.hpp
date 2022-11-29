#ifndef LAOPT_SQP_SOLVER_HPP
#define LAOPT_SQP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "laopt/solvers/qp_base.hpp"

namespace laopt
{

enum struct hessian_approximation_t {
    EXACT,
    EXACT_NO_CONSTRAINTS,
};

template <typename Scalar>
struct sqp_settings_t {
    Scalar tau                                    = 0.5;                            // line search iteration decrease, 0 < tau < 1
    Scalar eta                                    = 0.25;                           // line search parameter, 0 < eta < 1
    Scalar rho                                    = 0.5;                            // line search parameter, 0 < rho < 1
    Scalar eps_prim                               = 1e-6;                           // primal step termination threshold, eps_prim > 0
    Scalar eps_dual                               = 1e-4;                           // dual step termination threshold, eps_dual > 0
    hessian_approximation_t hessian_approximation = hessian_approximation_t::EXACT; // hessian approximation
    int max_iter                                  = 1000;
    int line_search_max_iter                      = 100;
    bool verbose             = false;

    bool validate()
    {
        return 0.0 < tau && tau < 1.0 &&
               0.0 < eta && eta < 1.0 &&
               0.0 < rho && rho < 1.0 &&
               eps_prim > 0.0 &&
               eps_dual > 0.0 &&
               max_iter > 0 &&
               line_search_max_iter > 0;
    }
};

enum struct sqp_status_t {
    SOLVED = 1,
    MAX_ITER_REACHED = -1,
    INFEASIBLE = -2,
    NON_CONVEX_QP = -3,
    QP_SOLVER_ERROR = -4,
    UNSOLVED = -9,
    INVALID_SETTINGS = -10,
};

struct sqp_info_t {
    sqp_status_t status = sqp_status_t::UNSOLVED;
    int iter = 0;
    int qp_iter = 0;
};

template<typename Derived, typename Problem, typename QPSolver>
class SQPBase
{
public:
    using scalar_t = typename Problem::scalar_t;

private:
    Problem& prob;

    Eigen::VectorX<scalar_t>      m_x;            // variable primal
    Eigen::VectorX<scalar_t>      m_lam;          // dual variable
    Eigen::VectorX<scalar_t>      m_lam_bounds;   // dual variable for bounds
    Eigen::VectorX<scalar_t>      m_lbx, m_ubx;   // variable bounds
    Eigen::VectorX<scalar_t>      m_cost_grad;    // Gradient of the cost function
    Eigen::SparseMatrix<scalar_t> m_lag_hess;     // Hessian of Lagrangian
    Eigen::VectorX<scalar_t>      m_g;            // constraints evaluation
    Eigen::SparseMatrix<scalar_t> m_g_jac;        // constraints Jacobian
    Eigen::VectorX<scalar_t>      m_lbg, m_ubg;   // equality/inequality constraints evaluated

    QPSolver m_qp_solver;
    sqp_settings_t<scalar_t> m_settings;
    sqp_info_t m_info;
    bool m_first_solve;

    Eigen::VectorX<scalar_t> m_p;             // primal search direction
    Eigen::VectorX<scalar_t> m_lam_qp;        // dual variable solution of qp
    Eigen::VectorX<scalar_t> m_lam_bounds_qp; // dual variable solution of qp for bounds

    scalar_t m_primal_feasibility_inf;
    scalar_t m_complementarity_inf;
    scalar_t m_stationarity_inf;

    // internal variables for line search
    Eigen::VectorX<scalar_t> m_x_step_line_search; // primal search step for line search

public:

    explicit SQPBase(Problem& prob) :
        prob(prob),
        m_x(prob.variables()),
        m_lam(prob.constraints.rows()),
        m_lam_bounds(prob.variables()),
        m_lbx(prob.variables()), m_ubx(prob.variables()),
        m_cost_grad(prob.variables()),
        m_g(prob.constraints.rows()),
        m_lbg(prob.constraints.rows()), m_ubg(prob.constraints.rows()),
        m_qp_solver(prob.variables(), prob.constraints.rows()),
        m_first_solve(true),
        m_p(prob.variables()),
        m_lam_qp(prob.constraints.rows()),
        m_lam_bounds_qp(prob.variables()),
        m_x_step_line_search(prob.variables())
    {
        m_x.setZero();
        m_lam.setZero();
        m_lam_bounds.setZero();

        prob.constraints.jacobian.allocate_memory(m_g_jac);

        prob.set_decision_variable(m_x);
    }

    EIGEN_STRONG_INLINE const sqp_settings_t<scalar_t>& settings() const noexcept { return m_settings; }
    EIGEN_STRONG_INLINE sqp_settings_t<scalar_t>& settings() noexcept { return m_settings; }

    EIGEN_STRONG_INLINE const sqp_info_t& info() const noexcept { return m_info; }
    EIGEN_STRONG_INLINE sqp_info_t& info() noexcept { return m_info; }

    EIGEN_STRONG_INLINE const QPSolver& qp_solver() const noexcept { return m_qp_solver; }
    EIGEN_STRONG_INLINE QPSolver& qp_solver() noexcept { return m_qp_solver; }

    /** solve the NLP */
    sqp_info_t solve() noexcept
    {
        if (m_first_solve) {
            m_first_solve = false;

            if (m_settings.hessian_approximation == hessian_approximation_t::EXACT_NO_CONSTRAINTS)
            {
                prob.objective.hessian.allocate_memory(m_lag_hess);
            }
            else
            {
                prob.lagrangian.hessian.allocate_memory(m_lag_hess);
            }

            if (m_settings.verbose)
            {
                std::cout << "----------------------------------------------------------" << std::endl;
                std::cout << "                        laOPT SQP                         " << std::endl;
                std::cout << "            (c) Roland Schwan, Johannes Waibel            " << std::endl;
                std::cout << "   École Polytechnique Fédérale de Lausanne (EPFL) 2022   " << std::endl;
                std::cout << "----------------------------------------------------------" << std::endl;
                std::cout << "variables n = " << prob.variables() << std::endl;
                std::cout << "constraints m = " << prob.constraints.rows() << std::endl;
                std::cout << "lagrangian hessian nnz = " << m_lag_hess.nonZeros() << std::endl;
                std::cout << "constraints jacobian nnz = " << m_g_jac.nonZeros() << std::endl;
            }
        }

        if (!m_settings.validate())
        {
            m_info.status = sqp_status_t::INVALID_SETTINGS;
            return m_info;
        }

        scalar_t alpha = 0;

        m_info.qp_iter = 0;
        m_info.iter = 0;

        // update convergence criteria
        m_primal_feasibility_inf = max_constraints_violation(m_x);
        m_complementarity_inf = max_complementarity_violation(m_x, m_lam);
        m_stationarity_inf = max_stationarity_violation(m_x, m_lam, m_lam_bounds);

        if (m_settings.verbose)
        {
            std::cout << "iter   objective     primal_inf    comp_inf      stat_inf     alpha     qp_iter" << std::endl;
            prob.set_decision_variable(m_x);
            printf("%4d   %.5e   %.5e   %.5e   %.5e  %.3e %7d\n",
                   m_info.iter,
                   prob.eval_objective(),
                   m_primal_feasibility_inf,
                   m_complementarity_inf,
                   m_stationarity_inf,
                   alpha,
                   0);
        }

        if (termination_criteria()) {
            m_info.status = sqp_status_t::SOLVED;
            if (m_settings.verbose) print_solve_info();
            return m_info;
        }

        while (m_info.iter < m_settings.max_iter)
        {
            m_info.iter++;

            // linearize and solve problem
            linearize_problem();
            solve_qp();

            // reuse sparsity pattern for further solves
            m_qp_solver.settings().reuse_pattern = true;

            if (m_settings.verbose && m_qp_solver.info().status == qp_status_t::MAX_ITER_REACHED)
            {
                std::cerr << "[SQPSolver::solve] QP problem has reached maximum iterations. Continue using suboptimal solution." << std::endl;
            }

            if (!(m_qp_solver.info().status == qp_status_t::SOLVED || m_qp_solver.info().status == qp_status_t::MAX_ITER_REACHED))
            {
                switch (m_qp_solver.info().status)
                {
                    case qp_status_t::INFEASIBLE:
                        m_info.status = sqp_status_t::INFEASIBLE;
                        break;
                    case qp_status_t::NON_CONVEX:
                        m_info.status = sqp_status_t::NON_CONVEX_QP;
                        break;
                    case qp_status_t::UNSOLVED:
                        m_info.status = sqp_status_t::UNSOLVED;
                        break;
                    default:
                        m_info.status = sqp_status_t::QP_SOLVER_ERROR;
                        break;
                }
                return m_info;
            }

            // calculate step size
            alpha = step_size_selection(m_p);

            // take step
            m_x          += alpha * m_p;
            m_lam        += alpha * (m_lam_qp - m_lam);
            m_lam_bounds += alpha * (m_lam_bounds_qp - m_lam_bounds);

            // update convergence criteria
            m_primal_feasibility_inf = max_constraints_violation(m_x);
            m_complementarity_inf = max_complementarity_violation(m_x, m_lam);
            m_stationarity_inf = max_stationarity_violation(m_x, m_lam, m_lam_bounds);

            if (m_settings.verbose)
            {
                prob.set_decision_variable(m_x);
                printf("%4d   %.5e   %.5e   %.5e   %.5e  %.3e %7d\n",
                       m_info.iter,
                       prob.eval_objective(),
                       m_primal_feasibility_inf,
                       m_complementarity_inf,
                       m_stationarity_inf,
                       alpha,
                       m_qp_solver.info().iter);
            }

            if (termination_criteria()) {
                m_info.status = sqp_status_t::SOLVED;
                if (m_settings.verbose) print_solve_info();
                return m_info;
            }
        }

        m_info.status = sqp_status_t::MAX_ITER_REACHED;
        if (m_settings.verbose) print_solve_info();
        return m_info;
    }

protected:
    EIGEN_STRONG_INLINE void print_solve_info() noexcept
    {
        std::cout << "status: ";
        switch (m_info.status)
        {
            case sqp_status_t::SOLVED:
                std::cout << "SOLVED" << std::endl;
                break;
            case sqp_status_t::MAX_ITER_REACHED:
                std::cout << "MAX_ITER_REACHED" << std::endl;
                break;
            case sqp_status_t::INFEASIBLE:
                std::cout << "INFEASIBLE" << std::endl;
                break;
            case sqp_status_t::NON_CONVEX_QP:
                std::cout << "NON_CONVEX_QP" << std::endl;
                break;
            case sqp_status_t::QP_SOLVER_ERROR:
                std::cout << "QP_SOLVER_ERROR" << std::endl;
                break;
            case sqp_status_t::UNSOLVED:
                std::cout << "UNSOLVED" << std::endl;
                break;
            case sqp_status_t::INVALID_SETTINGS:
                std::cout << "INVALID_SETTINGS" << std::endl;
                break;
            default:
                std::cout << "UNKNOWN STATUS" << std::endl;
                break;
        }
        std::cout << "sqp iterations: " << m_info.iter << std::endl;
        std::cout << "qp iterations: " << m_info.qp_iter << std::endl;
    }

    EIGEN_STRONG_INLINE void linearize_problem() noexcept
    {
        prob.set_decision_variable(m_x);

        prob.eval_objective_gradient(m_cost_grad);
        prob.eval_variable_bounds(m_lbx, m_ubx);

        Eigen::Map<Eigen::VectorX<scalar_t>> hessian_buffer(m_lag_hess.valuePtr(), m_lag_hess.nonZeros());
        if (m_settings.hessian_approximation == hessian_approximation_t::EXACT_NO_CONSTRAINTS)
        {
            prob.eval_objective_hessian(hessian_buffer);
        }
        else
        {
            prob.eval_lagrangian_hessian(1.0, m_lam, hessian_buffer);
        }

        prob.eval_constraints(m_g, m_lbg, m_ubg);
        Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer(m_g_jac.valuePtr(), m_g_jac.nonZeros());
        prob.eval_constraints_jacobian(jacobian_buffer);

        // add constant part from linearization to bounds
        m_lbx -= m_x;
        m_ubx -= m_x;
        m_lbg -= m_g;
        m_ubg -= m_g;

        // regularize hessian
        hessian_regularisation(m_lag_hess);
    }

    /** prepare and solve the qp */
    EIGEN_STRONG_INLINE void solve_qp() noexcept
    {
        m_qp_solver.solve(m_lag_hess, m_cost_grad, m_lbx, m_ubx, m_g_jac, m_lbg, m_ubg);
        m_info.qp_iter += m_qp_solver.info().iter;

        // extract primal and dual solutions
        m_p = m_qp_solver.primal_solution();
        m_lam_qp = m_qp_solver.dual_solution();
        m_lam_bounds_qp = m_qp_solver.dual_bounds_solution();
    }

    EIGEN_STRONG_INLINE scalar_t
    l1_constraints_violation(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        scalar_t c = scalar_t(0);

        prob.set_decision_variable(x);

        // l <= g(x) <= u
        prob.eval_constraints(m_g, m_lbg, m_ubg);
        c += (m_lbg - m_g).cwiseMax(0.0).sum();
        c += (m_g - m_ubg).cwiseMax(0.0).sum();

        // l <= x <= u
        prob.eval_variable_bounds(m_lbx, m_ubx);
        c += (m_lbx - x).cwiseMax(0.0).sum();
        c += (x - m_ubx).cwiseMax(0.0).sum();

        return c;
    }

    EIGEN_STRONG_INLINE scalar_t
    max_constraints_violation(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        scalar_t c = scalar_t(0);

        prob.set_decision_variable(x);

        // lbg <= g <= ubg
        prob.eval_constraints(m_g, m_lbg, m_ubg);
        c = fmax(c, (m_lbg - m_g).maxCoeff());
        c = fmax(c, (m_g - m_ubg).maxCoeff());

        // l <= x <= u
        prob.eval_variable_bounds(m_lbx, m_ubx);
        c = fmax(c, (m_lbx - x).maxCoeff());
        c = fmax(c, (x - m_ubx).maxCoeff());

        return c;
    }

    EIGEN_STRONG_INLINE scalar_t
    max_complementarity_violation(Eigen::Ref<Eigen::VectorX<scalar_t>> x, const Eigen::Ref<const Eigen::VectorX<scalar_t>> &dual) noexcept
    {
        scalar_t c = scalar_t(0);

        prob.set_decision_variable(x);

        // lbg <= g <= ubg
        prob.eval_constraints(m_g, m_lbg, m_ubg);
        c = fmax(c, (m_lbg - m_g).cwiseMax(0.0).cwiseProduct(dual).maxCoeff());
        c = fmax(c, (m_g - m_ubg).cwiseMax(0.0).cwiseProduct(dual).maxCoeff());

        // note that we only check the complementarity condition for non-linear constraints,
        // the complementarity condition for variable bounds should already be satisfied by the QP solution

        return c;
    }

    EIGEN_STRONG_INLINE scalar_t
    max_stationarity_violation(Eigen::Ref<Eigen::VectorX<scalar_t>> x,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>> &dual,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>> &dual_bounds) noexcept
    {
        scalar_t c = scalar_t(0);

        prob.set_decision_variable(x);

        prob.eval_objective_gradient(m_cost_grad);

        Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer(m_g_jac.valuePtr(), m_g_jac.nonZeros());
        prob.eval_constraints_jacobian(jacobian_buffer);

        c = (m_cost_grad + m_g_jac.transpose() * dual + dual_bounds).cwiseAbs().maxCoeff();

        return c;
    }

    /** step size selection: line search / filter / trust region */
    EIGEN_STRONG_INLINE scalar_t
    step_size_selection(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& p) noexcept
    {
        return static_cast<Derived*>(this)->step_size_selection_impl(p);
    }

    EIGEN_STRONG_INLINE bool
    termination_criteria() noexcept
    {
        return static_cast<Derived*>(this)->termination_criteria_impl();
    }

    EIGEN_STRONG_INLINE void
    hessian_regularisation(Eigen::SparseMatrix<scalar_t>& lag_hessian)
    {
        static_cast<Derived*>(this)->hessian_regularisation_impl(lag_hessian);
    }

    /**
    * default implementations
    */

    /** default algorithm: l1-norm line search */
    EIGEN_STRONG_INLINE scalar_t
    step_size_selection_impl(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& p) noexcept
    {
        const scalar_t tau = m_settings.tau; // line search step decrease, 0 < tau < settings.tau

        scalar_t constr_l1 = l1_constraints_violation(m_x);

//         scalar_t mu = abs(m_cost_grad.dot(p)) / ((1 - m_settings.rho) * constr_l1);
        scalar_t mu = m_lam_qp.template lpNorm<Eigen::Infinity>();

        prob.set_decision_variable(m_x);
        scalar_t cost_1 = prob.eval_objective();

        scalar_t phi_l1 = cost_1 + mu * constr_l1;
        scalar_t Dp_phi_l1 = m_cost_grad.dot(p) - mu * constr_l1;

        scalar_t alpha = scalar_t(1.0);
        scalar_t cost_step;
        for (int i = 1; i < m_settings.line_search_max_iter; i++)
        {
            m_x_step_line_search = m_x + alpha * p;
            prob.set_decision_variable(m_x_step_line_search);
            cost_step = prob.eval_objective();

            scalar_t phi_l1_step = cost_step + mu * l1_constraints_violation(m_x_step_line_search);

            if (phi_l1_step <= (phi_l1 + alpha * m_settings.eta * Dp_phi_l1))
            {
                return alpha;
            }
            else
            {
                alpha = tau * alpha;
            }
        }

        return alpha;
    }

    /** default termination criteria */
    EIGEN_STRONG_INLINE bool
    termination_criteria_impl() noexcept
    {
        return (m_primal_feasibility_inf <= m_settings.eps_prim) &&
               (m_complementarity_inf <= m_settings.eps_dual) &&
               (m_stationarity_inf <= m_settings.eps_dual);
    }

    /** default regularisation: do nothing */
    EIGEN_STRONG_INLINE void hessian_regularisation_impl(Eigen::SparseMatrix<scalar_t>& lag_hessian) noexcept {}
};

template<typename Problem, typename QPSolver>
class SQPSolver : public SQPBase<SQPSolver<Problem, QPSolver>, Problem, QPSolver>
{
    using SQPBase<SQPSolver<Problem, QPSolver>, Problem, QPSolver>::SQPBase;
};

} // namespace laopt

#endif // LAOPT_SQP_SOLVER_HPP
