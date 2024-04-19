#ifndef LAOPT_SQP_SOLVER_HPP
#define LAOPT_SQP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "laopt/solvers/qp_base.hpp"

namespace laopt
{

enum struct globalization_t {
    FULL_STEP,
    LINE_SEARCH_L1,
    LINE_SEARCH_FILTER
};

enum struct hessian_approximation_t {
    EXACT,
    GAUSS_NEWTON,
};

template <typename Scalar>
struct sqp_settings_t {
    globalization_t globalization_strategy        = globalization_t::LINE_SEARCH_L1;
    Scalar tau                                    = 0.7;   // line search iteration decrease, 0 < tau < 1
    Scalar eta                                    = 0.25;  // line search parameter, 0 < eta < 1
    Scalar rho                                    = 0.5;   // line search parameter, 0 < rho < 1
    Scalar K                                      = 1e1;   // constant added to penalty parameter of line search
    Scalar eps_prim                               = 1e-6;  // primal step termination threshold, eps_prim > 0
    Scalar eps_dual                               = 1e-4;  // dual step termination threshold, eps_dual > 0
    Scalar min_alpha                              = 1e-4;  // minimum step size
    int max_watchdog_steps                        = 0;     // minimum full size steps for non-monotone watchdog strategy (0 for normal line search)
    Scalar filter_beta                             = 1e-5;  // sufficient decrease value for filter
    hessian_approximation_t hessian_approximation = hessian_approximation_t::GAUSS_NEWTON; // hessian approximation
    bool regularize_hessian                       = false; // regularize hessian using the gershgorin circle theorem (can lead to bad convergence)
    Scalar elastic_max_delta                      = 0.9;   // maximum acceptable value of constant relaxation variable, if below penalty gets increased by a factor elastic_increase and resolved
    Scalar elastic_min_delta                      = 1e-6;  // if constant relaxation variable falls below this value the penalty gets decreased by a factor elastic_decrease
    Scalar elastic_max_penalty                    = 1e7;   // maximum value of relaxation penalty
    Scalar elastic_min_penalty                    = 1.0;   // minimum value of relaxation penalty
    Scalar elastic_increase                       = 4.8;   // increase factor for relaxation penalty
    Scalar elastic_decrease                       = 0.25;  // decrease factor for relaxation penalty
    int max_iter                                  = 1000;
    int line_search_max_iter                      = 100;
    bool verbose                                  = false;

    bool validate()
    {
        return 0.0 < tau && tau < 1.0 &&
               0.0 < eta && eta < 1.0 &&
               0.0 < rho && rho < 1.0 &&
               K > 0 &&
               eps_prim > 0.0 &&
               eps_dual > 0.0 &&
               0.0 < min_alpha && min_alpha <= 1.0 &&
               max_watchdog_steps >= 0 &&
               elastic_max_delta > 0 &&
               elastic_min_delta > 0 &&
               elastic_max_penalty > 0 &&
               elastic_min_penalty > 0 &&
               elastic_increase > 1 &&
               elastic_decrease < 1 &&
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
    using matrix_t = typename Problem::matrix_t::SparsityType;

protected:
    Problem& prob;

    Eigen::VectorX<scalar_t> m_x;            // variable primal
    Eigen::VectorX<scalar_t> m_lam;          // dual variable
    Eigen::VectorX<scalar_t> m_lam_bounds;   // dual variable for bounds
    Eigen::VectorX<scalar_t> m_lbx, m_ubx;   // variable bounds
    Eigen::VectorX<scalar_t> m_cost_grad;    // Gradient of the cost function
    matrix_t                 m_lag_hess;     // Hessian of Lagrangian
    Eigen::VectorX<scalar_t> m_g;            // constraints evaluation
    matrix_t                 m_g_jac;        // constraints Jacobian
    Eigen::VectorX<scalar_t> m_lbg, m_ubg;   // equality/inequality constraints evaluated

    QPSolver m_qp_solver;
    sqp_settings_t<scalar_t> m_settings;
    sqp_info_t m_info;
    bool m_first_solve;

    Eigen::VectorX<scalar_t> m_p;             // primal search direction
    Eigen::VectorX<scalar_t> m_lam_qp;        // dual variable solution of qp
    Eigen::VectorX<scalar_t> m_lam_bounds_qp; // dual variable solution of qp for bounds

    scalar_t m_primal_feasibility_inf = 0;
    scalar_t m_complementarity_inf = 0;
    scalar_t m_stationarity_inf = 0;

    // internal variables for line search
    Eigen::VectorX<scalar_t> m_x_step_line_search;           // primal search step for line search
    int                      m_watchdog_step;                // tracks how many steps we are in for the watchdog strategy
    scalar_t                 m_mu_search_begin;              // mu from where we start the non-monotone watchdog strategy
    scalar_t                 m_phi_begin;                    // merit function evaluated at watch dog beginning
    scalar_t                 m_Dp_phi_begin;                 // directional derivative of merit function evaluated at watch dog beginning
    Eigen::VectorX<scalar_t> m_x_merit_decrease;             // last primal iterate with merit function decrease
    Eigen::VectorX<scalar_t> m_lam_merit_decrease;           // last dual iterate with merit function decrease
    Eigen::VectorX<scalar_t> m_lam_bounds_merit_decrease;    // last dual for bounds iterate with merit function decrease
    Eigen::VectorX<scalar_t> m_p_merit_decrease;             // last primal search iterate with merit function decrease
    Eigen::VectorX<scalar_t> m_lam_qp_merit_decrease;        // last dual variable solution of qp iterate with merit function decrease
    Eigen::VectorX<scalar_t> m_lam_bounds_qp_merit_decrease; // last dual variable solution of qp for bounds iterate with merit function decrease

    static constexpr int max_filter_elements = 20;
    int m_current_filter_elements = 0;
    Eigen::Matrix<scalar_t, 2, max_filter_elements> m_filter_list; // list of elements

    Eigen::VectorX<scalar_t> m_tmp_variables; // temporary work vector of size variables

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
        m_x_step_line_search(prob.variables()),
        m_watchdog_step(0),
        m_x_merit_decrease(prob.variables()),
        m_lam_merit_decrease(prob.constraints.rows()),
        m_lam_bounds_merit_decrease(prob.variables()),
        m_p_merit_decrease(prob.variables()),
        m_lam_qp_merit_decrease(prob.constraints.rows()),
        m_lam_bounds_qp_merit_decrease(prob.variables()),
        m_tmp_variables(prob.variables())
    {
        m_x.setZero();
        m_lam.setZero();
        m_lam_bounds.setZero();

        // reuse sparsity pattern for further solves
        m_qp_solver.settings().reuse_pattern = true;
        // turn on elastic mode by default to make qp problem always feasible
        m_qp_solver.settings().elastic_mode = true;

        prob.constraints.jacobian.allocate_memory(m_g_jac);

        prob.set_decision_variable(m_x);
    }

    EIGEN_STRONG_INLINE const sqp_settings_t<scalar_t>& settings() const noexcept { return m_settings; }
    EIGEN_STRONG_INLINE sqp_settings_t<scalar_t>& settings() noexcept { return m_settings; }

    EIGEN_STRONG_INLINE const sqp_info_t& info() const noexcept { return m_info; }
    EIGEN_STRONG_INLINE sqp_info_t& info() noexcept { return m_info; }

    EIGEN_STRONG_INLINE const QPSolver& qp_solver() const noexcept { return m_qp_solver; }
    EIGEN_STRONG_INLINE QPSolver& qp_solver() noexcept { return m_qp_solver; }

    /* Setters */
    void set_initial_primal(const Eigen::VectorX<scalar_t> &init_primal) { m_x = init_primal; }
    void set_initial_dual(const Eigen::VectorX<scalar_t> &init_dual) { m_lam = init_dual; }
    void set_initial_dual_bounds(const Eigen::VectorX<scalar_t> &init_dual_bounds) { m_lam_bounds = init_dual_bounds; }

    /* Getters */
    const Eigen::VectorX<scalar_t>& primal() const { return m_x; }
    const Eigen::VectorX<scalar_t>& dual() const { return m_lam; }
    const Eigen::VectorX<scalar_t>& dual_bounds() const { return m_lam_bounds; }

    scalar_t primal_feasibility() const { return m_primal_feasibility_inf; }
    scalar_t complementarity() const { return m_complementarity_inf; }
    scalar_t stationarity() const { return m_stationarity_inf; }

    /** solve the NLP */
    sqp_info_t solve() noexcept
    {
        if (m_first_solve) {
            m_first_solve = false;

            if (m_settings.hessian_approximation == hessian_approximation_t::GAUSS_NEWTON)
            {
                prob.objective.hessian.allocate_memory(m_lag_hess);
            }
            else
            {
                prob.lagrangian.hessian.allocate_memory(m_lag_hess);
            }

            if (m_settings.verbose)
            {
                printf("----------------------------------------------------------\n");
                printf("                        laOPT SQP                         \n");
                printf("    (c) Roland Schwan, Johannes Waibel, Colin N. Jones    \n");
                printf("   Ecole Polytechnique Federale de Lausanne (EPFL) 2024   \n");
                printf("----------------------------------------------------------\n");
                printf("variables n = %d\n", prob.variables());
                printf("constraints m = %d\n", prob.constraints.rows());
                printf("lagrangian hessian nnz = %ld\n", m_lag_hess.nonZeros());
                printf("constraints jacobian nnz = %ld\n", m_g_jac.nonZeros());
            }
        }

        // reset filter
        m_current_filter_elements = 0;

        if (!m_settings.validate())
        {
            m_info.status = sqp_status_t::INVALID_SETTINGS;
            if (m_settings.verbose) print_solve_info();
            return m_info;
        }

        scalar_t alpha = 0;

        m_info.qp_iter = 0;
        m_info.iter = 0;

        // linearize problem which is needed to calculate convergence criteria
        linearize_problem(m_x);

        // update convergence criteria
        m_primal_feasibility_inf = max_constraints_violation(m_x);
        m_complementarity_inf = max_complementarity_violation(m_x, m_lam, m_lam_bounds);
        m_stationarity_inf = max_stationarity_violation(m_lam, m_lam_bounds);

        if (m_settings.verbose)
        {
            printf("iter    objective     primal_inf    comp_inf      stat_inf      alpha       qp_iter");
            if (m_settings.max_watchdog_steps > 0)
            {
                printf("   watchdog");
            }
            if (m_qp_solver.settings().elastic_mode) {
                printf("   elastic");
            }
            printf("\n");

            prob.set_decision_variable(m_x);

            printf("%4d   % .5e   %.5e   %.5e   %.5e   %.3e   %7d",
                   m_info.iter,
                   prob.eval_objective(),
                   m_primal_feasibility_inf,
                   m_complementarity_inf,
                   m_stationarity_inf,
                   alpha,
                   0);
            if (m_settings.max_watchdog_steps > 0)
            {
                printf("   %4d", m_watchdog_step);
            }
            if (m_qp_solver.settings().elastic_mode) {
                printf("   %.5f", m_qp_solver.elastic_var_solution());
            }
            printf("\n");


        }

        if (termination_criteria()) {
            m_info.status = sqp_status_t::SOLVED;
            if (m_settings.verbose) print_solve_info();
            return m_info;
        }

        while (m_info.iter < m_settings.max_iter)
        {
            m_info.iter++;

            // problem is already linearized, we just need to calculate the hessian
            calculate_regularized_problem_hessian(m_x);
            solve_qp();

            if (m_qp_solver.settings().elastic_mode)
            {
                while (m_qp_solver.elastic_var_solution() > m_settings.elastic_max_delta &&
                       m_qp_solver.settings().elastic_weight_l2 < m_settings.elastic_max_penalty)
                {
                    m_qp_solver.settings().elastic_weight_l1 = fmin(m_settings.elastic_max_penalty, m_qp_solver.settings().elastic_weight_l1 * m_settings.elastic_increase);
                    m_qp_solver.settings().elastic_weight_l2 = fmin(m_settings.elastic_max_penalty, m_qp_solver.settings().elastic_weight_l2 * m_settings.elastic_increase);
                    solve_qp();
                }

                if (m_qp_solver.elastic_var_solution() < m_settings.elastic_min_delta)
                {
                    m_qp_solver.settings().elastic_weight_l1 = fmax(m_settings.elastic_min_penalty, m_qp_solver.settings().elastic_weight_l1 * m_settings.elastic_decrease);
                    m_qp_solver.settings().elastic_weight_l2 = fmax(m_settings.elastic_min_penalty, m_qp_solver.settings().elastic_weight_l2 * m_settings.elastic_decrease);
                }
            }

            if (m_settings.verbose && m_qp_solver.info().status == qp_status_t::MAX_ITER_REACHED)
            {
                std::cerr << "[SQPSolver::solve] QP problem has reached maximum iterations. Continue using suboptimal solution." << std::endl;
            }
            if (m_settings.verbose && m_qp_solver.info().status == qp_status_t::INFEASIBLE)
            {
                std::cerr << "[SQPSolver::solve] QP problem was infeasible. Continue using suboptimal solution." << std::endl;
            }

            if (!(m_qp_solver.info().status == qp_status_t::SOLVED ||
                  m_qp_solver.info().status == qp_status_t::MAX_ITER_REACHED ||
                  m_qp_solver.info().status == qp_status_t::INFEASIBLE))
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
                if (m_settings.verbose) print_solve_info();
                return m_info;
            }

            // calculate step size
            alpha = fmax(m_settings.min_alpha, step_size_selection(m_p));

            // take step
            m_x          += alpha * m_p;
            m_lam        += alpha * (m_lam_qp - m_lam);
            m_lam_bounds += alpha * (m_lam_bounds_qp - m_lam_bounds);

            // linearize problem which is needed to calculate convergence criteria
            linearize_problem(m_x);

            // update convergence criteria
            m_primal_feasibility_inf = max_constraints_violation(m_x);
            m_complementarity_inf = max_complementarity_violation(m_x, m_lam, m_lam_bounds);
            m_stationarity_inf = max_stationarity_violation(m_lam, m_lam_bounds);

            if (m_settings.verbose)
            {
                prob.set_decision_variable(m_x);
                printf("%4d   % .5e   %.5e   %.5e   %.5e   %.3e   %7d",
                       m_info.iter,
                       prob.eval_objective(),
                       m_primal_feasibility_inf,
                       m_complementarity_inf,
                       m_stationarity_inf,
                       alpha,
                       m_qp_solver.info().iter);
                if (m_settings.max_watchdog_steps > 0)
                {
                    printf("   %4d", m_watchdog_step);
                }
                if (m_qp_solver.settings().elastic_mode) {
                    printf("   %.5f", m_qp_solver.elastic_var_solution());
                }
                printf("\n");
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

    EIGEN_STRONG_INLINE void eval_objective_gradient(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        prob.set_decision_variable(x);
        prob.eval_objective_gradient(m_cost_grad);
    }

    EIGEN_STRONG_INLINE void eval_constraints(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        prob.set_decision_variable(x);
        prob.eval_constraints(m_g, m_lbg, m_ubg);
        prob.eval_variable_bounds(m_lbx, m_ubx);
    }

    EIGEN_STRONG_INLINE void eval_constraints_jacobian(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        prob.set_decision_variable(x);
        Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer(m_g_jac.valuePtr(), m_g_jac.nonZeros());
        prob.eval_constraints_jacobian(jacobian_buffer);
    }

    EIGEN_STRONG_INLINE void linearize_problem(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        eval_objective_gradient(x);
        eval_constraints(x);
        eval_constraints_jacobian(x);
    }

    EIGEN_STRONG_INLINE void calculate_regularized_problem_hessian(Eigen::Ref<Eigen::VectorX<scalar_t>> x) noexcept
    {
        prob.set_decision_variable(x);

        Eigen::Map<Eigen::VectorX<scalar_t>> hessian_buffer(m_lag_hess.valuePtr(), m_lag_hess.nonZeros());
        if (m_settings.hessian_approximation == hessian_approximation_t::GAUSS_NEWTON)
        {
            prob.eval_objective_hessian(hessian_buffer);
        }
        else
        {
            prob.eval_lagrangian_hessian(1.0, m_lam, hessian_buffer);
        }

        // regularize hessian
        hessian_regularisation(m_lag_hess);
    }

    /** prepare and solve the qp */
    EIGEN_STRONG_INLINE void solve_qp() noexcept
    {
        // add constant part from linearization to bounds
        m_lbx -= m_x;
        m_ubx -= m_x;
        m_lbg -= m_g;
        m_ubg -= m_g;

        // replace inf values with finite values since some solver don't like them (e.g. HPIPM)
        scalar_t max_val = std::numeric_limits<scalar_t>::max();
        m_lbx = m_lbx.cwiseMin(max_val).cwiseMax(-max_val);
        m_ubx = m_ubx.cwiseMin(max_val).cwiseMax(-max_val);
        m_lbg = m_lbg.cwiseMin(max_val).cwiseMax(-max_val);
        m_ubg = m_ubg.cwiseMin(max_val).cwiseMax(-max_val);

        m_qp_solver.solve(m_lag_hess, m_cost_grad, m_lbx, m_ubx, m_g_jac, m_lbg, m_ubg);
        m_info.qp_iter += m_qp_solver.info().iter;

        // extract primal and dual solutions
        m_p = m_qp_solver.primal_solution();
        m_lam_qp = m_qp_solver.dual_solution();
        m_lam_bounds_qp = m_qp_solver.dual_bounds_solution();
    }

    EIGEN_STRONG_INLINE scalar_t
    l1_constraints_violation(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& x) noexcept
    {
        scalar_t c = scalar_t(0);

        // l <= g(x) <= u
        c += (m_lbg - m_g).cwiseMax(0.0).sum();
        c += (m_g - m_ubg).cwiseMax(0.0).sum();

        // l <= x <= u
        c += (m_lbx - x).cwiseMax(0.0).sum();
        c += (x - m_ubx).cwiseMax(0.0).sum();

        return c;
    }

    EIGEN_STRONG_INLINE scalar_t
    max_constraints_violation(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& x) noexcept
    {
        scalar_t c = scalar_t(0);

        // lbg <= g <= ubg
        c = fmax(c, (m_lbg - m_g).maxCoeff());
        c = fmax(c, (m_g - m_ubg).maxCoeff());

        // l <= x <= u
        c = fmax(c, (m_lbx - x).maxCoeff());
        c = fmax(c, (x - m_ubx).maxCoeff());

        return c;
    }

    EIGEN_STRONG_INLINE scalar_t
    max_complementarity_violation(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& x,
                                  const Eigen::Ref<const Eigen::VectorX<scalar_t>>& dual,
                                  const Eigen::Ref<const Eigen::VectorX<scalar_t>>& dual_bounds) noexcept
    {
        scalar_t c = scalar_t(0);

        // lbg <= g <= ubg
        c = fmax(c, (m_lbg - m_g).cwiseMax(0.0).cwiseProduct(dual).maxCoeff());
        c = fmax(c, (m_g - m_ubg).cwiseMax(0.0).cwiseProduct(dual).maxCoeff());

        // l <= x <= u
        c = fmax(c, (m_lbx - x).cwiseMax(0.0).cwiseProduct(dual_bounds).maxCoeff());
        c = fmax(c, (x - m_ubx).cwiseMax(0.0).cwiseProduct(dual_bounds).maxCoeff());

        return c;
    }

    EIGEN_STRONG_INLINE scalar_t
    max_stationarity_violation(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& dual,
                               const Eigen::Ref<const Eigen::VectorX<scalar_t>>& dual_bounds) noexcept
    {
        m_tmp_variables.noalias() = m_g_jac.transpose() * dual;
        return (m_cost_grad + m_tmp_variables + dual_bounds).cwiseAbs().maxCoeff();
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
    hessian_regularisation(matrix_t& lag_hessian)
    {
        static_cast<Derived*>(this)->hessian_regularisation_impl(lag_hessian);
    }

    /**
    * default implementations
    */

    EIGEN_STRONG_INLINE scalar_t
    step_size_selection_impl(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& p) noexcept
    {
        switch (m_settings.globalization_strategy) {
            case globalization_t::FULL_STEP:
                return scalar_t(1);
            case globalization_t::LINE_SEARCH_L1:
                return l1_line_search(p);
            case globalization_t::LINE_SEARCH_FILTER:
                return filter_line_search(p);
        }
        return scalar_t(1);
    }

    EIGEN_STRONG_INLINE scalar_t
    l1_line_search(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& p) noexcept
    {
        // we only use watchdog if QP solver succeeded
        if (m_settings.max_watchdog_steps > 0 && m_qp_solver.info().status == qp_status_t::SOLVED)
        {
            // we are at watchdog search beginning
            if (m_watchdog_step == 0)
            {
                eval_constraints(m_x);
                scalar_t constr_l1 = l1_constraints_violation(m_x);
                // for a full step m_lam_qp is lam_{k+1} and a good estimate for mu
                m_mu_search_begin = m_lam_qp.template lpNorm<Eigen::Infinity>() + m_settings.K;

                prob.set_decision_variable(m_x);
                scalar_t cost = prob.eval_objective();
                m_phi_begin = cost + m_mu_search_begin * constr_l1;
                m_Dp_phi_begin = m_cost_grad.dot(p) - m_mu_search_begin * constr_l1;

                // take full step
                m_x_step_line_search = m_x + p;

                prob.set_decision_variable(m_x_step_line_search);
                scalar_t cost_step = prob.eval_objective();
                eval_constraints(m_x_step_line_search);
                scalar_t phi_step = cost_step + m_mu_search_begin * l1_constraints_violation(m_x_step_line_search);

                if (phi_step > (m_phi_begin + m_settings.eta * m_Dp_phi_begin))
                {
                    // full step is not accepted; hence, we start watchdog search
                    // otherwise m_watchdog_step stays at 0, i.e., reset search
                    m_watchdog_step++;

                    // fallback is the beginning of the search
                    m_x_merit_decrease = m_x;
                    m_lam_merit_decrease = m_lam;
                    m_lam_bounds_merit_decrease = m_lam_bounds;
                    m_p_merit_decrease = m_p;
                    m_lam_qp_merit_decrease = m_lam_qp;
                    m_lam_bounds_qp_merit_decrease = m_lam_bounds_qp;
                }

                if (phi_step < 1e3 * m_phi_begin)
                {
                    // small check that we did not completely diverge

                    // we are taking full steps during watchdog search
                    return 1.0;
                }
                // sometimes it happens that a full steps diverges and then the QP solver fails
                // in this case we do a proper line search using the fallback strategy
            }
            else if (m_watchdog_step < m_settings.max_watchdog_steps)
            {
                // take full step
                m_x_step_line_search = m_x + p;

                prob.set_decision_variable(m_x_step_line_search);
                scalar_t cost_step = prob.eval_objective();
                eval_constraints(m_x_step_line_search);
                scalar_t phi_step = cost_step + m_mu_search_begin * l1_constraints_violation(m_x_step_line_search);

                if (phi_step <= (m_phi_begin + m_settings.eta * m_Dp_phi_begin))
                {
                    // step is accepted because it fulfills the decrease condition for the watchdog beginning
                    // reset watchdog
                    m_watchdog_step = 0;
                    // we are taking full steps during watchdog search
                    return 1.0;
                }
                else if (phi_step <= m_phi_begin)
                {
                    // keep track of merit decrease iterates for fall back strategy
                    m_x_merit_decrease = m_x;
                    m_lam_merit_decrease = m_lam;
                    m_lam_bounds_merit_decrease = m_lam_bounds;

                    // increase watchdog step
                    m_watchdog_step++;
                    // we are taking full steps during watchdog search
                    return 1.0;
                }
                else if (phi_step < 1e3 * m_phi_begin)
                {
                    // small check that we did not completely diverge

                    // increase watchdog step
                    m_watchdog_step++;
                    // we are taking full steps during watchdog search
                    return 1.0;
                }
                // sometimes it happens that a full steps diverges and then the QP solver fails
                // in this case we do a proper line search using the fallback strategy
            }
        }

        // watchdog strategy failed, we have to backtrack now...

        // first we start with a line search from current iterate
        eval_constraints(m_x);
        scalar_t constr_l1 = l1_constraints_violation(m_x);
        scalar_t mu = std::abs(m_cost_grad.dot(p)) / ((1 - m_settings.rho) * constr_l1) + m_settings.K;

        prob.set_decision_variable(m_x);
        scalar_t cost_current = prob.eval_objective();
        scalar_t phi_current = cost_current + mu * constr_l1;
        scalar_t Dp_phi_current = m_cost_grad.dot(p) - mu * constr_l1;

        scalar_t alpha = scalar_t(1.0);
        scalar_t phi_step = scalar_t(0);
        for (int i = 1; i < m_settings.line_search_max_iter; i++)
        {
            m_x_step_line_search = m_x + alpha * p;

            prob.set_decision_variable(m_x_step_line_search);
            scalar_t cost_step = prob.eval_objective();
            eval_constraints(m_x_step_line_search);
            phi_step = cost_step + mu * l1_constraints_violation(m_x_step_line_search);

            if (phi_step <= (phi_current + alpha * m_settings.eta * Dp_phi_current))
            {
                break;
            }
            else
            {
                alpha = m_settings.tau * alpha;
            }
        }

        if (m_settings.max_watchdog_steps > 0)
        {
            if (phi_current <= m_phi_begin || phi_step <= m_phi_begin + m_settings.eta * m_Dp_phi_begin)
            {
                // we accept step and reset watchdog
                m_watchdog_step = 0;
                return alpha;
            }

            // as a last resort we perform a line search on the last primal merit decrease iterate
            // in the worst case this is just the beginning of the watchdog search
            eval_constraints(m_x_merit_decrease);
            constr_l1 = l1_constraints_violation(m_x_merit_decrease);
            mu = std::abs(m_cost_grad.dot(p)) / ((1 - m_settings.rho) * constr_l1) + m_settings.K;

            prob.set_decision_variable(m_x_merit_decrease);
            cost_current = prob.eval_objective();
            phi_current = cost_current + mu * constr_l1;
            Dp_phi_current = m_cost_grad.dot(p) - mu * constr_l1;

            alpha = scalar_t(1.0);
            for (int i = 1; i < m_settings.line_search_max_iter; i++)
            {
                m_x_step_line_search = m_x_merit_decrease + alpha * p;

                prob.set_decision_variable(m_x_step_line_search);
                scalar_t cost_step = prob.eval_objective();
                eval_constraints(m_x_step_line_search);
                phi_step = cost_step + mu * l1_constraints_violation(m_x_step_line_search);

                if (phi_step <= (phi_current + alpha * m_settings.eta * Dp_phi_current))
                {
                    break;
                }
                else
                {
                    alpha = m_settings.tau * alpha;
                }
            }

            // migrate back current primal and dual variables to last merit decrease
            m_x = m_x_merit_decrease;
            m_lam = m_lam_merit_decrease;
            m_lam_bounds = m_lam_bounds_merit_decrease;
            m_p = m_p_merit_decrease;
            m_lam_qp = m_lam_qp_merit_decrease;
            m_lam_bounds_qp = m_lam_bounds_qp_merit_decrease;
        }

        // reset watchdog
        m_watchdog_step = 0;
        return alpha;
    }

    EIGEN_STRONG_INLINE scalar_t
    filter_line_search(const Eigen::Ref<const Eigen::VectorX<scalar_t>>& p) noexcept
    {
        scalar_t cost_current = prob.eval_objective();

        if (m_current_filter_elements == 0)
        {
            eval_constraints(m_x);
            scalar_t constraint_violation = l1_constraints_violation(m_x);
            m_filter_list(0, 0) = cost_current;
            m_filter_list(1, 0) = constraint_violation;
            m_current_filter_elements = 1;
        }

        scalar_t alpha = scalar_t(1.0);
        for (int i = 1; i < m_settings.line_search_max_iter; i++)
        {
            m_x_step_line_search = m_x + alpha * p;

            prob.set_decision_variable(m_x_step_line_search);
            scalar_t cost = prob.eval_objective();
            eval_constraints(m_x_step_line_search);
            scalar_t constraint_violation = l1_constraints_violation(m_x_step_line_search);

            bool violates_filter = false;
            for (int j = 0; j < m_current_filter_elements; j++)
            {
                scalar_t filter_cost = m_filter_list(0, j);
                scalar_t filter_constraint_violation = m_filter_list(1, j);
                if (cost > filter_cost && constraint_violation > filter_constraint_violation)
                {
                    violates_filter = true;
                    break;
                }
            }

            if (violates_filter)
            {
                alpha = m_settings.tau * alpha;
                continue;
            }

            scalar_t m = alpha * m_cost_grad.dot(p);
            if (m < scalar_t(0))
            {
                if (cost > cost_current + m_settings.eta * m)
                {
                    alpha = m_settings.tau * alpha;
                    continue;
                }
            }
            else
            {
                bool violates_reduced_filter = false;
                for (int j = 0; j < m_current_filter_elements; j++)
                {
                    scalar_t filter_cost = m_filter_list(0, j);
                    scalar_t filter_constraint_violation = m_filter_list(1, j);
                    if (cost > filter_cost - m_settings.filter_beta * filter_constraint_violation
                        && constraint_violation > (scalar_t(1) - m_settings.filter_beta) * filter_constraint_violation)
                    {
                        violates_reduced_filter = true;
                        break;
                    }
                }

                if (violates_reduced_filter)
                {
                    alpha = m_settings.tau * alpha;
                    continue;
                }
            }

            // remove dominated elements in filter
            int pointer = 0;
            for (int j = 0; j < m_current_filter_elements; j++)
            {
                scalar_t filter_cost = m_filter_list(0, j);
                scalar_t filter_constraint_violation = m_filter_list(1, j);
                if (cost > filter_cost || constraint_violation > filter_constraint_violation)
                {
                    // element is not dominated
                    // if there were dominated elements before we have to shift list
                    if (j > pointer)
                    {
                        m_filter_list(0, pointer) = filter_cost;
                        m_filter_list(1, pointer) = filter_constraint_violation;
                    }
                    pointer++;
                }
            }

            if (pointer < max_filter_elements)
            {
                // we have still space in the filter, let's add the current element
                m_filter_list(0, pointer) = cost;
                m_filter_list(1, pointer) = constraint_violation;
                pointer++;
            }
            else
            {
                // just replace the first element
                m_filter_list(0, 0) = cost;
                m_filter_list(1, 0) = constraint_violation;
            }

            if (pointer > 0)
            {
                m_current_filter_elements = pointer;
            }
            else
            {
                m_current_filter_elements = 1;
            }

            break;
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
    EIGEN_STRONG_INLINE void hessian_regularisation_impl(matrix_t& lag_hessian) noexcept
    {
        // gershgorin_bound is a reference to the temporary work variable
        Eigen::VectorX<scalar_t>& gershgorin_bound = m_tmp_variables;

        if (m_settings.regularize_hessian)
        {
            gershgorin_bound.setZero();

            for (int i = 0; i < lag_hessian.outerSize(); ++i)
            {
                for (typename matrix_t::InnerIterator it(lag_hessian, i); it; ++it)
                {
                    if (it.row() == it.col())
                    {
                        gershgorin_bound(it.row()) += it.value();
                    }
                    else
                    {
                        gershgorin_bound(it.row()) -= fabs(it.value());
                    }
                }
            }

            scalar_t eig_lower_bound = gershgorin_bound.minCoeff();
            scalar_t eig_offset = fmax(scalar_t(0), -eig_lower_bound);

            for (int i = 0; i < lag_hessian.outerSize(); ++i)
            {
                for (typename matrix_t::InnerIterator it(lag_hessian, i); it; ++it)
                {
                    if (it.row() == it.col())
                    {
                        it.valueRef() += eig_offset;
                    }
                }
            }
        }
    }
};

template<typename Problem, typename QPSolver>
class SQPSolver : public SQPBase<SQPSolver<Problem, QPSolver>, Problem, QPSolver>
{
    using SQPBase<SQPSolver<Problem, QPSolver>, Problem, QPSolver>::SQPBase;
};

} // namespace laopt

#endif // LAOPT_SQP_SOLVER_HPP
