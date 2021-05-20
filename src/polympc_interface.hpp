#ifndef POLYMPC_INTERFACE_HPP
#define POLYMPC_INTERFACE_HPP

#include "solvers/nlproblem.hpp"
#include "solvers/sqp_base.hpp"
#include "utils/helpers.hpp"

#include <iostream>

template<typename problem_t, int MatrixFormat> struct LAProblemBase;

template<typename problem_t, int MatrixFormat = DENSE>
struct LAProblemBase
{

    LAProblemBase()
    {
    }

    ~LAProblemBase() = default;

    using param_t = typename problem_t::param_t;
    param_t param;
    problem_t problem;

    enum
    {
        /** problem dimensions */
        VAR_SIZE  = problem_t::num_variables,
        NUM_EQ    = problem_t::num_equalities,
        NUM_INEQ  = problem_t::num_inequalities,
        NUM_BOX   = problem_t::num_variables,
        DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,

        /** Various flags */
        is_sparse = (MatrixFormat == SPARSE) ? 1 : 0,
        is_dense  = is_sparse ? 0 : 1,
        MATRIXFMT = MatrixFormat
    };

    using scalar_t = typename problem_t::scalar_t;

    /** NLP variables */
    using nlp_variable_t         = typename dense_matrix_type_selector<scalar_t, VAR_SIZE, 1>::type;
    using nlp_eq_constraints_t   = typename dense_matrix_type_selector<scalar_t, NUM_EQ, 1>::type;
    using nlp_ineq_constraints_t = typename dense_matrix_type_selector<scalar_t, NUM_INEQ, 1>::type;
    using nlp_constraints_t      = typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, 1>::type;

    // choose to allocate sparse or dense jacoabian and hessian
    using nlp_eq_jacobian_t      = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                   typename dense_matrix_type_selector<scalar_t, NUM_EQ, VAR_SIZE>::type>::type;
    using nlp_ineq_jacobian_t    = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                   typename dense_matrix_type_selector<scalar_t, NUM_INEQ, VAR_SIZE>::type>::type;

    using nlp_jacobian_t         = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                   typename dense_matrix_type_selector<scalar_t, NUM_EQ + NUM_INEQ, VAR_SIZE>::type>::type;

    using nlp_hessian_t          = typename std::conditional<is_sparse, Eigen::SparseMatrix<scalar_t>,
                                   typename dense_matrix_type_selector<scalar_t, VAR_SIZE, VAR_SIZE>::type>::type;
    using nlp_cost_t             = scalar_t;
    using nlp_dual_t             = typename dense_matrix_type_selector<scalar_t, DUAL_SIZE, 1>::type;
    using static_parameter_t     = typename dense_matrix_type_selector<scalar_t, 0, 1>::type;


    /**  NLP interface functions */
    template<typename solver_t>
    EIGEN_STRONG_INLINE void setBounds(solver_t& solver)
    {
        setBounds(solver.lower_bound_x(), solver.upper_bound_x(),
                  solver.lower_bound_g(), solver.upper_bound_g());
    }

    EIGEN_STRONG_INLINE void setBounds(Eigen::Ref<nlp_variable_t> lb_x, Eigen::Ref<nlp_variable_t> ub_x,
                                       Eigen::Ref<nlp_ineq_constraints_t> lb_g, Eigen::Ref<nlp_ineq_constraints_t> ub_g)
    {
        problem.setBounds(param, lb_x, ub_x, lb_g, ub_g);
    }


    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, 
                                  const Eigen::Ref<const static_parameter_t>& p, 
                                  scalar_t &cost) noexcept
    {
        cost = problem.objective(param, var);
    }

    EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, 
                                           const Eigen::Ref<const static_parameter_t>& p,
                                           scalar_t &_cost, 
                                           Eigen::Ref<nlp_variable_t> cost_gradient_) noexcept
    {
        _cost = problem.objective(param, var, cost_gradient_);
    }

    EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, 
                                                   const Eigen::Ref<const static_parameter_t>& p,
                                                   scalar_t &_cost, 
                                                   Eigen::Ref<nlp_variable_t> cost_gradient, 
                                                   Eigen::Ref<nlp_hessian_t> hessian) noexcept
    {
        _cost = problem.objective(param, var, cost_gradient, hessian);
    }

    EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, 
                                          const Eigen::Ref<const static_parameter_t>& p,
                                          Eigen::Ref<nlp_ineq_constraints_t> inequalities) const noexcept
    {
        problem.inequalities(param, var, inequalities);
    }

    template<int NI = NUM_INEQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NI >= 1 >::type inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                         const Eigen::Ref<const static_parameter_t>& p,
                                                                                         Eigen::Ref<nlp_ineq_constraints_t> inequalities,
                                                                                         Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
    {
        problem.inequalities(param, var, inequalities, jacobian);
    }

    template<int NI = NUM_INEQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NI < 1 >::type inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                        const Eigen::Ref<const static_parameter_t>& p,
                                                                                        Eigen::Ref<nlp_ineq_constraints_t> inequalities,
                                                                                        Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept
    {
        /** @badcode : remove setting to zero? */
        //jacobian   = nlp_eq_jacobian_t::Zero(NUM_INEQ, VAR_SIZE);
        //inequalities = nlp_eq_constraints_t::Zero(NUM_INEQ);

        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
    }


    EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, 
                                        const Eigen::Ref<const static_parameter_t>& p,
                                        Eigen::Ref<nlp_eq_constraints_t> equalities) const noexcept
    {
        problem.equalities(param, var, equalities);
    }

    template<int NE = NUM_EQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NE >= 1 >::type equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                       const Eigen::Ref<const static_parameter_t>& p,
                                                                                       Eigen::Ref<nlp_eq_constraints_t> equalities,
                                                                                       Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        problem.equalities(param, var, equalities, jacobian);
    }

    template<int NE = NUM_EQ>
    EIGEN_STRONG_INLINE typename std::enable_if< NE < 1 >::type equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                                                      const Eigen::Ref<const static_parameter_t>& p,
                                                                                      Eigen::Ref<nlp_eq_constraints_t> equalities,
                                                                                      Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        /** @badcode : remove setting to zero? */
        jacobian   = nlp_eq_jacobian_t::Zero(NUM_EQ, VAR_SIZE);
        equalities = nlp_eq_constraints_t::Zero(NUM_EQ);

        polympc::ignore_unused_var(var);
        polympc::ignore_unused_var(p);
    }

    EIGEN_STRONG_INLINE void lagrangian(const Eigen::Ref<const nlp_variable_t>& var,
                                        const Eigen::Ref<const static_parameter_t>& p,
                                        const Eigen::Ref<const nlp_dual_t>& lam,
                                        scalar_t &_lagrangian) const noexcept
    {
        _lagrangian = problem.lagrangian(param, var, 1.0,
                                         lam.template head<NUM_EQ>(),
                                         lam.template segment<NUM_INEQ>(NUM_EQ),
                                         lam.template tail<NUM_BOX>());
    }

    EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                 Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept
    {
        _lagrangian = problem.lagrangian(param, var, 1.0,
                                         lam.template head<NUM_EQ>(),
                                         lam.template segment<NUM_INEQ>(NUM_EQ),
                                         lam.template tail<NUM_BOX>(),
                                         _lag_gradient);
    }

    EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var,
                                                 const Eigen::Ref<const static_parameter_t>& p,
                                                 const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                 Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_variable_t> cost_gradient,
                                                 Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
    {
        // TODO: We're computing the jacobian anyway - use it to avoid re-computation in the lagrangian
        _lagrangian = problem.lagrangian(param, var, 1.0,
                                         lam.template head<NUM_EQ>(),
                                         lam.template segment<NUM_INEQ>(NUM_EQ),
                                         lam.template tail<NUM_BOX>(),
                                         lag_gradient);
        problem.objective(param, var, cost_gradient);
        problem.constraints(param, var, g, jac_g);
    }

    EIGEN_STRONG_INLINE void lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var, const Eigen::Ref<const static_parameter_t> &p,
                                                         const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                         Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                         Eigen::Ref<nlp_variable_t> cost_gradient,
                                                         Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g) noexcept
    {
        _lagrangian = problem.lagrangian(param, var, 1.0, 
                                         lam.template head<NUM_EQ>(),
                                         lam.template segment<NUM_INEQ>(NUM_EQ),
                                         lam.template tail<NUM_BOX>(),
                                         lag_gradient, lag_hessian);
        problem.objective(param, var, cost_gradient);
        problem.constraints(param, var, g, jac_g);
    }
};


/** create solver */
template<typename Problem> class SQPSolver;
template<typename Problem>
class SQPSolver : public SQPBase<SQPSolver<Problem>, Problem>
{};


#endif // POLYMPC_INTERFACE_HPP
