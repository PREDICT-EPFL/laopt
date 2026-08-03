#ifndef LAOPT_VARIABLE_BOUNDS_EVALUATOR_HPP
#define LAOPT_VARIABLE_BOUNDS_EVALUATOR_HPP

#include "laopt/problem_dispatch_types.hpp"
#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename MatrixType, typename VectorType>
class VariableBoundsEvaluator : public OptProblem<VariableBoundsEvaluator<MatrixType, VectorType>>
{
    friend OptProblem<VariableBoundsEvaluator<MatrixType, VectorType>>;

    VectorFunction<MatrixType, VectorType>& variable_bounds;

public:
    explicit VariableBoundsEvaluator(VectorFunction<MatrixType, VectorType>& variable_bounds) : variable_bounds(variable_bounds) {}

protected:
    template<typename DerivedLhs, typename DerivedRhs>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<IneqConstraintExpr<DerivedLhs, DerivedRhs>>::value &&
                                                is_variable<DerivedRhs>::value, void>::type
    add_constr_impl(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        variable_bounds.assign_lower_bound(variable_indices(ineq.rhs), ineq.lhs);
    }

    template<typename DerivedLhs, typename DerivedRhs>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<IneqConstraintExpr<DerivedLhs, DerivedRhs>>::value &&
                                                is_variable<DerivedLhs>::value, void>::type
    add_constr_impl(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        variable_bounds.assign_upper_bound(variable_indices(ineq.lhs), ineq.rhs);
    }

    template<typename DerivedLb, typename Derived, typename DerivedUb>
    EIGEN_STRONG_INLINE void add_constr_impl(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr)
    {
        this->add_constr(IneqConstraintExpr<DerivedLb, Derived>(bounded_expr.lb, bounded_expr.expr));
        this->add_constr(IneqConstraintExpr<Derived, DerivedUb>(bounded_expr.expr, bounded_expr.ub));
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE void add_constr_impl(const ConstraintExpr<Derived>&) {}
};

} // namespace laopt

#endif // LAOPT_VARIABLE_BOUNDS_EVALUATOR_HPP
