#ifndef LAOPT_VARIABLE_BOUNDS_EVALUATOR_HPP
#define LAOPT_VARIABLE_BOUNDS_EVALUATOR_HPP

#include "../problem_dispatch_types.hpp"
#include "../problem_vector_function.hpp"
#include "../problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename Matrix, typename Vector>
class VariableBoundsEvaluator : public OptProblem<VariableBoundsEvaluator<Matrix, Vector>>
{
    friend OptProblem<VariableBoundsEvaluator<Matrix, Vector>>;

    VectorFunction<Matrix, Vector>& variable_bounds;

public:
    explicit VariableBoundsEvaluator(VectorFunction<Matrix, Vector>& variable_bounds) : variable_bounds(variable_bounds) {}

protected:
    template<typename DerivedLhs, typename DerivedRhs>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<IneqConstraintExpr<DerivedLhs, IndexedVector<DerivedRhs>>>::value, void>::type
    add_constr_impl(const IneqConstraintExpr<DerivedLhs, IndexedVector<DerivedRhs>>& ineq)
    {
        variable_bounds.assign_lower_bound(ineq.rhs.indices(), ineq.lhs);
    }

    template<typename DerivedLhs, typename DerivedRhs>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<IneqConstraintExpr<IndexedVector<DerivedLhs>, DerivedRhs>>::value, void>::type
    add_constr_impl(const IneqConstraintExpr<IndexedVector<DerivedLhs>, DerivedRhs>& ineq)
    {
        variable_bounds.assign_upper_bound(ineq.lhs.indices(), ineq.rhs);
    }

    template<typename DerivedLb, typename Derived, typename DerivedUb>
    EIGEN_STRONG_INLINE void add_constr_impl(const BoundedExpr<DerivedLb, IndexedVector<Derived>, DerivedUb>& bounded_expr)
    {
        this->add_constr(IneqConstraintExpr<DerivedLb, IndexedVector<Derived>>(bounded_expr.lb, bounded_expr.expr));
        this->add_constr(IneqConstraintExpr<IndexedVector<Derived>, DerivedUb>(bounded_expr.expr, bounded_expr.ub));
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE void add_constr_impl(const ConstraintExpr<Derived>&) {}
};

} // namespace laopt

#endif // LAOPT_VARIABLE_BOUNDS_EVALUATOR_HPP
