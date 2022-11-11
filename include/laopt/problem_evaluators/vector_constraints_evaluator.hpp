#ifndef LAOPT_VECTOR_CONSTRAINTS_EVALUATOR_HPP
#define LAOPT_VECTOR_CONSTRAINTS_EVALUATOR_HPP

#include "../problem_dispatch_types.hpp"
#include "../problem_vector_function.hpp"
#include "../problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename DType, typename Matrix, typename Vector>
class VectorConstraintsEvaluator
{
    VectorFunction<Matrix, Vector>& constraints;
    int row_offset;

public:
    explicit VectorConstraintsEvaluator(VectorFunction<Matrix, Vector>& constraints) : constraints(constraints), row_offset(0) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_variable(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj(Args...) {}

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        constraints.value(out_indices) = ExprEvaluator<Derived>::function(const_expr.derived());
        constraints.assign_lower_bound(out_indices, ExprEvaluator<Derived>::lower_bound(const_expr.derived()));
        constraints.assign_upper_bound(out_indices, ExprEvaluator<Derived>::upper_bound(const_expr.derived()));
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Jacobian>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        ExprEvaluator<Derived>::jacobian(const_expr.derived(), constraints.jacobian(out_indices, const_expr.derived().indices()));
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<Derived>::value>::type
    add_constr(ConstraintExpr<Derived>) {}
};

} // namespace laopt

#endif // LAOPT_VECTOR_CONSTRAINTS_EVALUATOR_HPP
