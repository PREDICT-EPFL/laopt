#ifndef LAOPT_WEIGHTED_SUM_CONSTRAINTS_EVALUATOR_HPP
#define LAOPT_WEIGHTED_SUM_CONSTRAINTS_EVALUATOR_HPP

#include "laopt/problem_dispatch_types.hpp"
#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename DType, typename Matrix, typename Vector>
class WeightedSumConstraintsEvaluator : public OptProblem<WeightedSumConstraintsEvaluator<DType, Matrix, Vector>>
{
    friend OptProblem<WeightedSumConstraintsEvaluator<DType, Matrix, Vector>>;

    WeightedSumFunction<Matrix, Vector>& constraints;
    int row_offset;

public:
    explicit WeightedSumConstraintsEvaluator(WeightedSumFunction<Matrix, Vector>& constraints) : constraints(constraints), row_offset(0) {}

protected:
    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        constraints.value += ExprEvaluator<Derived>::wsum(const_expr.derived(), constraints.weights(out_indices));
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Gradient>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        ExprEvaluator<Derived>::gradient(const_expr.derived(), constraints.gradient, constraints.weights(out_indices));
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Hessian>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        ExprEvaluator<Derived>::hessian(const_expr.derived(), constraints.hessian, constraints.weights(out_indices));
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>&) {}
};

} // namespace laopt

#endif // LAOPT_WEIGHTED_SUM_CONSTRAINTS_EVALUATOR_HPP
