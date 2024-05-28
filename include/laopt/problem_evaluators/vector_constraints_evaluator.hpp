#ifndef LAOPT_VECTOR_CONSTRAINTS_EVALUATOR_HPP
#define LAOPT_VECTOR_CONSTRAINTS_EVALUATOR_HPP

#include "laopt/utility.hpp"
#include "laopt/problem_dispatch_types.hpp"
#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename DType, typename MatrixType, typename VectorType>
class VectorConstraintsEvaluator : public OptProblem<VectorConstraintsEvaluator<DType, MatrixType, VectorType>>
{
    friend OptProblem<VectorConstraintsEvaluator<DType, MatrixType, VectorType>>;

    VectorFunction<MatrixType, VectorType>& constraints;
    int row_offset;

public:
    explicit VectorConstraintsEvaluator(VectorFunction<MatrixType, VectorType>& constraints) : constraints(constraints), row_offset(0) {}

protected:
    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        constraints.value(out_indices) = to_matrix_type(ExprEvaluator<Derived>::function(const_expr.derived()));
        constraints.assign_lower_bound(out_indices, ExprEvaluator<Derived>::lower_bound(const_expr.derived()));
        constraints.assign_upper_bound(out_indices, ExprEvaluator<Derived>::upper_bound(const_expr.derived()));
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Jacobian>::value && !is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>& const_expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        auto out_indices = Eigen::seqN(row_offset, Eigen::fix<n_outputs>);
        row_offset += n_outputs;

        ExprEvaluator<Derived>::jacobian(const_expr.derived(), constraints.jacobian(out_indices, Eigen::all), 1);
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>&) {}
};

} // namespace laopt

#endif // LAOPT_VECTOR_CONSTRAINTS_EVALUATOR_HPP
