#ifndef LAOPT_PROBLEM_SIZE_EVALUATOR_HPP
#define LAOPT_PROBLEM_SIZE_EVALUATOR_HPP

#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename MatrixType, typename VectorType>
class ProblemSizeEvaluator : public OptProblem<ProblemSizeEvaluator<MatrixType, VectorType>>
{
    friend OptProblem<ProblemSizeEvaluator<MatrixType, VectorType>>;

    VectorFunction<MatrixType, VectorType>& variable_bounds;
    WeightedSumFunction<MatrixType, VectorType>& objective;
    VectorFunction<MatrixType, VectorType>& constraints;
    WeightedSumFunction<MatrixType, VectorType>& lagrangian;

public:
    explicit ProblemSizeEvaluator(VectorFunction<MatrixType, VectorType>& variable_bounds,
                                  WeightedSumFunction<MatrixType, VectorType>& objective,
                                  VectorFunction<MatrixType, VectorType>& constraints,
                                  WeightedSumFunction<MatrixType, VectorType>& lagrangian) :
          variable_bounds(variable_bounds),
          objective(objective),
          constraints(constraints),
          lagrangian(lagrangian) {}

protected:
    template<typename Scalar, int n>
    EIGEN_STRONG_INLINE void add_variable_impl(Variable<Scalar, n>& var)
    {
        variable_bounds.extend_variables(n);
        variable_bounds.extend_rows(n);

        objective.extend_variables(n);
        constraints.extend_variables(n);
        lagrangian.extend_variables(n);
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE typename std::enable_if<!is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>&)
    {
        constraints.extend_rows(Derived::RowsAtCompileTime);
        lagrangian.extend_rows(Derived::RowsAtCompileTime);
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable_constraint_expr<Derived>::value>::type
    add_constr_impl(const ConstraintExpr<Derived>&) {}
};

} // namespace laopt

#endif // LAOPT_PROBLEM_SIZE_EVALUATOR_HPP
