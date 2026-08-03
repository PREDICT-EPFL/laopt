#ifndef LAOPT_OBJECTIVE_EVALUATOR_HPP
#define LAOPT_OBJECTIVE_EVALUATOR_HPP

#include "laopt/problem_dispatch_types.hpp"
#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename DType, typename MatrixType, typename VectorType>
class ObjectiveEvaluator : public OptProblem<ObjectiveEvaluator<DType, MatrixType, VectorType>>
{
    friend OptProblem<ObjectiveEvaluator<DType, MatrixType, VectorType>>;

    using scalar_t = typename VectorType::scalar_t;

    WeightedSumFunction<MatrixType, VectorType>& objective;
    const scalar_t obj_factor;

public:
    explicit ObjectiveEvaluator(WeightedSumFunction<MatrixType, VectorType>& objective, const scalar_t obj_factor = static_cast<scalar_t>(1)) :
        objective(objective), obj_factor(obj_factor) {}

protected:
    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value>::type
    add_obj_impl(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        Eigen::Vector<scalar_t, n_outputs> weights = Eigen::Vector<scalar_t, n_outputs>::Constant(obj_factor);

        objective.value += ExprEvaluator<Derived>::wsum(expr.derived(), weights);
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Gradient>::value>::type
    add_obj_impl(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        Eigen::Vector<scalar_t, n_outputs> weights = Eigen::Vector<scalar_t, n_outputs>::Constant(obj_factor);

        ExprEvaluator<Derived>::gradient(expr.derived(), objective.gradient, weights);
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Hessian>::value>::type
    add_obj_impl(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::RowsAtCompileTime;

        Eigen::Vector<scalar_t, n_outputs> weights = Eigen::Vector<scalar_t, n_outputs>::Constant(obj_factor);
        ExprEvaluator<Derived>::hessian(expr.derived(), objective.hessian, weights);
    }
};

} // namespace laopt

#endif // LAOPT_OBJECTIVE_EVALUATOR_HPP
