#ifndef LAOPT_OBJECTIVE_EVALUATOR_HPP
#define LAOPT_OBJECTIVE_EVALUATOR_HPP

#include "laopt/problem_dispatch_types.hpp"
#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename DType, typename Matrix, typename Vector>
class ObjectiveEvaluator : public OptProblem<ObjectiveEvaluator<DType, Matrix, Vector>>
{
    friend OptProblem<ObjectiveEvaluator<DType, Matrix, Vector>>;

    using scalar_t = typename Vector::scalar_t;

    WeightedSumFunction<Matrix, Vector>& objective;
    const scalar_t obj_factor;

public:
    explicit ObjectiveEvaluator(WeightedSumFunction<Matrix, Vector>& objective, const scalar_t obj_factor = static_cast<scalar_t>(1)) :
        objective(objective), obj_factor(obj_factor) {}

protected:
    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value>::type
    add_obj_impl(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        Eigen::Vector<scalar_t, n_outputs> weights = Eigen::Vector<scalar_t, n_outputs>::Constant(obj_factor);

        objective.value += ExprEvaluator<Derived>::wsum(expr.derived(), weights);
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Gradient>::value>::type
    add_obj_impl(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        Eigen::Vector<scalar_t, n_outputs> weights = Eigen::Vector<scalar_t, n_outputs>::Constant(obj_factor);

        ExprEvaluator<Derived>::gradient(expr.derived(), objective.gradient(expr.derived().indices()), weights);
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Hessian>::value>::type
    add_obj_impl(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        Eigen::Vector<scalar_t, n_outputs> weights = Eigen::Vector<scalar_t, n_outputs>::Constant(obj_factor);
        auto in_indices = expr.derived().indices();

        ExprEvaluator<Derived>::hessian(expr.derived(), objective.hessian(in_indices, in_indices), weights);
    }
};

} // namespace laopt

#endif // LAOPT_OBJECTIVE_EVALUATOR_HPP
