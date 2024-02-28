#ifndef LAOPT_DECISION_VARIABLE_SETTER_HPP
#define LAOPT_DECISION_VARIABLE_SETTER_HPP

#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename Scalar>
class DecisionVariableSetter : public OptProblem<DecisionVariableSetter<Scalar>>
{
    friend OptProblem<DecisionVariableSetter<Scalar>>;

    Eigen::Ref<Eigen::VectorX<Scalar>>& master_var;
    int offset = 0;

public:
    explicit DecisionVariableSetter(Eigen::Ref<Eigen::VectorX<Scalar>>& var) : master_var(var) {}

protected:
    template<int n>
    EIGEN_STRONG_INLINE void add_variable_impl(Variable<Scalar, n>& var)
    {
        var.set_memory(offset, master_var.data());
        offset += n;
    }
};

} // namespace laopt

#endif // LAOPT_DECISION_VARIABLE_SETTER_HPP
