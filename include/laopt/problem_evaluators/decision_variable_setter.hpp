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

    Scalar* master_var_ptr;
    int offset = 0;

public:
    explicit DecisionVariableSetter(Scalar* var_ptr) : master_var_ptr(var_ptr) {}

protected:
    template<int n>
    EIGEN_STRONG_INLINE void add_variable_impl(Variable<Scalar, n>& var)
    {
        if (master_var_ptr != nullptr) {
            var.set_memory(offset, master_var_ptr);
        } else {
            var.set_memory(0, nullptr);
        }
        offset += n;
    }
};

} // namespace laopt

#endif // LAOPT_DECISION_VARIABLE_SETTER_HPP
