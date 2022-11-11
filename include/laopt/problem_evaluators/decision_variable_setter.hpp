#ifndef LAOPT_DECISION_VARIABLE_SETTER_HPP
#define LAOPT_DECISION_VARIABLE_SETTER_HPP

#include "../problem_vector_function.hpp"
#include "../problem_weighted_sum_function.hpp"

namespace laopt
{

template<typename Scalar>
class DecisionVariableSetter
{
    Eigen::Ref<Eigen::VectorX<Scalar>>& master_var;
    int offset = 0;

public:
    explicit DecisionVariableSetter(Eigen::Ref<Eigen::VectorX<Scalar>>& var) : master_var(var) {}

    template<int n>
    EIGEN_STRONG_INLINE void add_variable(Variable<Scalar, n>& var)
    {
        var.set_memory(offset, master_var.data());
        offset += n;
    }

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr(Args...) {}
};

} // namespace laopt

#endif // LAOPT_DECISION_VARIABLE_SETTER_HPP
