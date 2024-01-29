#pragma once

namespace laopt
{

template <typename Evaluator>
class OptProblem
{
public:
    template<typename Scalar, int n>
    EIGEN_STRONG_INLINE void add_variable(Variable<Scalar, n>& var)
    {
        static_cast<Evaluator*>(this)->add_variable_impl(var);
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE void add_obj(const ExprBase<Derived>& expr)
    {
        static_cast<Evaluator*>(this)->add_obj_impl(expr.derived());
    }

    template<typename Derived>
    EIGEN_STRONG_INLINE void add_constr(const ConstraintExpr<Derived>& expr)
    {
        static_cast<Evaluator*>(this)->add_constr_impl(expr.derived());
    }

protected:
    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_variable_impl(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj_impl(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr_impl(Args...) {}
};

} // namespace laopt
