#ifndef LAMPC_LAMPC_FUNCTION_LIBRARY_HPP
#define LAMPC_LAMPC_FUNCTION_LIBRARY_HPP

#include "lampc_function_tag.hpp"

namespace lampc {

//
// For a given function F with Tag, EQ<F, Tag> is the function eq(xp, x...) = -xp + F(Tag, x...)
//
template<typename F, typename Tag = DefaultTag>
struct EQ : public Differentiable<EQ<F, Tag>, true>
{
    F& f;

    explicit EQ(F& f) : f(f) {}

    template<typename XP, typename... X>
    EIGEN_STRONG_INLINE typename XP::PlainObject
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>&... x) noexcept
    {
        return f(Tag{}, x...) - xp;
    }

    template<typename OutValue, typename OutJacobian, typename XP, typename... X>
    EIGEN_STRONG_INLINE void
    jacobian_impl(OutValue& value, OutJacobian& jac, const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>&... x) noexcept
    {
        // Jacobian is [-I jac_f]
        constexpr int nx = XP::RowsAtCompileTime;
        f.jacobian(Tag{}, value, jac(Eigen::all, Eigen::seq(nx, Eigen::last)), x...);
        value -= xp;

        using scalar_t = typename Eigen::MatrixBase<XP>::Scalar;
        for(int i=0; i < value.rows(); i++)
        {
            jac(Eigen::seqN(i,Eigen::fix<1>), Eigen::seqN(i, Eigen::fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(-1);
        }
    }
};

template<typename F, typename Scalar, typename Tag = DefaultTag>
struct RK4 : public Differentiable<RK4<F, Scalar, Tag>, true>
{
    F& f;
    Scalar h;

    explicit RK4(F& f, Scalar step_size) : f(f), h(step_size) {}

    template<typename X, typename... Params>
    EIGEN_STRONG_INLINE typename X::PlainObject
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        using Vec = typename X::PlainObject;
        Vec k1 = f(Tag{}, x,          params...);
        Vec k2 = f(Tag{}, x+h*0.5*k1, params...);
        Vec k3 = f(Tag{}, x+h*0.5*k2, params...);
        Vec k4 = f(Tag{}, x+h*k3,     params...);
        return x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }
};

} // namespace lampc

#endif //LAMPC_LAMPC_FUNCTION_LIBRARY_HPP
