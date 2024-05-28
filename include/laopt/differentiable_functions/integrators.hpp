#ifndef LAOPT_INTEGRATORS_HPP
#define LAOPT_INTEGRATORS_HPP

#include "laopt/autodiff/differentiable.hpp"
#include "laopt/differentiable_functions/identity.hpp"

namespace laopt {

// Explicit euler method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class ERK1 : public Differentiable<ERK1<F, Scalar, Tag, Options>, Options>
{
protected:
    class Integrator : public Differentiable<Integrator, Options>
    {
    protected:
        F& f;
        Scalar h;

    public:
        explicit Integrator(F& f, const Scalar& step_size) : f(f), h(step_size) {}

        template<typename X, typename... Params>
        EIGEN_STRONG_INLINE typename X::PlainObject
        function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
        {
            return x + h * f.function(Tag{}, x, params...);
        }
    };
    Identity id;

public:
    Integrator integrator;

    explicit ERK1(F& f, const Scalar& step_size) : integrator(f, step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE auto
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return integrator(x, params...) - id(xp);
    }
};
// Alias for ERK1
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
using Euler = ERK1<F, Scalar, Tag, Options>;

// Ralston's method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class ERK2 : public Differentiable<ERK2<F, Scalar, Tag, Options>, Options>
{
protected:
    class Integrator : public Differentiable<Integrator, Options>
    {
    protected:
        F& f;
        Scalar h;

    public:
        explicit Integrator(F& f, const Scalar& step_size) : f(f), h(step_size) {}

        template<typename X, typename... Params>
        EIGEN_STRONG_INLINE typename X::PlainObject
        function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
        {
            using State = typename X::PlainObject;
            State k1 = f.function(Tag{}, x,                      params...);
            State k2 = f.function(Tag{}, x + h * 2.0 / 3.0 * k1, params...);
            return x + h / 4.0 * (k1 + 3.0 * k2);
        }
    };
    Identity id;

public:
    Integrator integrator;

    explicit ERK2(F& f, const Scalar& step_size) : integrator(f, step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE auto
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return integrator(x, params...) - id(xp);
    }
};

// Ralston's third-order method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class ERK3 : public Differentiable<ERK3<F, Scalar, Tag, Options>, Options>
{
protected:
    class Integrator : public Differentiable<Integrator, Options>
    {
    protected:
        F& f;
        Scalar h;

    public:
        explicit Integrator(F& f, const Scalar& step_size) : f(f), h(step_size) {}

        template<typename X, typename... Params>
        EIGEN_STRONG_INLINE typename X::PlainObject
        function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
        {
            using State = typename X::PlainObject;
            State k1 = f.function(Tag{}, x,                 params...);
            State k2 = f.function(Tag{}, x + h * 0.5 * k1,  params...);
            State k3 = f.function(Tag{}, x + h * 0.75 * k2, params...);
            return x + h / 9.0 * (2.0 * k1 + 3.0 * k2 + 4.0 * k3);
        }
    };
    Identity id;

public:
    Integrator integrator;

    explicit ERK3(F& f, const Scalar& step_size) : integrator(f, step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE auto
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return integrator(x, params...) - id(xp);
    }
};

// Classic fourth-order Runge–Kutta method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class ERK4 : public Differentiable<ERK4<F, Scalar, Tag, Options>, Options>
{
protected:
    class Integrator : public Differentiable<Integrator, Options>
    {
    protected:
        F& f;
        Scalar h;

    public:
        explicit Integrator(F& f, const Scalar& step_size) : f(f), h(step_size) {}

        template<typename X, typename... Params>
        EIGEN_STRONG_INLINE typename X::PlainObject
        function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
        {
            using State = typename X::PlainObject;
            State k1 = f.function(Tag{}, x,                params...);
            State k2 = f.function(Tag{}, x + h * 0.5 * k1, params...);
            State k3 = f.function(Tag{}, x + h * 0.5 * k2, params...);
            State k4 = f.function(Tag{}, x + h * k3,       params...);
            return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    };
    Identity id;

public:
    Integrator integrator;

    explicit ERK4(F& f, const Scalar& step_size) : integrator(f, step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE auto
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return integrator(x, params...) - id(xp);
    }
};

// Implicit euler method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class IRK1 : public Differentiable<IRK1<F, Scalar, Tag, Options>, Options>
{
protected:
    class BackIntegrator : public Differentiable<BackIntegrator, Options>
    {
    protected:
        F& f;
        Scalar h;

    public:
        explicit BackIntegrator(F& f, const Scalar& step_size) : f(f), h(step_size) {}

        template<typename XP, typename... Params>
        EIGEN_STRONG_INLINE typename XP::PlainObject
        function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<Params>&... params) noexcept
        {
            return h * f.function(Tag{}, xp, params...) - xp;
        }
    };
    Identity id;

public:
    BackIntegrator back_integrator;

    explicit IRK1(F& f, const Scalar& step_size) : back_integrator(f, step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE auto
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return id(x) + back_integrator(xp, params...);
    }
};

// Implicit midpoint method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class IRK2 : public Differentiable<IRK2<F, Scalar, Tag, Options>, Options>
{
protected:
    F& f;
    Scalar h;

public:
    explicit IRK2(F& f, const Scalar& step_size) : f(f), h(step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE typename X::PlainObject
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return x + h * f.function(Tag{}, 0.5 * (x + xp), params...) - xp;
    }
};

// Implicit trapezoidal method
template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class ImplicitTrapezoidal : public Differentiable<ImplicitTrapezoidal<F, Scalar, Tag, Options>, Options>
{
protected:
    F& f;
    Scalar h;

public:
    explicit ImplicitTrapezoidal(F& f, const Scalar& step_size) : f(f), h(step_size) {}

    template<typename XP, typename X, typename... Params>
    EIGEN_STRONG_INLINE typename X::PlainObject
    function_impl(const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        return x + 0.5 * h * (f.function(Tag{}, x, params...) + f.function(Tag{}, xp, params...)) - xp;
    }
};

} // namespace laopt

#endif //LAOPT_INTEGRATORS_HPP
