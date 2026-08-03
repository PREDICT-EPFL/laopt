#ifndef LAOPT_DIFFERENTIABLE_FUNCTOR_HPP
#define LAOPT_DIFFERENTIABLE_FUNCTOR_HPP

#include "laopt/autodiff/differentiable.hpp"

namespace laopt {

template<typename Derived, typename Tag>
class Functor
{
    Derived& derived;
public:
    explicit Functor(Derived& derived) : derived(derived) {};

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    function(DefaultTag, Args&&... args) noexcept
    {
        return derived.function(Tag{}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    function(Args&&... args) noexcept
    {
        return derived.function(Tag{}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    operator()(Args&&... args) noexcept
    {
        return derived.function(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian(DefaultTag, Args&&... args) noexcept
    {
        derived.jacobian(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian(Args&&... args) noexcept
    {
        derived.jacobian(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE auto
    wsum(DefaultTag, Args&&... args) noexcept
    {
        return derived.wsum(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE auto
    wsum(Args&&... args) noexcept
    {
        return derived.wsum(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE void
    gradient(DefaultTag, Args&&... args) noexcept
    {
        derived.gradient(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE void
    gradient(Args&&... args) noexcept
    {
        derived.gradient(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE void
    hessian(DefaultTag, Args&&... args) noexcept
    {
        derived.hessian(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE auto
    hessian(Args&&... args) noexcept
    {
        derived.hessian(Tag{}, std::forward<Args>(args)...);
    }
};

} // namespace laopt

#endif // LAOPT_DIFFERENTIABLE_FUNCTOR_HPP
