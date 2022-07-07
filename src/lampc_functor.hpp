#ifndef LAMPC_LAMPC_FUNCTOR_HPP
#define LAMPC_LAMPC_FUNCTOR_HPP

#include "lampc_function_tag.hpp"

namespace lampc {

template<typename Derived, typename Tag>
class Function
{
    Derived& derived;
public:
    explicit Function(Derived& derived) : derived(derived) {};

    template<typename... Args>
    EIGEN_STRONG_INLINE void
    function(DefaultTag, Args&&... args) noexcept
    {
        derived.function(Tag{}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE void
    function(Args&&... args) noexcept
    {
        derived.function(Tag{}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE void
    operator()(Args&&... args) noexcept
    {
        derived.function(Tag{}, std::forward<Args>(args)...);
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
    EIGEN_STRONG_INLINE auto
    gradient(DefaultTag, Args&&... args) noexcept
    {
        return derived.gradient(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE auto
    gradient(Args&&... args) noexcept
    {
        return derived.gradient(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE auto
    hessian(DefaultTag, Args&&... args) noexcept
    {
        return derived.hessian(Tag{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    EIGEN_STRONG_INLINE auto
    hessian(Args&&... args) noexcept
    {
        return derived.hessian(Tag{}, std::forward<Args>(args)...);
    }
};

} // namespace lampc

#endif //LAMPC_LAMPC_FUNCTOR_HPP
