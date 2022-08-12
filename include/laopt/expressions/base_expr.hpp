#ifndef LAOPT_BASE_EXPR_HPP
#define LAOPT_BASE_EXPR_HPP

namespace laopt {

template<typename Derived>
class BaseExpr {
public:
    Derived& derived()
    {
        return static_cast<Derived&>(*this);
    }

    const Derived& derived() const
    {
        return static_cast<const Derived&>(*this);
    }

    EIGEN_STRONG_INLINE auto indices() const
    {
        return static_cast<const Derived*>(this)->indices();
    }
};

} // namespace laopt

#endif // LAOPT_BASE_EXPR_HPP
