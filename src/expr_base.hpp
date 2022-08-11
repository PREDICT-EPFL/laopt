#ifndef LAOPT_EXPR_BASE_HPP
#define LAOPT_EXPR_BASE_HPP

namespace laopt {

template<typename Derived>
class ExprBase {
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

#endif // LAOPT_EXPR_BASE_HPP
