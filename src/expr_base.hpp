#ifndef LAMPC_EXPR_BASE_HPP
#define LAMPC_EXPR_BASE_HPP

namespace lampc {

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
};

} // namespace lampc

#endif //LAMPC_EXPR_BASE_HPP
