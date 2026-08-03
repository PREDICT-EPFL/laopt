#ifndef LAOPT_TOUCHABLE_DERIVATIVE_HPP
#define LAOPT_TOUCHABLE_DERIVATIVE_HPP

namespace laopt
{

#define LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED(FUNC) \
    friend inline TouchableDerivative FUNC(const TouchableDerivative& other) \
    { \
        return TouchableDerivative(1); \
    }

#define LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(FUNC) \
    friend inline TouchableDerivative FUNC(const TouchableDerivative& other) \
    { \
        return other; \
    } \

// type that is "touchable", meaning it records if some operations was
// performed with it relevant for giving non-zero derivatives
template<typename Scalar_>
class TouchableDerivative
{
protected:
    int m_value;

public:
    using Scalar = Scalar_;

    // by default, it is not touched
    TouchableDerivative() : m_value(0) {};

    // allow direct initialization
    TouchableDerivative(const Scalar& value) : m_value(1) {};

    TouchableDerivative(const TouchableDerivative& other) : m_value(other.m_value) {};

    inline const int& value() const { return m_value; }

    inline bool operator< (const TouchableDerivative& other) const { return false; }
    inline bool operator<=(const TouchableDerivative& other) const { return m_value == 0 && other.m_value == 0; }
    inline bool operator> (const TouchableDerivative& other) const { return false; }
    inline bool operator>=(const TouchableDerivative& other) const { return m_value == 0 && other.m_value == 0; }
    inline bool operator==(const TouchableDerivative& other) const { return m_value == 0 && other.m_value == 0; }
    inline bool operator!=(const TouchableDerivative& other) const { return m_value != 0 || other.m_value != 0; }

    friend std::ostream& operator<<(std::ostream& s, const TouchableDerivative& a)
    {
        return s << a.m_value;
    }

    inline TouchableDerivative& operator=(const TouchableDerivative& other) = default;

    inline TouchableDerivative& operator=(const Scalar& other)
    {
        m_value = 1;
        return *this;
    }

    inline TouchableDerivative operator+(const Scalar& other) const
    {
        return TouchableDerivative(1);
    }

    friend inline TouchableDerivative operator+(const Scalar& a, const TouchableDerivative& b)
    {
        return TouchableDerivative(1);
    }

    inline TouchableDerivative& operator+=(const Scalar& other)
    {
        m_value = 1;
        return *this;
    }

    inline TouchableDerivative operator+(const TouchableDerivative& other) const
    {
        TouchableDerivative result;
        result.m_value = m_value || other.m_value;
        return result;
    }

    inline TouchableDerivative& operator+=(const TouchableDerivative& other)
    {
        (*this) = (*this) + other;
        return *this;
    }

    inline TouchableDerivative operator-(const Scalar& b) const
    {
        return TouchableDerivative(1);
    }

    friend inline TouchableDerivative operator-(const Scalar& a, const TouchableDerivative& b)
    {
        return TouchableDerivative(1);
    }

    inline TouchableDerivative& operator-=(const Scalar& other)
    {
        m_value = 1;
        return *this;
    }

    inline TouchableDerivative operator-(const TouchableDerivative& other) const
    {
        TouchableDerivative result;
        result.m_value = m_value || other.m_value;
        return result;
    }

    inline TouchableDerivative& operator-=(const TouchableDerivative& other)
    {
        (*this) = (*this) - other;
        return *this;
    }

    inline TouchableDerivative operator-() const
    {
        return TouchableDerivative(*this);
    }

    inline TouchableDerivative operator*(const Scalar& other) const
    {
        return TouchableDerivative(*this);
    }

    friend inline TouchableDerivative operator*(const Scalar& a, const TouchableDerivative& b)
    {
        return TouchableDerivative(b);
    }

    inline TouchableDerivative& operator*=(const Scalar& other)
    {
        return *this;
    }

    inline TouchableDerivative operator*(const TouchableDerivative& other) const
    {
        TouchableDerivative result;
        result.m_value = m_value && other.m_value;
        return result;
    }

    inline TouchableDerivative& operator*=(const TouchableDerivative& other)
    {
        (*this) = (*this) * other;
        return *this;
    }

    inline TouchableDerivative operator/(const Scalar& other) const
    {
        return TouchableDerivative(*this);
    }

    friend inline TouchableDerivative operator/(const Scalar& a, const TouchableDerivative& b)
    {
        return TouchableDerivative(1);
    }

    inline TouchableDerivative& operator/=(const Scalar& other)
    {
        return *this;
    }

    inline TouchableDerivative operator/(const TouchableDerivative& other) const
    {
        return TouchableDerivative(*this);
    }

    inline TouchableDerivative& operator/=(const TouchableDerivative& other)
    {
        (*this) = (*this) * other;
        return *this;
    }

    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(abs)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(abs2)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(sqrt)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED(cos)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(sin)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED(exp)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED(log)

    friend inline TouchableDerivative pow(const TouchableDerivative& x, const Scalar& y)
    {
        return TouchableDerivative(1);
    }

    friend inline TouchableDerivative atan2(const TouchableDerivative& a, const TouchableDerivative& b)
    {
        return TouchableDerivative(1);
    }

    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(tan)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(asin)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED(acos)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(tanh)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD(sinh)
    LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED(cosh)
};

#undef LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_TOUCHED
#undef LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD

} // namespace laopt

namespace Eigen
{

template<typename Scalar> struct NumTraits<laopt::TouchableDerivative<Scalar>>
{
    typedef laopt::TouchableDerivative<Scalar> Real;
    typedef laopt::TouchableDerivative<Scalar> NonInteger;
    typedef laopt::TouchableDerivative<Scalar> Nested;
    typedef laopt::TouchableDerivative<Scalar> Literal;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };
};

} // namespace Eigen

#endif // LAOPT_TOUCHABLE_DERIVATIVE_HPP
