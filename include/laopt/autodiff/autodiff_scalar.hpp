#ifndef LAOPT_AUTODIFF_SCALAR_HPP
#define LAOPT_AUTODIFF_SCALAR_HPP

#include <Eigen/Dense>
#include "touchable_derivative.hpp"

#define EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(EXPR, SCALAR, OPNAME) \
  Eigen::CwiseBinaryOp<EIGEN_CAT(EIGEN_CAT(Eigen::internal::scalar_,OPNAME),_op)<typename Eigen::internal::traits<EXPR>::Scalar,SCALAR>, const EXPR, \
                const typename Eigen::internal::plain_constant_type<EXPR,SCALAR>::type>

namespace laopt
{

namespace internal
{

template<typename A, typename B>
struct laopt_make_coherent_impl
{
    static void run(A&, B&) {}
};

// resize a to match b is a.size()==0, and conversely.
template<typename A, typename B>
void laopt_make_coherent(const A& a, const B& b)
{
    laopt_make_coherent_impl<A, B>::run(a.const_cast_derived(), b.const_cast_derived());
}

template<typename DerivativeType, bool Enable>
struct auto_diff_special_op;

} // end namespace internal

/** \class AutoDiffScalar
  * \brief A scalar type replacement with automatic differentiation capability
  *
  * \param DerivativeType the vector type used to store/represent the derivatives. The base scalar type
  *                 as well as the number of derivatives to compute are determined from this type.
  *                 Typical choices include, e.g., \c Vector4f for 4 derivatives, or \c VectorXf
  *                 if the number of derivatives is not known at compile time, and/or, the number
  *                 of derivatives is large.
  *                 Note that DerivativeType can also be a reference (e.g., \c VectorXf&) to wrap a
  *                 existing vector into an AutoDiffScalar.
  *                 Finally, DerivativeType can also be any Eigen compatible expression.
  *
  * This class represents a scalar value while tracking its respective derivatives using Eigen's expression
  * template mechanism.
  *
  * It supports the following list of global math function:
  *  - std::abs, std::sqrt, std::pow, std::exp, std::log, std::sin, std::cos,
  *  - internal::abs, internal::sqrt, numext::pow, internal::exp, internal::log, internal::sin, internal::cos,
  *  - internal::conj, internal::real, internal::imag, numext::abs2.
  *
  * AutoDiffScalar can be used as the scalar type of an Eigen::Matrix object. However,
  * in that case, the expression template mechanism only occurs at the top Matrix level,
  * while derivatives are computed right away.
  *
  */
template<typename DerivativeType, typename Enable = void>
class AutoDiffScalar;

template<typename NewDerType>
inline AutoDiffScalar<NewDerType> laopt_make_autodiff_scalar(const typename NewDerType::Scalar& value, const NewDerType& der)
{
    return AutoDiffScalar<NewDerType>(value, der);
}

template<typename DerivativeType>
class AutoDiffScalarBase
    : public internal::auto_diff_special_op
        <DerivativeType, !Eigen::internal::is_same<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerivativeType>::type>::Scalar,
            typename Eigen::NumTraits<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerivativeType>::type>::Scalar>::Real>::value>
{
public:
    typedef internal::auto_diff_special_op
        <DerivativeType, !Eigen::internal::is_same<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerivativeType>::type>::Scalar,
            typename Eigen::NumTraits<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerivativeType>::type>::Scalar>::Real>::value> Base;
    typedef typename Eigen::internal::remove_all<DerivativeType>::type DerType;
    typedef typename Eigen::internal::traits<DerType>::Scalar Scalar;
    typedef typename Eigen::NumTraits<Scalar>::Real Real;

    using Base::operator+;
    using Base::operator*;

protected:
    template<typename RealType>
    inline void init_derivatives(const RealType&)
    {
        if (m_derivatives.size() > 0)
        {
            m_derivatives.setZero();
        }
    }

    // special override for the TouchableDerivative type
    // to not "touch" the derivatives, we don't set them to zero
    template<typename TouchableDerivativeScalar>
    inline void init_derivatives(const TouchableDerivative<TouchableDerivativeScalar>&) {}

    // same for second order autodiff scalar types
    template<typename TouchableDerivativeScalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    inline void init_derivatives(const AutoDiffScalar<Eigen::Matrix<TouchableDerivative<TouchableDerivativeScalar>, Rows, Cols, Options, MaxRows, MaxCols>>&) {}

public:
    /** Default constructor without any initialization. */
    AutoDiffScalarBase() {}

    /** Constructs an active scalar from its \a value,
        and initializes the \a nbDer derivatives such that it corresponds to the \a derNumber -th variable */
    AutoDiffScalarBase(const Scalar& value, int nbDer, int derNumber)
        : m_value(value), m_derivatives(nbDer)
    {
        init_derivatives(value);
        m_derivatives.coeffRef(derNumber) = Scalar(1);
    }

    /** Conversion from a scalar constant to an active scalar.
      * The derivatives are set to zero. */
    /*explicit*/ AutoDiffScalarBase(const Real& value) : m_value(value)
    {
        init_derivatives(value);
    }

    /** Constructs an active scalar from its \a value and derivatives \a der */
    AutoDiffScalarBase(const Scalar& value, const DerType& der)
        : m_value(value), m_derivatives(der) {}

    template<typename OtherDerType>
    AutoDiffScalarBase(const AutoDiffScalar<OtherDerType>& other
#ifndef EIGEN_PARSED_BY_DOXYGEN
        , typename Eigen::internal::enable_if<
        Eigen::internal::is_same<Scalar, typename Eigen::internal::traits<typename Eigen::internal::remove_all<OtherDerType>::type>::Scalar>::value
        && Eigen::internal::is_convertible<OtherDerType, DerType>::value, void*>::type = 0
#endif
    )
        : m_value(other.value()), m_derivatives(other.derivatives()) {}

    friend std::ostream& operator<<(std::ostream& s, const AutoDiffScalarBase& a)
    {
        return s << a.value();
    }

    template<typename OtherDerType>
    inline AutoDiffScalarBase& operator=(const AutoDiffScalar<OtherDerType>& other)
    {
        m_value = other.value();
        m_derivatives = other.derivatives();
        return *this;
    }

    inline AutoDiffScalarBase& operator=(const Scalar& other)
    {
        m_value = other;
        if (m_derivatives.size() > 0)
            m_derivatives.setZero();
        return *this;
    }

    inline const Scalar& value() const { return m_value; }
    inline Scalar& value() { return m_value; }

    inline const DerType& derivatives() const { return m_derivatives; }
    inline DerType& derivatives() { return m_derivatives; }

    inline bool operator< (const Scalar& other) const { return m_value <  other; }
    inline bool operator<=(const Scalar& other) const { return m_value <= other; }
    inline bool operator> (const Scalar& other) const { return m_value >  other; }
    inline bool operator>=(const Scalar& other) const { return m_value >= other; }
    inline bool operator==(const Scalar& other) const { return m_value == other; }
    inline bool operator!=(const Scalar& other) const { return m_value != other; }

    friend inline bool operator< (const Scalar& a, const AutoDiffScalarBase& b) { return a <  b.value(); }
    friend inline bool operator<=(const Scalar& a, const AutoDiffScalarBase& b) { return a <= b.value(); }
    friend inline bool operator> (const Scalar& a, const AutoDiffScalarBase& b) { return a >  b.value(); }
    friend inline bool operator>=(const Scalar& a, const AutoDiffScalarBase& b) { return a >= b.value(); }
    friend inline bool operator==(const Scalar& a, const AutoDiffScalarBase& b) { return a == b.value(); }
    friend inline bool operator!=(const Scalar& a, const AutoDiffScalarBase& b) { return a != b.value(); }

    template<typename OtherDerType> inline bool operator< (const AutoDiffScalar<OtherDerType>& b) const { return m_value <  b.value(); }
    template<typename OtherDerType> inline bool operator<=(const AutoDiffScalar<OtherDerType>& b) const { return m_value <= b.value(); }
    template<typename OtherDerType> inline bool operator> (const AutoDiffScalar<OtherDerType>& b) const { return m_value >  b.value(); }
    template<typename OtherDerType> inline bool operator>=(const AutoDiffScalar<OtherDerType>& b) const { return m_value >= b.value(); }
    template<typename OtherDerType> inline bool operator==(const AutoDiffScalar<OtherDerType>& b) const { return m_value == b.value(); }
    template<typename OtherDerType> inline bool operator!=(const AutoDiffScalar<OtherDerType>& b) const { return m_value != b.value(); }

    inline const AutoDiffScalar<DerType&> operator+(const Scalar& other) const
    {
        return AutoDiffScalar<DerType&>(m_value + other, m_derivatives);
    }

    friend inline const AutoDiffScalar<DerType&> operator+(const Scalar& a, const AutoDiffScalar<DerivativeType>& b)
    {
        return AutoDiffScalar<DerType&>(a + b.value(), b.derivatives());
    }

    inline AutoDiffScalarBase& operator+=(const Scalar& other)
    {
        value() += other;
        return *this;
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar>, const DerType, const typename Eigen::internal::remove_all<OtherDerType>::type>>
    operator+(const AutoDiffScalar<OtherDerType>& other) const
    {
        internal::laopt_make_coherent(m_derivatives, other.derivatives());
        return AutoDiffScalar<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar>, const DerType, const typename Eigen::internal::remove_all<OtherDerType>::type>>(
            m_value + other.value(),
            m_derivatives + other.derivatives());
    }

    template<typename OtherDerType>
    inline AutoDiffScalarBase&
    operator+=(const AutoDiffScalar<OtherDerType>& other)
    {
        (*this) = (*this) + other;
        return *this;
    }

    inline const AutoDiffScalar<DerType&> operator-(const Scalar& b) const
    {
        return AutoDiffScalar<DerType&>(m_value - b, m_derivatives);
    }

    friend inline const AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, const DerType>>
    operator-(const Scalar& a, const AutoDiffScalar<DerivativeType>& b)
    {
        return AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, const DerType>>
            (a - b.value(), -b.derivatives());
    }

    inline AutoDiffScalarBase& operator-=(const Scalar& other)
    {
        value() -= other;
        return *this;
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar>, const DerType, const typename Eigen::internal::remove_all<OtherDerType>::type>>
    operator-(const AutoDiffScalar<OtherDerType>& other) const
    {
        internal::laopt_make_coherent(m_derivatives, other.derivatives());
        return AutoDiffScalar<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar>, const DerType, const typename Eigen::internal::remove_all<OtherDerType>::type>>(
            m_value - other.value(),
            m_derivatives - other.derivatives());
    }

    template<typename OtherDerType>
    inline AutoDiffScalarBase&
    operator-=(const AutoDiffScalar<OtherDerType>& other)
    {
        *this = *this - other;
        return *this;
    }

    inline const AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, const DerType>>
    operator-() const
    {
        return AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, const DerType>>(
            -m_value,
            -m_derivatives);
    }

    inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType, Scalar, product)>
    operator*(const Scalar& other) const
    {
        return laopt_make_autodiff_scalar(m_value * other, m_derivatives * other);
    }

    friend inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType, Scalar, product)>
    operator*(const Scalar& other, const AutoDiffScalar<DerivativeType>& a)
    {
        return laopt_make_autodiff_scalar(a.value() * other, a.derivatives() * other);
    }

    inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType, Scalar, product)>
    operator/(const Scalar& other) const
    {
        return laopt_make_autodiff_scalar(m_value / other, (m_derivatives * (Scalar(1) / other)));
    }

    friend inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType, Scalar, product)>
    operator/(const Scalar& other, const AutoDiffScalar<DerivativeType>& a)
    {
        return laopt_make_autodiff_scalar(other / a.value(), a.derivatives() * (Scalar(-other) / (a.value() * a.value())));
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(
        Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar> EIGEN_COMMA
        const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType, Scalar, product) EIGEN_COMMA
        const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(typename Eigen::internal::remove_all<OtherDerType>::type, Scalar,product)>, Scalar, product)>
    operator/(const AutoDiffScalar<OtherDerType>& other) const
    {
        internal::laopt_make_coherent(m_derivatives, other.derivatives());
        return laopt_make_autodiff_scalar(
            m_value / other.value(),
            ((m_derivatives * other.value()) - (other.derivatives() * m_value)) * (Scalar(1) / (other.value() * other.value())));
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar>,
        const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType, Scalar, product),
        const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(typename Eigen::internal::remove_all<OtherDerType>::type, Scalar, product)>>
    operator*(const AutoDiffScalar<OtherDerType>& other) const
    {
        internal::laopt_make_coherent(m_derivatives, other.derivatives());
        return laopt_make_autodiff_scalar(m_value * other.value(), (m_derivatives * other.value()) + (other.derivatives() * m_value));
    }

    inline AutoDiffScalarBase& operator*=(const Scalar& other)
    {
        *this = *this * other;
        return *this;
    }

    template<typename OtherDerType>
    inline AutoDiffScalarBase& operator*=(const AutoDiffScalar<OtherDerType>& other)
    {
        *this = *this * other;
        return *this;
    }

    inline AutoDiffScalarBase& operator/=(const Scalar& other)
    {
        *this = *this / other;
        return *this;
    }

    template<typename OtherDerType>
    inline AutoDiffScalarBase& operator/=(const AutoDiffScalar<OtherDerType>& other)
    {
        *this = *this / other;
        return *this;
    }

protected:
    Scalar m_value;
    DerType m_derivatives;

};

// struct to determine if type T has ::Scalar::DerType type defined
// used to add additional functions to AutoDiffScalar to make it compatible for second order derivatives
template<typename T, typename = void>
struct has_scalar_der_type : std::false_type {};
// suppress GCC warning: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90881
#if defined(__GNUC__) && (__GNUC__ >= 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
template<typename T>
struct has_scalar_der_type<T, decltype(sizeof(typename T::Scalar::DerType), void())> : std::true_type {};
#if defined(__GNUC__) && (__GNUC__ >= 7)
#pragma GCC diagnostic pop
#endif

template<typename DerivativeType>
class AutoDiffScalar<DerivativeType, typename std::enable_if<!has_scalar_der_type<AutoDiffScalarBase<DerivativeType>>::value>::type>
    : public AutoDiffScalarBase<DerivativeType>
{
public:
    using Base = class AutoDiffScalarBase<DerivativeType>;
    using Base::Base;
    using Base::operator=;
};

template<typename DerivativeType>
class AutoDiffScalar<DerivativeType, typename std::enable_if<has_scalar_der_type<AutoDiffScalarBase<DerivativeType>>::value>::type>
    : public AutoDiffScalarBase<DerivativeType>
{
public:
    typedef AutoDiffScalarBase<DerivativeType> Base;
    typedef typename Base::DerType DerType;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::Real Real;

    typedef typename Scalar::DerType InnerDerType;
    typedef typename Eigen::internal::traits<InnerDerType>::Scalar InnerScalar;
    typedef typename Eigen::NumTraits<InnerScalar>::Real InnerReal;

    using Base::Base;
    using Base::operator=;
    using Base::operator<;
    using Base::operator<=;
    using Base::operator>;
    using Base::operator>=;
    using Base::operator==;
    using Base::operator!=;
    using Base::operator+;
    using Base::operator+=;
    using Base::operator-;
    using Base::operator-=;
    using Base::operator*;
    using Base::operator*=;
    using Base::operator/;
    using Base::operator/=;

    AutoDiffScalar(const InnerScalar& value, int nbDer, int derNumber) : AutoDiffScalar(Scalar(value), nbDer, derNumber) {};

    /*explicit*/ AutoDiffScalar(const InnerReal& value) : AutoDiffScalar(Real(value)) {};

    inline AutoDiffScalar& operator=(const InnerScalar& other)
    {
        *this = Scalar(other);
        return *this;
    }

    inline bool operator< (const InnerScalar& other) const { return this->m_value <  Scalar(other); }
    inline bool operator<=(const InnerScalar& other) const { return this->m_value <= Scalar(other); }
    inline bool operator> (const InnerScalar& other) const { return this->m_value >  Scalar(other); }
    inline bool operator>=(const InnerScalar& other) const { return this->m_value >= Scalar(other); }
    inline bool operator==(const InnerScalar& other) const { return this->m_value == Scalar(other); }
    inline bool operator!=(const InnerScalar& other) const { return this->m_value != Scalar(other); }

    friend inline bool operator< (const InnerScalar& a, const AutoDiffScalar& b) { return Scalar(a) <  b.value(); }
    friend inline bool operator<=(const InnerScalar& a, const AutoDiffScalar& b) { return Scalar(a) <= b.value(); }
    friend inline bool operator> (const InnerScalar& a, const AutoDiffScalar& b) { return Scalar(a) >  b.value(); }
    friend inline bool operator>=(const InnerScalar& a, const AutoDiffScalar& b) { return Scalar(a) >= b.value(); }
    friend inline bool operator==(const InnerScalar& a, const AutoDiffScalar& b) { return Scalar(a) == b.value(); }
    friend inline bool operator!=(const InnerScalar& a, const AutoDiffScalar& b) { return Scalar(a) != b.value(); }

    inline const AutoDiffScalar<DerType&> operator+(const InnerScalar& other) const
    {
        return *this + Scalar(other);
    }

    friend inline const AutoDiffScalar<DerType&> operator+(const InnerScalar& a, const AutoDiffScalar& b)
    {
        return Scalar(a) + b;
    }

    inline AutoDiffScalar& operator+=(const InnerScalar& other)
    {
        *this = *this + other;
        return *this;
    }

    inline const AutoDiffScalar<DerType&> operator-(const InnerScalar& b) const
    {
        return *this - Scalar(b);
    }

    friend inline const AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, const DerType>>
    operator-(const InnerScalar& a, const AutoDiffScalar& b)
    {
        return Scalar(a) - b;
    }

    inline AutoDiffScalar& operator-=(const InnerScalar& other)
    {
        *this = *this - other;
        return *this;
    }

    inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType,Scalar,product)>
    operator*(const InnerScalar& other) const
    {
        return laopt_make_autodiff_scalar(this->m_value * Scalar(other), this->m_derivatives * Scalar(other));
    }

    friend inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType,Scalar,product)>
    operator*(const InnerScalar& other, const AutoDiffScalar& a)
    {
        return laopt_make_autodiff_scalar(a.value() * Scalar(other), a.derivatives() * Scalar(other));
    }

    inline AutoDiffScalar& operator*=(const InnerScalar& other)
    {
        *this = *this * other;
        return *this;
    }

    inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType,Scalar,product)>
    operator/(const InnerScalar& other) const
    {
        return laopt_make_autodiff_scalar(this->m_value / Scalar(other), (this->m_derivatives * (Scalar(1) / Scalar(other))));
    }

    friend inline const AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(DerType,Scalar,product)>
    operator/(const InnerScalar& other, const AutoDiffScalar& a)
    {
        return laopt_make_autodiff_scalar(Scalar(other) / a.value(), a.derivatives() * (Scalar(-other) / (a.value() * a.value())));
    }

    inline AutoDiffScalar& operator/=(const InnerScalar& other)
    {
        *this = *this / other;
        return *this;
    }
};

namespace internal
{

template<typename DerivativeType>
struct auto_diff_special_op<DerivativeType, true>
{
    typedef typename Eigen::internal::remove_all<DerivativeType>::type DerType;
    typedef typename Eigen::internal::traits<DerType>::Scalar Scalar;
    typedef typename Eigen::NumTraits<Scalar>::Real Real;

    const AutoDiffScalar<DerivativeType>&
    derived() const { return *static_cast<const AutoDiffScalar<DerivativeType>*>(this); }

    AutoDiffScalar<DerivativeType>& derived() { return *static_cast<AutoDiffScalar<DerivativeType>*>(this); }


    inline const AutoDiffScalar<DerType&> operator+(const Real& other) const
    {
        return AutoDiffScalar<DerType&>(derived().value() + other, derived().derivatives());
    }

    friend inline const AutoDiffScalar<DerType&> operator+(const Real& a, const AutoDiffScalar<DerivativeType>& b)
    {
        return AutoDiffScalar<DerType&>(a + b.value(), b.derivatives());
    }

    inline AutoDiffScalar<DerivativeType>& operator+=(const Real& other)
    {
        derived().value() += other;
        return derived();
    }


    inline const AutoDiffScalar<typename Eigen::CwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<Scalar, Real>>, DerType>::Type>
    operator*(const Real& other) const
    {
        return AutoDiffScalar<typename Eigen::CwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<Scalar, Real>>, DerType>::Type>(
            derived().value() * other, derived().derivatives() * other);
    }

    friend inline const AutoDiffScalar<typename Eigen::CwiseUnaryOp<Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<Real, Scalar>>, DerType>::Type>
    operator*(const Real& other, const AutoDiffScalar<DerivativeType>& a)
    {
        return AutoDiffScalar<typename Eigen::CwiseUnaryOp<Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<Real, Scalar>>, DerType>::Type>(
            a.value() * other, a.derivatives() * other);
    }

    inline AutoDiffScalar<DerivativeType>& operator*=(const Scalar& other)
    {
        *this = *this * other;
        return derived();
    }
};

template<typename DerivativeType>
struct auto_diff_special_op<DerivativeType, false>
{
    void operator*() const;

    void operator-() const;

    void operator+() const;
};

template<typename BinOp, typename A, typename B, typename RefType>
void laopt_make_coherent_expression(Eigen::CwiseBinaryOp<BinOp, A, B> xpr, const RefType& ref)
{
    laopt_make_coherent(xpr.const_cast_derived().lhs(), ref);
    laopt_make_coherent(xpr.const_cast_derived().rhs(), ref);
}

template<typename UnaryOp, typename A, typename RefType>
void laopt_make_coherent_expression(const Eigen::CwiseUnaryOp<UnaryOp, A>& xpr, const RefType& ref)
{
    laopt_make_coherent(xpr.nestedExpression().const_cast_derived(), ref);
}

// needed for compilation only
template<typename UnaryOp, typename A, typename RefType>
void laopt_make_coherent_expression(const Eigen::CwiseNullaryOp<UnaryOp, A>&, const RefType&) {}

template<typename A_Scalar, int A_Rows, int A_Cols, int A_Options, int A_MaxRows, int A_MaxCols, typename B>
struct laopt_make_coherent_impl<Eigen::Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols>, B>
{
    typedef Eigen::Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols> A;

    static void run(A& a, B& b)
    {
        if ((A_Rows == Eigen::Dynamic || A_Cols == Eigen::Dynamic) && (a.size() == 0)) {
            a.resize(b.size());
            a.setZero();
        } else if (B::SizeAtCompileTime == Eigen::Dynamic && a.size() != 0 && b.size() == 0) {
            laopt_make_coherent_expression(b, a);
        }
    }
};

template<typename A, typename B_Scalar, int B_Rows, int B_Cols, int B_Options, int B_MaxRows, int B_MaxCols>
struct laopt_make_coherent_impl<A, Eigen::Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols>>
{
    typedef Eigen::Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols> B;

    static void run(A& a, B& b)
    {
        if ((B_Rows == Eigen::Dynamic || B_Cols == Eigen::Dynamic) && (b.size() == 0)) {
            b.resize(a.size());
            b.setZero();
        } else if (A::SizeAtCompileTime == Eigen::Dynamic && b.size() != 0 && a.size() == 0) {
            laopt_make_coherent_expression(a, b);
        }
    }
};

template<typename A_Scalar, int A_Rows, int A_Cols, int A_Options, int A_MaxRows, int A_MaxCols,
    typename B_Scalar, int B_Rows, int B_Cols, int B_Options, int B_MaxRows, int B_MaxCols>
struct laopt_make_coherent_impl<Eigen::Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols>,
    Eigen::Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols>>
{
    typedef Eigen::Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols> A;
    typedef Eigen::Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols> B;

    static void run(A& a, B& b)
    {
        if ((A_Rows == Eigen::Dynamic || A_Cols == Eigen::Dynamic) && (a.size() == 0)) {
            a.resize(b.size());
            a.setZero();
        } else if ((B_Rows == Eigen::Dynamic || B_Cols == Eigen::Dynamic) && (b.size() == 0)) {
            b.resize(a.size());
            b.setZero();
        }
    }

};

} // end namespace internal

#define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC, CODE) \
  template<typename DerType> \
  inline const laopt::AutoDiffScalar< \
  EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(typename Eigen::internal::remove_all<DerType>::type, typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar, product) > \
  FUNC(const laopt::AutoDiffScalar<DerType>& x) { \
    using namespace Eigen; \
    typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
    EIGEN_UNUSED_VARIABLE(sizeof(Scalar)); \
    CODE; \
  }

template<typename DerType>
struct CleanedUpDerType
{
    typedef AutoDiffScalar<typename Eigen::internal::remove_all<DerType>::type::PlainObject> type;
};

template<typename DerType>
inline const AutoDiffScalar<DerType>& conj(const AutoDiffScalar<DerType>& x) { return x; }

template<typename DerType>
inline const AutoDiffScalar<DerType>& real(const AutoDiffScalar<DerType>& x) { return x; }

template<typename DerType>
inline typename DerType::Scalar imag(const AutoDiffScalar<DerType>&) { return 0.; }

template<typename DerType, typename T>
inline typename CleanedUpDerType<DerType>::type min(const AutoDiffScalar<DerType>& x, const T& y)
{
    typedef typename CleanedUpDerType<DerType>::type ADS;
    return (x <= y ? ADS(x) : ADS(y));
}

template<typename DerType, typename T>
inline typename CleanedUpDerType<DerType>::type max(const AutoDiffScalar<DerType>& x, const T& y)
{
    typedef typename CleanedUpDerType<DerType>::type ADS;
    return (x >= y ? ADS(x) : ADS(y));
}

template<typename DerType, typename T>
inline typename CleanedUpDerType<DerType>::type min(const T& x, const AutoDiffScalar<DerType>& y)
{
    typedef typename CleanedUpDerType<DerType>::type ADS;
    return (x < y ? ADS(x) : ADS(y));
}

template<typename DerType, typename T>
inline typename CleanedUpDerType<DerType>::type max(const T& x, const AutoDiffScalar<DerType>& y)
{
    typedef typename CleanedUpDerType<DerType>::type ADS;
    return (x > y ? ADS(x) : ADS(y));
}

template<typename DerType>
inline typename CleanedUpDerType<DerType>::type min(const AutoDiffScalar<DerType>& x, const AutoDiffScalar<DerType>& y)
{
    return (x.value() < y.value() ? x : y);
}

template<typename DerType>
inline typename CleanedUpDerType<DerType>::type max(const AutoDiffScalar<DerType>& x, const AutoDiffScalar<DerType>& y)
{
    return (x.value() >= y.value() ? x : y);
}


EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(abs,
using std::abs;
return laopt::laopt_make_autodiff_scalar(abs(x.value()), x.derivatives() * (x.value() < Scalar(0) ? Scalar(-1) : Scalar(1)));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(abs2,
using numext::abs2;
return laopt::laopt_make_autodiff_scalar(abs2(x.value()), x.derivatives() * (Scalar(2) * x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sqrt,
using std::sqrt;
Scalar sqrtx = sqrt(x.value());
return laopt::laopt_make_autodiff_scalar(sqrtx, x.derivatives() * (Scalar(0.5) / sqrtx));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cos,
using std::cos;
using std::sin;
return laopt::laopt_make_autodiff_scalar(cos(x.value()), x.derivatives() * (-sin(x.value())));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sin,
using std::sin;
using std::cos;
return laopt::laopt_make_autodiff_scalar(sin(x.value()), x.derivatives() * cos(x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(exp,
using std::exp;
Scalar expx = exp(x.value());
return laopt::laopt_make_autodiff_scalar(expx, x.derivatives() * expx);)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log,
using std::log;
return laopt::laopt_make_autodiff_scalar(log(x.value()), x.derivatives() * (Scalar(1) / x.value()));)

template<typename DerType>
inline const laopt::AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(typename Eigen::internal::remove_all<DerType>::type, typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar, product)>
pow(const laopt::AutoDiffScalar<DerType>& x, const typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar& y)
{
    using namespace Eigen;
    using std::pow;
    return laopt::laopt_make_autodiff_scalar(pow(x.value(), y), x.derivatives() * (y * pow(x.value(), y - 1)));
}


template<typename DerTypeA, typename DerTypeB>
inline const AutoDiffScalar<Eigen::Matrix<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerTypeA>::type>::Scalar, Eigen::Dynamic, 1>>
atan2(const AutoDiffScalar<DerTypeA>& a, const AutoDiffScalar<DerTypeB>& b)
{
    using std::atan2;
    typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerTypeA>::type>::Scalar Scalar;
    typedef AutoDiffScalar<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> PlainADS;
    PlainADS ret;
    ret.value() = atan2(a.value(), b.value());

    Scalar squared_hypot = a.value() * a.value() + b.value() * b.value();

    // if (squared_hypot==0) the derivation is undefined and the following results in a NaN:
    ret.derivatives() = (a.derivatives() * b.value() - a.value() * b.derivatives()) / squared_hypot;

    return ret;
}

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(tan,
using std::tan;
using std::cos;
return laopt::laopt_make_autodiff_scalar(tan(x.value()), x.derivatives() * (Scalar(1) / numext::abs2(cos(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(asin,
using std::sqrt;
using std::asin;
return laopt::laopt_make_autodiff_scalar(asin(x.value()), x.derivatives() * (Scalar(1) / sqrt(1 - numext::abs2(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(acos,
using std::sqrt;
using std::acos;
return laopt::laopt_make_autodiff_scalar(acos(x.value()), x.derivatives() * (Scalar(-1) / sqrt(1 - numext::abs2(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(tanh,
using std::cosh;
using std::tanh;
return laopt::laopt_make_autodiff_scalar(tanh(x.value()), x.derivatives() * (Scalar(1) / numext::abs2(cosh(x.value()))));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sinh,
using std::sinh;
using std::cosh;
return laopt::laopt_make_autodiff_scalar(sinh(x.value()), x.derivatives() * cosh(x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cosh,
using std::sinh;
using std::cosh;
return laopt::laopt_make_autodiff_scalar(cosh(x.value()), x.derivatives() * sinh(x.value()));)

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

} // namespace laopt

namespace Eigen
{

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<laopt::AutoDiffScalar < DerType>, typename DerType::Scalar, BinOp>
{
typedef laopt::AutoDiffScalar <DerType> ReturnType;
};

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<typename DerType::Scalar, laopt::AutoDiffScalar<DerType>, BinOp>
{
    typedef laopt::AutoDiffScalar<DerType> ReturnType;
};

// Allow 2nd derivative without casting
template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<laopt::AutoDiffScalar<DerType>, typename DerType::Scalar::Scalar, BinOp>
{
    typedef laopt::AutoDiffScalar<DerType> ReturnType;
};

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<typename DerType::Scalar::Scalar, laopt::AutoDiffScalar<DerType>, BinOp>
{
    typedef laopt::AutoDiffScalar<DerType> ReturnType;
};

// Allow 2nd derivative without casting for the TouchableDerivative type
template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<laopt::AutoDiffScalar<DerType>,typename DerType::Scalar::Scalar::Scalar,BinOp>
{
typedef laopt::AutoDiffScalar<DerType> ReturnType;
};

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<typename DerType::Scalar::Scalar::Scalar,laopt::AutoDiffScalar<DerType>, BinOp>
{
typedef laopt::AutoDiffScalar<DerType> ReturnType;
};

template<typename DerType>
struct NumTraits<laopt::AutoDiffScalar<DerType>>
    : NumTraits<typename NumTraits<typename internal::remove_all<DerType>::type::Scalar>::Real>
{
    typedef typename internal::remove_all<DerType>::type DerTypeCleaned;
    typedef laopt::AutoDiffScalar<Matrix<typename NumTraits<typename DerTypeCleaned::Scalar>::Real,
                                         DerTypeCleaned::RowsAtCompileTime, DerTypeCleaned::ColsAtCompileTime, 0,
                                         DerTypeCleaned::MaxRowsAtCompileTime, DerTypeCleaned::MaxColsAtCompileTime>> Real;
    typedef laopt::AutoDiffScalar<DerType> NonInteger;
    typedef laopt::AutoDiffScalar<DerType> Nested;
    typedef typename NumTraits<typename DerTypeCleaned::Scalar>::Literal Literal;
    enum
    {
        RequireInitialization = 1
    };
};

} // namespace Eigen

namespace std
{

template<typename T>
class numeric_limits<laopt::AutoDiffScalar < T>>

: public numeric_limits<typename T::Scalar>
{
};

template<typename T>
class numeric_limits<laopt::AutoDiffScalar<T&>

>
    : public numeric_limits<typename T::Scalar>
{
};

} // namespace std

#endif // LAOPT_AUTODIFF_SCALAR_HPP
