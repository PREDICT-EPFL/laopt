#ifndef LAOPT_EIGEN_CASADI_SUPPORT_HPP
#define LAOPT_EIGEN_CASADI_SUPPORT_HPP

#include <Eigen/Core>
#include "casadi/casadi.hpp"

namespace Eigen
{

template<typename Scalar, typename BinOp>
struct ScalarBinaryOpTraits<casadi::Matrix<Scalar>, double, BinOp>
{
    using ReturnType = casadi::Matrix<Scalar>;
};

template<typename Scalar, typename BinOp>
struct ScalarBinaryOpTraits<double, casadi::Matrix<Scalar>, BinOp>
{
    using ReturnType = casadi::Matrix<Scalar>;
};

template<typename Scalar>
struct NumTraits<casadi::Matrix<Scalar>>
{
    using Real = casadi::Matrix<Scalar>;
    using NonInteger = casadi::Matrix<Scalar>;
    using Literal = casadi::Matrix<Scalar>;
    using Nested = casadi::Matrix<Scalar>;

    enum {
        IsComplex             = 0,
        IsInteger             = 0,
        IsSigned              = 1,
        RequireInitialization = 1,
        ReadCost              = 1,
        AddCost               = 2,
        MulCost               = 2
    };

    static casadi::Matrix<Scalar> epsilon()
    {
        return casadi::Matrix<Scalar>(std::numeric_limits<double>::epsilon());
    }

    static casadi::Matrix<Scalar> dummy_precision()
    {
        return casadi::Matrix<Scalar>(NumTraits<double>::dummy_precision());
    }

    static casadi::Matrix<Scalar> highest()
    {
        return casadi::Matrix<Scalar>(std::numeric_limits<double>::max());
    }

    static casadi::Matrix<Scalar> lowest()
    {
        return casadi::Matrix<Scalar>(std::numeric_limits<double>::min());
    }

    static int digits10()
    {
        return std::numeric_limits<double>::digits10;
    }
};

} // namespace Eigen

namespace casadi
{
    inline bool operator||(const bool x, const Matrix<SXElem>& y)
    {
        return x || !y.is_zero();
    }
} // namespace casadi

#endif // LAOPT_EIGEN_CASADI_SUPPORT_HPP
