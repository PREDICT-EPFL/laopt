#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions
#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

namespace lampc {
template<typename _scalar_t, typename _param_t, size_t num_outputs, size_t... input_sizes>
struct FuncInfo
{
    using scalar_t = _scalar_t;
    using param_t = _param_t;
    static constexpr int num_inputs = lampc::meta::sum_template<input_sizes...>();  // Total number of inputs
};

template<template<typename Info> class DerivedT, typename Info>
struct Eval;

template<template<typename> class DerivedT, typename scalar_t, typename param_t, size_t num_outputs, size_t... input_sizes>
struct Eval<DerivedT, FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>>
{
    using Info = FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>;
    using Derived = DerivedT<Info>;

    using value_t = Eigen::Vector<scalar_t, num_outputs>;
    value_t value; // Memory to store current value

    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, num_outputs>& 
    eval( const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args) noexcept        
    {
        value = static_cast<Derived*>(this)->impl(args...);
        return value;
    }
};

template<template<typename Info> class DerivedT, typename Info>
struct Jacobian;

template<template<typename> class DerivedT, typename scalar_t, typename param_t, size_t num_outputs, size_t... input_sizes>
struct Jacobian<DerivedT, FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>>
            : public Eval<DerivedT, FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>>
{
    using Info = FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>;
    using Derived = DerivedT<Info>;

    // using eval_t = Eval<DerivedT, FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>>;
    // using value_t = typename eval_t::value_t;
    // static constexpr int num_inputs = eval_t::num_inputs;

    using jacobian_t = Eigen::Matrix<scalar_t, num_outputs, Info::num_inputs>;
    jacobian_t jacobian; // Memory to store current jacobian 

    // First order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, Info::num_inputs, 1>>;
    using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

    EIGEN_STRONG_INLINE void 
    eval_jacobian(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args) noexcept
    {
        // Convert to AD variables for the inputs and call our function
        AD_output_t out = seed_and_call(make_ad(args)...);

        // Copy Jacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            this->value(i) = out[i].value();
            jacobian.row(i) = out[i].derivatives();
        }
    }

private:

    // Sets the input derivatives to the identity. 
    // Assumes that the derivative matrix is initially zero
    template <typename vec>
    static constexpr int AD_Seed(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
            x[i].derivatives().coeffRef(i + offset) = 1;
        return offset + x.rows();
    }

    // Take a vector input and return a AD version of the vector
    template<int n>
    static EIGEN_STRONG_INLINE 
    Eigen::Matrix<AD_scalar, n, 1> 
    make_ad(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x)
    // static EIGEN_STRONG_INLINE auto make_ad(const Arg& x)
    {
        Eigen::Matrix<AD_scalar, n, 1> y;
        y = x;
        for (int i=0; i<y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    EIGEN_STRONG_INLINE Eigen::Matrix<AD_scalar, num_outputs, 1>
    seed_and_call(Eigen::Matrix<AD_scalar, input_sizes, 1>... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed(args, offset), // Set to unit vectors
                0
            )...
        };
        return static_cast<Derived*>(this)->template impl<AD_scalar>(args...);
    }
};


template<template<typename Info> class DerivedT, typename Info>
struct Hessian;

template<template<typename> class DerivedT, typename scalar_t, typename param_t, size_t num_outputs, size_t... input_sizes>
struct Hessian<DerivedT, FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>>
            : public Jacobian<DerivedT, FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>>
{
    using Info = FuncInfo<scalar_t, param_t, num_outputs, input_sizes...>;
    using Derived = DerivedT<Info>;

    // Hessian for each of the outputs in an array
    using hessian_single_t = Eigen::Matrix<scalar_t, Info::num_inputs, Info::num_inputs>;
    using hessian_t = std::array<hessian_single_t, num_outputs>;    
    hessian_t hessian;

    // Second order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, Info::num_inputs, 1>>;
    using outerDerivatives = Eigen::Matrix<AD_scalar, Info::num_inputs, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  


    EIGEN_STRONG_INLINE void
    eval_hessian(const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) noexcept
    {
        // Convert to AD variables for the inputs and call our function
        outerAD_t out = seed_and_call2(make_ad2<input_sizes>(args)...);

        // Copy Hessian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            this->value(i) = out[i].value().value();
            this->jacobian.row(i) = out[i].value().derivatives();
            for (int j = 0; j < Info::num_inputs; j++) {
                hessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
            }
        }
    }

private:

    // Take a vector input and return a AD version of the vector
    template<int n>
    static EIGEN_STRONG_INLINE Eigen::Matrix<outerADScalar, n, 1> 
        make_ad2(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x)
    {
        Eigen::Matrix<outerADScalar, n, 1> y;
        // y = x;
        for (int i=0; i<n; i++) {
            y(i).value().value() = x(i);
            y(i).value().derivatives().setZero();
            y(i).derivatives().setZero();
            for (int j = 0; j < n; j++) {
                y(i).derivatives()(j).derivatives().setZero();
            }
        }
        return y;
    }

    // Sets the input derivatives to the identity. 
    // Assumes that the derivative matrix is initially zero
    template <typename vec>
    static constexpr int AD_Seed2(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
        {
            x(i).value().derivatives().coeffRef(i + offset) = 1;
            x(i).derivatives().coeffRef(i + offset) = 1;
        }

        return offset + x.rows();
    }

    EIGEN_STRONG_INLINE outerAD_t seed_and_call2(Eigen::Matrix<outerADScalar, input_sizes, 1>... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed2(args, offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        return static_cast<Derived*>(this)->template impl<outerADScalar>(args...);
    }
};
};

#endif // __LAMPC__FUNCTION_HPP