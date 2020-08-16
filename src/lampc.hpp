#include <iostream>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <functional>
#include <string>
#include <sstream>
#include <array>
#include <chrono>
#include <utility>
using namespace std;

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
using namespace Eigen;

#include "unsupported/Eigen/AutoDiff"

#define SEG(size, offset) template segment<size>(offset) // Segment of an Eigen vector
#define BLK(x_size, y_size, x_offset, y_offset) template block<x_size, y_size>(x_offset, y_offset) // Block of an eigen matrix

#define MatX Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>

template<typename Scalar, int Size>
using RCVec = const Ref<const Eigen::Matrix<Scalar, Size, 1>>;
template<typename Scalar, int Rows, int Cols>
using RCMat = const Ref<const Eigen::Matrix<Scalar, Rows, Cols>>;
template<typename Scalar, int Size>
using Vec = Eigen::Matrix<Scalar, Size, 1>;
template<typename Scalar, int Size>
using RVec = Eigen::Ref<Eigen::Matrix<Scalar, Size, 1>>;

template <typename vec>
void AD_seed(vec &x)
{
    for (int i=0; i<x.rows(); i++) {
        x[i].derivatives().coeffRef(i) = 1;
    }
}

template <typename vec>
void AD_clear(vec &x)
{
    for (int i=0; i<x.rows(); i++) {
        x[i].derivatives().coeffRef(i) = 0;
    }
}

// Computes total length of all eigen vectors passed in
// template<typename... T>
//   constexpr auto get_total_length(T... args) {
//   int acc = 0;
//   (void)std::initializer_list<int>{ (acc += T::RowsAtCompileTime, 0)... };
//   return acc;
// }

// template<typename... T>
//   constexpr auto get_total_length() {
//   int acc = 0;
//   (void)std::initializer_list<int>{ (acc += T::RowsAtCompileTime, 0)... };
//   return acc;
// }

template<typename... T>
  constexpr auto get_total_length(tuple<T&...> args) {
  int acc = 0;
  (void)std::initializer_list<int>{ (acc += T::Size, 0)... };
  return acc;
}




template<typename T> struct NLP;

/*
    Represents a problem of the form

    min f(x)
    s.t.
        xlb <= x <= xub
        lb <= g(x) <= ub

    lb, ub can be +- inf
    if lb == ub, then this is an equality
*/
template<template<typename, typename> class T, typename _Scalar, typename Traits>
struct NLP< T<_Scalar, Traits> >
{
    using Scalar = _Scalar;
    using Derived = T<Scalar, Traits>;

    enum {
        num_vars =  Traits::num_vars,
        num_eq   =  Traits::num_eq    // Number of equations
    };

    using jacobian_t = Eigen::Matrix<Scalar, num_eq, num_vars>;
    using primal_t = Eigen::Matrix<Scalar, num_vars, 1>;
    using constraint_t = Eigen::Matrix<Scalar, num_eq, 1>;
    using gradient_f_t = Eigen::Matrix<Scalar, num_vars, 1>;
    using hessian_f_t = Eigen::Matrix<Scalar, num_vars, num_vars>;

    primal_t     x;
    jacobian_t   J;
    constraint_t g;
    Scalar       f;
    gradient_f_t gradient_f;
    hessian_f_t  hessian_f;

    // Expressions to access the variables and constraints
    // constexpr auto var(pair<const int, int> p) {return x.template segment<p.first>(p.second);};
    // constexpr auto con(pair<const int, int> p) {return g.template segment<p.first>(p.second);};
    // constexpr auto Jac(pair<const int, int> con, pair<int, int> var)
    //     {return J.template block<con.first, var.first>(con.second, var.second);};

    // Upper and lower bounds
    Eigen::Matrix<Scalar, num_eq, 1> lb;
    Eigen::Matrix<Scalar, num_eq, 1> ub;
    Eigen::Matrix<Scalar, num_vars, 1> xlb;
    Eigen::Matrix<Scalar, num_vars, 1> xub;

    NLP()
    {
        x.setZero();

        J.setZero();
        g.setZero();
        f = 0;
        gradient_f.setZero();
        hessian_f.setZero();

        lb.setZero();
        ub.setZero();
        xlb.setZero();
        xub.setZero();
    }

    // Must be implemented:
    // bnds(x) - sets [lb, ub, xlb, xub]
    // g(x), f(x)
    // Jg(x), gradf(x)
};


template<class Derived, typename Scalar, typename param_t, int NumOutputs, int... n>
struct JJacobian
{
    static constexpr int input_len()
    {
        int NumInputs = 0;
        (void)initializer_list<int>{ (NumInputs += n, 0)... };
        return NumInputs;
    };
    static constexpr int NumInputs = input_len();         // Total number of inputs
    static constexpr auto input_sizes = make_tuple(n...); // Size of each input

    // AD dual variable
    using contiguous_input_t = Matrix<Scalar, NumInputs, 1>;
    using ADScalar = AutoDiffScalar<contiguous_input_t>;

    // AD variables for block-inputs and output
    using AD_input_t = tuple<Matrix<ADScalar, n, 1>...>;
    using AD_output_t = Matrix<ADScalar, NumOutputs, 1>;

    // Tuples of input and jacobian blocks
    using TplInputs = tuple<const Ref<const Matrix<Scalar, n, 1>>...>;
    using TplJacobians = tuple<Ref<Matrix<Scalar, NumOutputs, n>>...>;

    AD_input_t _x;    // Tuple of dual variables as inputs
    AD_output_t _out; // Vector of dual variables as outputs


    // Evaluation operator
    inline void operator()(param_t& param,
                    Ref<Matrix<Scalar,NumOutputs,1>> out, 
                    const Ref<const Matrix<Scalar, n, 1>>... inputs
                  )
    {
        auto derived = static_cast< Derived* >(this);
        derived->eval(param, out, std::forward<decltype(inputs)>(inputs)...);
    }

    // Jacobian operator
    inline void operator()(param_t& param,
                    Ref<Matrix<Scalar,NumOutputs,1>> out, 
                    const Ref<const Matrix<Scalar, n, 1>>... inputs,
                    Ref<Matrix<Scalar, NumOutputs, n>>... J
                  )
    {
        jacobian_impl(forward<decltype(out)>(out), 
                      forward_as_tuple(inputs...), 
                      forward_as_tuple(J...),
                      param,
                      std::make_index_sequence<sizeof...(n)>{});
    }

    template <typename vec>
    constexpr int AD_Seed(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++) {
            x[i].derivatives().coeffRef(i + offset) = 1;
        }
        return offset + x.rows();
    }

    template<std::size_t... Is>
    void jacobian_impl(Ref<Matrix<Scalar,NumOutputs,1>> val, 
                  TplInputs inputs,
                  TplJacobians J,
                  param_t& param,
                  std::index_sequence<Is...>)
    {
        // Copy the current value of the inputs into the AD variable
        // and set derivative equal to identity
        int offset = 0;
        (void)initializer_list<int>{ 
            (
                get<Is>(_x) = get<Is>(inputs),         // Copy inputs
                offset = AD_Seed(get<Is>(_x), offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        auto derived = static_cast< Derived* >(this);
        derived->template eval<ADScalar>(param, _out, std::get<Is>(_x)...);
        // std::forward<decltype(inputs)>(inputs)...);
        // f(std::get<Is>(_x)..., _out, param);

        // Copy Jacobian into output variables
        for(int i=0; i<NumOutputs; i++)
        {
            val(i) = _out[i].value();
            Ref<contiguous_input_t> deriv = _out[i].derivatives();
            
            // Copy gradients to Jacobian matrices
            offset = 0;
            (void)initializer_list<int>{ 
                (
                    get<Is>(J).row(i) = deriv.template segment<get<Is>(input_sizes)>(offset), 
                    offset += get<Is>(input_sizes),
                    0
                )... 
            };
        }
    }
};


// template <typename Scalar, typename param_t> 
            // struct name##_diff : public JJacobian< name##_diff<Scalar, param_t>, 

#define make_jacobian(name, NumOutputs, NumInputs...) \
    struct name##_diff : public JJacobian< name##_diff, \
                                Scalar, param_t, NumOutputs, NumInputs> \
{ \
    template<typename T, typename... Inputs> \
    constexpr void eval(param_t& param, RVec<T,NumOutputs> out, \
                     Inputs&&... inputs) \
    { \
        _##name<T,param_t>(param, out, inputs...); \
    } \
} name;



template<class Derived, typename Scalar, typename param_t, int... n>
struct HHessian
{
    static constexpr int input_len()
    {
        int NumInputs = 0;
        (void)initializer_list<int>{ (NumInputs += n, 0)... };
        return NumInputs;
    };
    static constexpr int NumInputs = input_len();         // Total number of inputs
    static constexpr auto input_sizes = make_tuple(n...); // Size of each input

    // AD dual variable
    using Derivatives = Matrix<Scalar, NumInputs, 1>;
    using ADScalar = AutoDiffScalar<Derivatives>;
    using outerDerivatives = Eigen::Matrix<ADScalar, NumInputs, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;

    using Hessian_t = Eigen::Matrix<Scalar, NumInputs, NumInputs>;
    using AD_input_t = tuple<Matrix<outerADScalar, n, 1>...>; 

    // Tuples of input and gradient blocks
    using TplInputs = tuple<const Ref<const Matrix<Scalar, n, 1>>...>;
    using TplGradients = tuple<Ref<Matrix<Scalar, n, 1>>...>;

    AD_input_t _x;      // Tuple of AD input variables
    outerADScalar _out; // Dual variable as output

    // Evaluation operator
    inline void operator()(param_t& param,
                    Scalar& out, 
                    const Ref<const Matrix<Scalar, n, 1>>... inputs
                  )
    {
        auto derived = static_cast< Derived* >(this);
        derived->eval(param, out, std::forward<decltype(inputs)>(inputs)...);
    }

    // Gradient operator (Note: Also computes the Hessian, but doesn't return it - slow!)
    // inline void operator()(param_t& param,
    //                 Scalar& out, 
    //                 const Ref<const Matrix<Scalar, n, 1>>... inputs,
    //                 Ref<Matrix<Scalar, n, 1>>... g
    //               )
    // {
    //     jacobian_impl(forward<decltype(out)>(out), 
    //                   forward_as_tuple(inputs...), 
    //                   forward_as_tuple(J...),
    //                   param,
    //                   std::make_index_sequence<sizeof...(n)>{});
    // }

    template<typename... H_t>
    inline void operator()(param_t& param,
                    Scalar& out, 
                    const Ref<const Matrix<Scalar, n, 1>>... inputs,
                    Ref<Matrix<Scalar, n, 1>>... g,
                    H_t... H // tuple of rows
                  )
    {
        hessian_impl(param,
                     forward<decltype(out)>(out), 
                     forward_as_tuple(inputs...), 
                     forward_as_tuple(g...),
                     forward_as_tuple(H...),
                     std::make_index_sequence<sizeof...(n)>{});
    }


    template <typename vec>
    constexpr int AD_Seed(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++) {
            x(i).value().derivatives() = Derivatives::Unit(NumInputs, i + offset);
            x(i).derivatives() = Derivatives::Unit(NumInputs, i + offset);
            
            // Initialize hessian matrix to zero 
            for (int j = 0; j < NumInputs; j++) {
                x(i).derivatives()(j).derivatives().setZero();
            }
        }
        return offset + x.rows();
    }

    // Copy the vector x to the AD vector AD_x
    template <typename ADvec, typename vec>
    constexpr void copy_input(ADvec& AD_x, vec& x)
    {
        for(int i=0; i<x.rows(); i++) 
            AD_x(i).value().value() = x(i);
    }

    // template<std::size_t... Is, typename... H_t, int m>
    // constexpr int copy_hessian_block(tuple<Ref<Matrix<Scalar,m,n...>>> H, //H_t&...> H, // Row of hessian blocks

    template<std::size_t... Is, typename... H_t>
    constexpr int copy_hessian_block(tuple<H_t&...> H, //H_t&...> H, // Row of hessian blocks
                      int offset,                      // Offset of the input
                      std::index_sequence<Is...>)
    {
        int H_offset = 0;        
        for(int i=0; i<get<0>(H).rows(); i++)
        {
            Ref<Derivatives> hess_row = _out.derivatives()(i + offset).derivatives(); // The i'th row of the hessian

            H_offset = 0;
            (void)initializer_list<int>{ 
                (
                    get<Is>(H).template middleRows<1>(i) = hess_row.template segment<get<Is>(input_sizes)>(H_offset), 
                    H_offset += get<Is>(input_sizes),
                    0
                )... 
            };

        }
        return get<0>(H).rows();
    }

    template<std::size_t... Is, typename... H_t>
    void hessian_impl(param_t& param,
                      Scalar& out, 
                      TplInputs inputs,
                      TplGradients g,
                      tuple<H_t&...> H,
                      std::index_sequence<Is...>)
    {
        // Copy the current value of the inputs into the AD variable
        // and set derivative equal to identity
        int offset = 0;
        (void)initializer_list<int>{ 
            (
                copy_input(get<Is>(_x), get<Is>(inputs)),
                offset = AD_Seed(get<Is>(_x), offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        auto derived = static_cast< Derived* >(this);
        derived->template eval<outerADScalar>(param, _out, std::get<Is>(_x)...);

        // Function value
        out = _out.value().value();

        // Copy gradient into output variables
        Ref<Derivatives> grad = _out.value().derivatives();
        offset = 0;
        (void)initializer_list<int>{ 
            (
                get<Is>(g) = grad.template segment<get<Is>(input_sizes)>(offset), 
                offset += get<Is>(input_sizes),
                0
            )... 
        };

        // Ref<Derivatives> hess_row;
        offset = 0;
        (void)initializer_list<int>{
            (
                offset += copy_hessian_block(get<Is>(H), offset, std::make_index_sequence<sizeof...(n)>{}),
                0
            )...
        };

    }
};


#define make_hessian(name, NumInputs...) \
    struct name##_hess : public HHessian< name##_hess, Scalar, param_t, NumInputs> \
{ \
    template<typename T, typename... Inputs> \
    constexpr void eval(param_t& param, T& out, Inputs&&... inputs) \
    { \
        _##name<T,param_t>(param, out, inputs...); \
    } \
} name;
