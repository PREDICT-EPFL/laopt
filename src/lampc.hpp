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

// autodiff include
// #include <autodiff/forward.hpp>
// #include <autodiff/forward/eigen.hpp>
// using namespace autodiff;

#define SEG(size, offset) template segment<size>(offset) // Segment of an Eigen vector
#define BLK(x_size, y_size, x_offset, y_offset) template block<x_size, y_size>(x_offset, y_offset) // Block of an eigen matrix
#define VEC(Scalar, size) Eigen::Matrix<Scalar, size, 1> // Eigen vector
#define MAT(Scalar, rows, cols) Eigen::Matrix<Scalar, rows, cols> // Eigen matrix
#define RVEC(Scalar, size) Eigen::Ref<Eigen::Matrix<Scalar, size, 1>> // Reference to eigen vector
// #define RMAT(Scalar, rows, cols) Eigen::Ref<Eigen::Matrix<Scalar, rows, cols, Eigen::OuterStride<> >> // Reference to eigen matrix
#define RMAT(Scalar, rows, cols) Eigen::Ref<Eigen::Matrix<Scalar, rows, cols >> // Reference to eigen matrix

#define ADMatrix(Scalar, ninputs, noutputs) Eigen::Matrix<Eigen::AutoDiffScalar<Eigen::Matrix<Scalar, ninputs, 1>>, noutputs, 1> // AutoDiff type

#define MatX Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>


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
    using gradient_f_t = Eigen::Matrix<Scalar, 1, num_vars>;

    primal_t     x;
    jacobian_t   J;
    constraint_t g;
    Scalar       f;
    gradient_f_t gradient_f;

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

        lb.setZero();
        ub.setZero();
        xlb.setZero();
        xub.setZero();
    }

    void eval()
    {

    }

    void eval_jacobian()
    {
        // Copy x to dual variables
        // x_d = x;
    }

    // Must be implemented:
    // bnds(x) - sets [lb, ub, xlb, xub]
    // g(x), f(x)
    // Jg(x), gradf(x)





};


template <typename F, typename Vec>
double timeit(const F &f, const Vec &x)
{
    const auto samples = 100;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
        f(x);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    return duration.count() / samples;
}



// Template for a function type
// We can add all sorts of static_asserts in NLP to 
// confirm that the function types are the right structure
// template<typename Scalar>
// struct Function
// {
//     enum {
//         nvars =  ***,
//         nfuncs = ***
//     };

//     Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function
//     Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function

//     void Function() {
//         J.setZero();
//         f.setZero();
//     }

//     void eval(Ref<Matrix<Scalar, nvars, 1>> x)
//     {
//         f(x) = ...
//     }

//     void eval_jacobian(Ref<Matrix<dual, nvars, 1>> x)
//     {
//         J(x) = ...
//     }
// }