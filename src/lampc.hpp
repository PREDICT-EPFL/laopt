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
template<template<typename, typename> class T, typename Scalar, typename Traits>
struct NLP< T<Scalar, Traits> >
{
    using Derived = T<Scalar, Traits>;

    enum {
        num_vars =  Traits::num_vars,
        num_eq   =  Traits::num_eq    // Number of equations
    };

    using jacobian_t = Eigen::Matrix<Scalar, num_eq, num_vars>;
    using primal_t = Eigen::Matrix<Scalar, num_vars, 1>;

    primal_t x;
    // Eigen::Matrix<dual, num_vars, 1> x_d; // Dual variables for gradients

    jacobian_t J; // Jacobian
    Eigen::Matrix<Scalar, num_eq, 1> g;
    Scalar f;
    Eigen::Matrix<Scalar, 1, num_vars> gradf; // Gradient of f

    // Upper and lower bounds
    Eigen::Matrix<Scalar, num_eq, 1> lb;
    Eigen::Matrix<Scalar, num_eq, 1> ub;
    Eigen::Matrix<Scalar, num_vars, 1> xlb;
    Eigen::Matrix<Scalar, num_vars, 1> xub;

    NLP()
    {
        x.setZero();
        // x_d.setZero();

        J.setZero();
        g.setZero();
        f = 0;
        gradf.setZero();

        lb.setZero();
        ub.setZero();
        xlb.setZero();
        xub.setZero();
    }

    void eval()
    {

    }

    void eval_jacobians()
    {
        // Copy x to dual variables
        // x_d = x;
    }

    // Must be implemented:
    // bnds(x) - sets [lb, ub, xlb, xub]
    // g(x), f(x)
    // Jg(x), gradf(x)
};

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