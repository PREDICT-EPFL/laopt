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
#include <Eigen/StdVector>
using namespace Eigen;

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;

template<typename T> struct NLP;

template<template<typename, typename> class T, typename Scalar, typename Traits>
struct NLP< T<Scalar, Traits> >
{
    public:
    using Derived = T<Scalar, Traits>;

    enum {
        num_vars =  Traits::num_vars,
        num_eq   =  Traits::num_eq,
        num_ineq =  Traits::num_ineq
    };

    Eigen::Matrix<Scalar, num_vars, 1> primal;
    Eigen::Matrix<dual, num_vars, 1> primal_d; // Dual variables for gradients

    Eigen::Matrix<Scalar, num_eq, num_vars> J_eq;
    Eigen::Matrix<Scalar, num_eq, 1> g_eq;

    Eigen::Matrix<Scalar, num_ineq, num_vars> J_ineq;
    Eigen::Matrix<Scalar, num_ineq, 1> g_ineq;

    int test = 4;

    void eval()
    {
        static_cast<Derived *>(this)->eval_impl();
    }

    void eval_jacobians()
    {
        // Copy primal to dual variables
        primal_d = primal;
        static_cast<Derived *>(this)->eval_jacobians_impl();
    }
};
