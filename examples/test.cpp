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

#include "DoubleIntegrator.hpp"


/*[[[cog
from lampc import *
]]]*/
//[[[end]]]




// https://stackoverflow.com/questions/11795915/crtp-traits-class-no-type-named

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
        static_cast<Derived *>(this)->eval_jacobians_impl();
    }
};


/*[[[cog
import cog
nlp = NLP()

N = 5
U = nlp.var("u", 1, N-1)
X = nlp.var("x", 2, N)

for i in range(N-1):
    nlp.add_equality("DoubleIntegrator::dynamics", 2, (X[i+1], X[i], U[i]))
]]]*/
//[[[end]]]

#define VAR(size, offset) Base::primal_d.template segment<size>(offset)
#define VAR(name, size, offset) Base::name.template segment<size>(offset)

#define GET_MACRO(_1,_2,_3,NAME,...) NAME
#define VAR(...) GET_MACRO(__VA_ARGS__, VAR1, VAR2)(__VA_ARGS__)

template<typename Scalar, typename Traits>
struct MyNLP : public NLP< MyNLP<Scalar, Traits> >
{
    using Base = NLP< MyNLP<Scalar, Traits> >;

    public:
    enum {
        num_vars =  Traits::num_vars,
        num_eq   =  Traits::num_eq,
        num_ineq =  Traits::num_ineq
    };

    // Evaluate functions and jacobians
    void eval_impl()
    {
        /*[[[cog
        nlp.gen_eval_eq("g_eq")
        ]]]*/
        VAR(g_eq, 2, 0) = dynamics<double>(VAR(2,6), VAR(2,4), VAR(1,0)); // x2, x1, u1

        // Base::g_eq.template segment<2>(2) = DoubleIntegrator::dynamics<double>(
        //     Base::primal.template segment<2>(8),  // x2
        //     Base::primal.template segment<2>(6),  // x1
        //     Base::primal.template segment<1>(1));  // u1
        // Base::g_eq.template segment<2>(4) = DoubleIntegrator::dynamics<double>(
        //     Base::primal.template segment<2>(10),  // x3
        //     Base::primal.template segment<2>(8),  // x2
        //     Base::primal.template segment<1>(2));  // u2
        // Base::g_eq.template segment<2>(6) = DoubleIntegrator::dynamics<double>(
        //     Base::primal.template segment<2>(12),  // x4
        //     Base::primal.template segment<2>(10),  // x3
        //     Base::primal.template segment<1>(3));  // u3
        //[[[end]]]
    }

    void eval_jacobians_impl()
    {
        /*[[[cog
        nlp.gen_eval_jacobian_eq("J_eq")
        ]]]*/
        Base::J_eq.template block<2, 2>(0, 6) 
            = jacobian(dynamics<dual>, wrt(VAR(2,6)), at(VAR(2,6), VAR(2,4), VAR(1,0)); // x1, x0, u0
        Base::J_eq.template block<2, 2>(0, 4) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(4)),  // x0
            at(Base::primal_d.template segment<2>(6),   // x1
                Base::primal_d.template segment<2>(4),   // x0
                Base::primal_d.template segment<1>(0));  // u0
        Base::J_eq.template block<2, 1>(0, 0) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<1>(0)),  // u0
            at(Base::primal_d.template segment<2>(6),   // x1
                Base::primal_d.template segment<2>(4),   // x0
                Base::primal_d.template segment<1>(0));  // u0
        Base::J_eq.template block<2, 2>(2, 8) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(8)),  // x2
            at(Base::primal_d.template segment<2>(8),   // x2
                Base::primal_d.template segment<2>(6),   // x1
                Base::primal_d.template segment<1>(1));  // u1
        Base::J_eq.template block<2, 2>(2, 6) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(6)),  // x1
            at(Base::primal_d.template segment<2>(8),   // x2
                Base::primal_d.template segment<2>(6),   // x1
                Base::primal_d.template segment<1>(1));  // u1
        Base::J_eq.template block<2, 1>(2, 1) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<1>(1)),  // u1
            at(Base::primal_d.template segment<2>(8),   // x2
                Base::primal_d.template segment<2>(6),   // x1
                Base::primal_d.template segment<1>(1));  // u1
        Base::J_eq.template block<2, 2>(4, 10) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(10)),  // x3
            at(Base::primal_d.template segment<2>(10),   // x3
                Base::primal_d.template segment<2>(8),   // x2
                Base::primal_d.template segment<1>(2));  // u2
        Base::J_eq.template block<2, 2>(4, 8) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(8)),  // x2
            at(Base::primal_d.template segment<2>(10),   // x3
                Base::primal_d.template segment<2>(8),   // x2
                Base::primal_d.template segment<1>(2));  // u2
        Base::J_eq.template block<2, 1>(4, 2) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<1>(2)),  // u2
            at(Base::primal_d.template segment<2>(10),   // x3
                Base::primal_d.template segment<2>(8),   // x2
                Base::primal_d.template segment<1>(2));  // u2
        Base::J_eq.template block<2, 2>(6, 12) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(12)),  // x4
            at(Base::primal_d.template segment<2>(12),   // x4
                Base::primal_d.template segment<2>(10),   // x3
                Base::primal_d.template segment<1>(3));  // u3
        Base::J_eq.template block<2, 2>(6, 10) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<2>(10)),  // x3
            at(Base::primal_d.template segment<2>(12),   // x4
                Base::primal_d.template segment<2>(10),   // x3
                Base::primal_d.template segment<1>(3));  // u3
        Base::J_eq.template block<2, 1>(6, 3) = jacobian(
            DoubleIntegrator::dynamics<dual>,
            wrt(Base::primal_d.template segment<1>(3)),  // u3
            at(Base::primal_d.template segment<2>(12),   // x4
                Base::primal_d.template segment<2>(10),   // x3
                Base::primal_d.template segment<1>(3));  // u3
        //[[[end]]]
    }
};

struct MyTraits
{
    enum {
        /*[[[cog
        nlp.traits()
        ]]]*/
        num_vars = 14,
        num_eq   = 8,
        num_ineq = 0
        //[[[end]]]
    };
};

MyNLP<double, MyTraits> myNLP;

int main()
{
    // NLP<myNLP, > nlp;
    // nlp.eval();

    cout << "myNLP.primal = " << myNLP.primal.transpose() << endl;
    myNLP.primal.setConstant(1.2);
    cout << "myNLP.primal = " << myNLP.primal.transpose() << endl;

    cout << "myNLP.g_eq = " << myNLP.g_eq.transpose() << endl;
    myNLP.eval();
    cout << "myNLP.g_eq = " << myNLP.g_eq.transpose() << endl;
    
    return 0;
}


