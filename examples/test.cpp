#include <iostream>

#define test_mult(...)

int main()
{
    std::cout << "HELLO WORLD\n";
    return 0;
}

// #define EIGEN_NO_MALLOC

// #include "lampc.hpp"


// /*[[[cog 
// from lampc import *
// nlp = NLP()

// N = nlp.const("N", 5) # Prediction horizon
// n = nlp.const("n", 2) # State dimension
// m = nlp.const("m", 1) # Input dimension

// X   = nlp.var("X",   n, N)
// U   = nlp.var("U",   m, N-1)
// xss = nlp.var("xss", n)
// uss = nlp.var("uss", m)

// ]]]*/
// constexpr int N = 5;
// constexpr int n = 2;
// constexpr int m = 1;
// //[[[end]]]


// template<typename _Scalar>
// struct my_param_t
// {
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     using Scalar = _Scalar;

//     Vec<Scalar, n + m> p2; // Some other fake parameter
//     Vec<Scalar, n> x0;
//     Vec<Scalar, 15> p; // Some other fake parameter

//     Vec<Scalar, n> q;
//     Vec<Scalar, m> r;

//     my_param_t() 
//     {
//         x0.setConstant(0);
//         p2.setConstant(1.3);
//         p.setConstant(0);

//         for(int i=0; i<n; i++) q(i) = i;
//         for(int i=0; i<m; i++) r(i) = 2*i;
//     }
// };

// // System dynamics (need a different type for x)
// template<typename T, typename param_t, typename Tx>
// inline void sys_general(param_t& param, RVec<T,n> out, RCVec<T,n> xp, RCVec<Tx,n> x, RCVec<T,m> u)
// {
//     out(0) = param.p(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + param.p(0);
//     out(1) = param.p(2) * xp(1) - cos(x(0))*u(0) + param.p(1);
// }


// // System dynamics
// //[[[cog sys = nlp.function("sys", n, ("xp", n), ("x", n), ("u", m)) ]]]
// template <typename T, typename param_t>
// inline void _sys(param_t& param, RVec<T,2> out, RCVec<T,2> xp, RCVec<T,2> x, RCVec<T,1> u)
// //[[[end]]]
// {
//     sys_general<T, param_t, T>(param, out, xp, x, u);
// }

// // System dynamics for the first state
// //[[[cog sys0 = nlp.function("sys0", n, ("xp", n), ("u", m)) ]]]
// template <typename T, typename param_t>
// inline void _sys0(param_t& param, RVec<T,2> out, RCVec<T,2> xp, RCVec<T,1> u)
// //[[[end]]]
// {
//     // x0 is a constant, the rest we want to take the derivative 
//     using Scalar = typename decltype(param.p)::Scalar;
//     sys_general<T, param_t, Scalar>(param, out, xp, param.x0, u);
// }

// // Enforce equality between two states
// //[[[cog equal = nlp.function("equal", n, ("a", n), ("b", n)) ]]]
// template <typename T, typename param_t>
// inline void _equal(param_t& param, RVec<T,2> out, RCVec<T,2> a, RCVec<T,2> b)
// //[[[end]]]
// {
//     out = a - b;
// }

// // // Objective function
// // //[[[cog #objective = nlp.scalar_function("objective", ("_X", n*N), ("_U", m*(N-1))) ]]]
// // //[[[end]]]
// // // {
// // //     const Map<const Matrix<T,n,N> > X(_X.data(),n,N);
// // //     const Map<const Matrix<T,m,N-1> > U(_U.data(),m,N-1);

// // //     static const auto Q = param.q.template cast<T>().asDiagonal();
// // //     static const auto R = param.r.template cast<T>().asDiagonal();

// // //     out = (X.array() * (Q * X).array()).sum() +
// // //           (U.array() * (R * U).array()).sum() + 
// // //           (_X.template segment<m*(N-1)>(4).array() * _U.array()).sum() - 
// // //           (_U.sum()) * (_X.sum());
// // //           ;
// // // }


// // template <typename Scalar, typename param_t>
// // struct MyConstraint
// // {

// //     Vec<Scalar, num_eq> g;
// //     Mat<Scalar, num_eq, num_vars> J;

// //     [[[cog

// //     nlp.new_constraint()

// //     i = Index(range(1, N-1))
// //     sys0(X[0], U[0]) == 0
// //     sys(X[i+1], X[i], U[i+1]) == 0

// //     sys(xss, xss, uss) == 0
// //     equal(X[N-1], xss) == 0

// //     # i = Index(range(1, N))
// //     # lb <= out_bnd(X[i], U[i]) <= "ub_func(i)"

// //     nlp.generate()

// //     cog]]]
// //     //[[[end]]]

// //     MyConstraint(param_t& _param, 
// //                  RCVec<Scalar, num_vars> x, 
// //                  RVec<Scalar, num_eq> g, 
// //                  RMat<Scalar, num_eq, num_vars> J)
// //         : param(_param), x(_x), g(_g), J(_J) 
// //         {};
// // };


// //[[[cog nlp.generate_traits() 
// struct MyTraits
// {
//     enum {
//         num_vars = 17,
//         num_eq = 0
//     };
// };
// //[[[end]]]

// using Scalar = float;
// using param_t = my_param_t<Scalar>;
// // // // make_jacobian(sys, 2, 2,2,1);


// // // template<template<typename, typename> class Constraint, // Constraint<Scalar, param_t>
// // //          template<typename, typename> class Objective>  // Objective<Scalar, param_t>
// // // struct NLP
// // // {
// // //     using Constraint::num_eq;
// // //     using Constraint::num_vars;

// // //     Vec<Scalar, num_vars> x;

// // //     Constraint c(x);

// // // }

// int main()
// {
//     cout << "Hello world" << endl;

//     // MyConstraint<float, MyTraits> con();

//     // cout << "sX = " << sX << endl << "oX(2) = " << oX(2) << endl;
//     // cout << "xss = " << sxss << endl << "oxss() = " << oxss() << endl;


//     param_t param;
//     Vec<Scalar, n> out;
//     Vec<Scalar, n> xp;
//     Vec<Scalar, n> x;
//     Vec<Scalar, m> u;

//     xp << 1,2;
//     x << 3,4;
//     u << 5;

//     _sys<Scalar, param_t>(param, out, xp, x, u);
//     _sys<Scalar, param_t>(param, out, xp, param.x0, u);

//     // sys(param, out, xp, x, u);

//     cout << "x = " << x.transpose() << " xp = " << xp.transpose() << " u = " << u.transpose() << endl;
//     cout << "out = " << out.transpose() << endl;

//     // Matrix<Scalar, n, n> J_xp;
//     // Matrix<Scalar, n, n> J_x;
//     // Matrix<Scalar, n, m> J_u;

//     // sys(param, out, xp, x, u, J_xp, J_x, J_u);

//     // cout << "J_xp = " << endl << J_xp << endl;
//     // cout << "J_x = " << endl << J_x << endl;
//     // cout << "J_u = " << endl << J_u << endl;

//     // // Final goal
//     // // - number of vars
//     // // - number of constraints
//     // // - scalar type
//     // ADMM<2, 1, Scalar> prob;

//     // // Compute hessian and linear term
//     // // Compute 'A' and upper lower bounds on A and x
//     // prob.solve(H,h,A,al,au,xl,xu);
//     // Eigen::Vector2f sol = prob.primal_solution();

//     return 0;
// }