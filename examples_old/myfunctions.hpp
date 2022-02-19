#include "lampc_function.hpp"

template<typename scalar_t_>
struct MyFunctions
{
    using scalar_t = scalar_t_;

    struct param_t_
    {
        Eigen::Matrix<scalar_t, 2, 1> x0 {-1, -2};
        const Eigen::Matrix<scalar_t, 2, 2> A {{1.0, 2.0}, {3.0, 4.0}};
        Eigen::Matrix<scalar_t, 2, 1> B {10, 20};
        Eigen::Matrix<scalar_t, 1, 1> ref {3};

        Eigen::Matrix<scalar_t, 2, 1> q {1, 1e3}; // Stage-cost weights
        Eigen::Matrix<scalar_t, 1, 1> r {1e-3}; // Stage-cost weights
    };
    using param_t = param_t_;

    FUNCTION(dynamics, scalar_t, param_t, (xplus, 2), (x, 2), (u, 1))
    {
        xplus = p.A.template cast<T>() * x + p.B.template cast<T>() * u;
    }

    FUNCTION(dynamics_eq, scalar_t, param_t, (out, 2), (xplus, 2), (x, 2), (u, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - xplus;
    }

    FUNCTION(dynamics_ss, scalar_t, param_t, (out, 2), (x, 2), (u, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - x;
    }

    FUNCTION(test_func, scalar_t, param_t, (out, 1), (x, 2), (u, 1))
    {
        out << x(0) + x(1) + u(0);
        // , 
        	   // 2*x(0) + 3*x(1) + 4*u(0);
    }

    FUNCTION(test_func2, scalar_t, param_t, (out, 2), (xplus, 2), (x, 2), (u, 1))
    {
        out << x(0) + x(1) + u(0)*xplus(0), 
               2*x(0) + 3*x(1) + 4*u(0)*xplus(1);
    }

    FUNCTION(stage_cost, scalar_t, param_t, (val, 1), (x, 2), (u, 1), (xss, 2), (uss, 1))
    {
        Eigen::Matrix<T, 2, 1> x_err = x - xss;
        Eigen::Matrix<T, 1, 1> u_err = u - uss;

        val(0) = x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + u_err.cwiseProduct(p.r.template cast<T>()).dot(u_err);
    }

    // FUNCTION(test_func, scalar_t, param_t, (out, 2), (x, 2), (u, 1))
    // {
    //     out << x(0) + x(1) + u(0), 
    //            2*x(0) + 3*x(1) + 4*u(0);
    // }

    // using functions = std::tuple<dynamics_eq, dynamics, dynamics_ss, test_func, test_func2>;
};
