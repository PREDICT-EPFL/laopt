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

        Eigen::Matrix<scalar_t, 2, 1> q {1, 2}; // Stage-cost weights
        Eigen::Matrix<scalar_t, 1, 1> r {3}; // Stage-cost weights
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

    FUNCTION(dynamics_0, scalar_t, param_t, (out, 2), (x1, 2), (u0, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, p.x0.template cast<T>(), u0);
        out = tmp - x1;
    }

    FUNCTION(dynamics_ss, scalar_t, param_t, (out, 2), (x, 2), (u, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - x;
    }

    FUNCTION(stage_cost, scalar_t, param_t, (val, 1), (x, 2), (u, 1), (xss, 2), (uss, 1))
    {
        Eigen::Matrix<T, 2, 1> x_err = x - xss;
        Eigen::Matrix<T, 1, 1> u_err = u - uss;

        val(0) = x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + u_err.cwiseProduct(p.r.template cast<T>()).dot(u_err);
    }
};