/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include "lampc.hpp"

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

    FUNCTION(stage_cost, scalar_t, param_t, (val, 1), (x, 2), (u, 1), (xss, 2), (uss, 1))
    {
        Eigen::Matrix<T, 2, 1> x_err = x - xss;
        Eigen::Matrix<T, 1, 1> u_err = u - uss;

        val(0) = x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + u_err.cwiseProduct(p.r.template cast<T>()).dot(u_err);
    }
};

#ifdef MAKE_PROBLEM
#include "la_compiler.hpp"

int main()
{
	using scalar_t = double;
	using Functions = MyFunctions<scalar_t>;
	using param_t = Functions::param_t;
	param_t param;

	const int N = 10;
	const int n = param.B.rows();
	const int m = param.B.cols();

	LACompiler prob("LinearMPC");

	// Order of creation of variables defines their order in the compiled problem
	auto x = prob.variable("x", 2, N);
	auto u = prob.variable("u", 1, N-1);
	auto xss = prob.variable("xss", 2);
	auto uss = prob.variable("uss", 1);

	auto equalities = prob.function("equalities");
	for(int i=0; i<N-1; i++)
		equalities->add(Functions::dynamics_eq(), {x[i+1], x[i], u[i]});

	equalities->add(Functions::dynamics_ss(), {xss, uss});

	// auto inequalities = prob.function("inequalities");
	// for(int i=0; i<N-1; i++)
	// 	inequalities->add(Functions::dynamics(), {xss, u[i]});

	auto objective = prob.function("objective");
	for(int i=0; i<N-1; i++)
		objective->add(Functions::stage_cost(), {x[i], u[i], xss, uss});

	std::cout << prob;

	LAProblem<scalar_t, param_t> compiled(prob);	

	std::ofstream f;
	f.open("/Users/cnjones/git/lampc/examples/LinearMPC.compiled.hpp", std::ios::trunc);
	compiled.toFile(f);
	f.close();

	return 0;
};

#endif

#ifndef MAKE_PROBLEM
#include <iostream>
#include "LinearMPC.compiled.hpp"

int main()
{
    std::cout << "Solving with ADMM" << std::endl;

    using vars = LinearMPC::variables_info;

    // auto x0 = vars::variables();

    // vars::x(x0, 2).array() = 3;
    // vars::u(x0, 4) << {1,2};    

    // using equalities_t = LinearMPC::equalities_t;
    // equalities_t eq;
    // std::cout << eq.jacobian << std::endl;

    // using prob_t = opt::problem_t;
    // auto x0  = prob_t::var_types::make_variable_vec();
    // auto H   = prob_t::var_types::make_obj_hessian_mat();
    // auto h   = prob_t::var_types::make_obj_gradient_vec();
    // auto lb  = prob_t::var_types::make_variable_vec();
    // auto ub  = prob_t::var_types::make_variable_vec();
    // auto A   = prob_t::var_types::make_constraints_jacobian_mat();
    // auto Alb = prob_t::var_types::make_constraints_vec();
    // auto Aub = prob_t::var_types::make_constraints_vec();
	return 0;
}

#endif