/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <iostream>

#include "la_optimization_compiler.hpp"
#include "qp_functions.hpp"


int main()
{
	using vec = Eigen::VectorX<double>;

	using scalar_t = double;
	using Functions = MyFunctions<scalar_t>;
	using param_t = Functions::param_t;
	param_t param;

	const int N = 10;
	const int n = param.B.rows();
	const int m = param.B.cols();

	LAOptimizationProblem prob("QP", "MyFunctions<double>::param_t");

	// Order of creation of variables defines their order in the compiled problem
	auto x = prob.variable("x", 2, N);
	auto u = prob.variable("u", 1, N-1);
	auto xss = prob.variable("xss", 2);
	auto uss = prob.variable("uss", 1);

	// Add the list of callables automatically
	auto dynamics_0 = prob.callable(callable_info(Functions::dynamics_0()));
    auto dynamics_eq = prob.callable(callable_info(Functions::dynamics_eq()));
    auto dynamics_ss = prob.callable(callable_info(Functions::dynamics_ss()));
    auto output = prob.callable(callable_info(Functions::output()));
    auto stage_cost = prob.callable(callable_info(Functions::stage_cost()));
    auto terminal_cost = prob.callable(callable_info(Functions::terminal_cost()));
    // auto id = prob.id(); // Special callable for variable bounds

    prob << (dynamics_0(x[1], u[0]) == 0) << setname("initial state");
	for(int i=0; i<N-1; i++) 
		prob << (dynamics_eq(x[i+1], x[i], u[i]) == 3) << setname("dynamics", i)
			 << (-1 <= output(x[i]) <= 1) << setname("output")
			 << (-12 <= x[i])
			 << (-2*i <= u[i] <= 1);

	prob << (dynamics_ss(xss, uss) == 0) << setname("steadystate")
		 << (-1 <= uss <= 1);
	prob << (-2 <= output(x[N-1]) <= 2) << setname("terminal bound");

	for(int i=0; i<N-1; i++)
		prob.objective += stage_cost(x[i],u[i],xss,uss);
	prob.objective += terminal_cost(x[N-1], xss);

	// std::cout << prob << std::endl;
	// std::cout << "\n\n";

	std::ofstream f;
	f.open("/Users/cnjones/git/lampc/examples/qp.compiled.hpp", std::ios::trunc);
	f << prob.generate();
	f.close();

	// std::cout << prob.generate() << std::endl;

	return 0;
};

