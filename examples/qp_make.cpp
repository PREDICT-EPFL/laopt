/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include <sstream>
#include <fstream>
#include <typeinfo>

#include "la_optimization_compiler.hpp"
#include "qp_functions.hpp"


template<typename param_t>
Eigen::VectorX<double> test(param_t param) 
{ 
	Eigen::VectorX<double> y(3);
	y[0] = 1; y[1] = 1; y[2] = 2;
	return y; 
}

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

	LAOptimizationProblem prob("QP");

	// Order of creation of variables defines their order in the compiled problem
	auto x = prob.variable("x", 2, N);
	auto u = prob.variable("u", 1, N-1);
	auto xss = prob.variable("xss", 2);
	auto uss = prob.variable("uss", 1);

	// std::cout << "prob.vars = " << prob.variables << std::endl;

	// Add the list of callables automatically
	auto dynamics_0 = prob.callable(callable_info(Functions::dynamics_0()));
    auto dynamics_eq = prob.callable(callable_info(Functions::dynamics_eq()));
    auto dynamics_ss = prob.callable(callable_info(Functions::dynamics_ss()));
    auto stage_cost = prob.callable(callable_info(Functions::stage_cost()));
    auto terminal_cost = prob.callable(callable_info(Functions::terminal_cost()));
    auto id = prob.id(); // Special callable for variable bounds

    // std::cout << ((vec(2) << 3,4).finished() >= dynamics_eq(x[2],x[1],u[0]) >= -1) << std::endl;
    // std::cout << (10 <= dynamics_eq(x[2],x[1],u[0]) <= (vec(2) << 3,4).finished()) << std::endl;
    // std::cout << (vec::Constant(2,7) <= dynamics_eq(x[2],x[1],u[0]) <= (vec(2) << 3,4).finished()) << std::endl;

    // std::cout << (1.34 == dynamics_eq(x[2],x[1],u[0])) << std::endl;
    // std::cout << (dynamics_eq(x[2],x[1],u[0]) == (vec(2) << 6.34,67).finished()) << std::endl;

    std::cout << "type(id) = " << type_name<decltype(id)>() << std::endl;
    std::cout << "type(id(x)) = " << type_name<decltype(id(x[0]))>() << std::endl;
    std::cout << "id(x) = " << id(x[0]) << std::endl;

    auto test = id(x[3]);
    std::cout << "test.callable->name = " << test.callable->name << std::endl;

    prob << (dynamics_0(x[1], u[0]) == 0) << setname("initial state");
	for(int i=0; i<N-1; i++) prob << (dynamics_eq(x[i+1], x[i], u[i]) == 3) << setname("dynamics", i);
	for(int i=0; i<N-1; i++) prob << (-12 <= id(x[i]) <= 6)                 << setname("state bound", i);
	for(int i=0; i<N-1; i++) prob << (-1 <= id(u[i]) <= 1)                  << setname("input bound", i);
	prob << (-2 <= id(x[N-1]) <= 2) << setname("terminal bound");

	std::cout << prob << std::endl;
	std::cout << "\n\n";
	std::cout << prob.generate() << std::endl;



	return 0;
};

