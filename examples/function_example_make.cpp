/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include <sstream>
#include <fstream>


#include "lampc.hpp"
#include "la_compiler.hpp"

#include "qp_functions.hpp"



int main()
{
	using scalar_t = double;
	using Functions = MyFunctions<scalar_t>;
	using param_t = Functions::param_t;
	param_t param;

	const int N = 10;
	const int n = param.B.rows();
	const int m = param.B.cols();

	LACompiler prob("QP", std::string(type_name<param_t>()), std::string(type_name<scalar_t>()));

	// Order of creation of variables defines their order in the compiled problem
	auto x = prob.variable("x", 2, N);
	auto u = prob.variable("u", 1, N-1);
	auto xss = prob.variable("xss", 2);
	auto uss = prob.variable("uss", 1);

	std::cout << "variable x[0] = " << x[0] << std::endl;
	std::cout << "variable u = " << u << std::endl;
	std::cout << "prob.vars = " << prob.variables << std::endl;

	// Add the list of callables automatically
    auto dynamics = prob.callable(callable_info(Functions::dynamics()));
    auto dynamics_eq = prob.callable(callable_info(Functions::dynamics_eq()));
    auto dynamics_ss = prob.callable(callable_info(Functions::dynamics_ss()));
    auto stage_cost = prob.callable(callable_info(Functions::stage_cost()));
    auto terminal_cost = prob.callable(callable_info(Functions::terminal_cost()));

    std::cout << prob.callables << std::endl;

	auto equalities = prob.function("equalities");
	equalities->call(dynamics, {x[1], u[0]});
	equalities->call(dynamics_ss, {xss, uss});
	for(int i=0; i<N-1; i++)
		equalities->call(dynamics_eq, {x[i+1], x[i], u[i]});

	// equalities->add_iterator(Functions::dynamics_eq(), {it(x,i+1), it(x,i), it(u,i), x[N], u[0]});


	auto constraints = prob.function("constraints");
	constraints->equality(dynamics, {x[1], u[0]});
	constraints->equality(dynamics_ss, {xss, uss});
	for(int i=0; i<N-1; i++)
	{
		constraints->equality(dynamics_eq, {x[i+1], x[i], u[i]});
		constraints->inequality(dynamics_eq, {x[i+1], x[i], u[i]}, lb, ub);
	}


	std::cout << equalities << std::endl;

	// std::cout << "equalities.jacobianStructure = \n" << Eigen::MatrixX<int>(equalities->jacobianStructure()) << std::endl;
	// std::cout << "equalities.hessianStructure = \n" << Eigen::MatrixX<int>(equalities->hessianStructure()) << std::endl;

	// auto lagrangian = prob.weighed_sum("lagrangian");
	// lagrangian->add(objective);
	// lagrangian->add(equalities);
	// lagrangian->add(inequalities);

	auto objective = prob.weighted_sum("objective");
	for(int i=0; i<N-1; i++)
		objective->call(stage_cost, {x[i], u[i], xss, uss});
	objective->call(terminal_cost, {x[N-1], xss});

	// auto test = prob.weighted_sum("test");
	// for(int i=0; i<N-1; i++)
	// 	test->call(dynamics_eq, {x[i+1], x[i], u[i]});
	
	// auto inequalities = prob.function("inequalities");
	// for(int i=0; i<N-1; i++)
	// 	inequalities->add(Functions::dynamics(), {xss, u[i]});

	// auto objective = prob.function("objective");
	// for(int i=0; i<N-1; i++)
	// 	objective->add(Functions::stage_cost(), {x[i], u[i], xss, uss});

	// std::cout << prob;
	// std::cout << "========================\n";
	// std::cout << "========================\n";

	// objective->generate_eval(std::cout);
	// std::cout << objective->generate_jacobian();
	// auto Jobj = objective->build_jacobian();
	// std::cout << "Jobj = \n" << Jobj << std::endl;

	// std::cout << equalities->generate_eval();
	// auto J = equalities->build_jacobian();
	// std::cout << "J = " << std::endl << J << std::endl;

	// LAProblem<scalar_t, param_t> compiled(prob);

	std::ofstream f;
	f.open("/Users/cnjones/git/lampc/examples/qp.compiled.hpp", std::ios::trunc);

	f << prob.generate();
	// // std::cout << prob.generate();

	// // equalities->template generate<scalar_t>(f);
	// // equalities->template generate<scalar_t>(std::cout);

	// // compiled.toFile(f);
	f.close();

	// std::cout << indent(equalities->generate_eval());
	// std::cout << "========================\n";

	// std::cout << equalities->template generate<scalar_t>();

	// for(std::string line; std::getline(tmp, line);)
	// {
	// 	std::cout << "HELLO " << line << "\n";
	// }

 //    for (std::string line; std::getline(tmp, line); ) {
 //    	std::cout << "HELLO " << line << "\n";
 //        // sum += std::stoi(line);
 //    }
	// std::cout << tmp.rdbuf();

	return 0;
};

