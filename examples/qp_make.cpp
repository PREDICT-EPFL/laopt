/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include <sstream>

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

	LACompiler<Functions> prob("QP");

	// Order of creation of variables defines their order in the compiled problem
	auto x = prob.variable("x", 2, N);
	auto u = prob.variable("u", 1, N-1);
	auto xss = prob.variable("xss", 2);
	auto uss = prob.variable("uss", 1);

	auto equalities = prob.function("equalities");
	equalities->add(Functions::dynamics_0(), {x[1], u[0]});
	for(int i=0; i<N-1; i++)
		equalities->add(Functions::dynamics_eq(), {x[i+1], x[i], u[i]});

	// equalities->add_iterator(Functions::dynamics_eq(), {it(x,i+1), it(x,i), it(u,i), x[N], u[0]});

	equalities->add(Functions::dynamics_ss(), {xss, uss});

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
	// std::cout << prob.generate();

	// equalities->template generate<scalar_t>(f);
	// equalities->template generate<scalar_t>(std::cout);

	// compiled.toFile(f);
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

