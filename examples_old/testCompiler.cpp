#include <cstdio>
#include <iostream>
#include <fstream>
#include "la_compiler.hpp"

// User functions that will be used in defining the problem
#include "myfunctions.hpp"

int main()
{
	using scalar_t = double;
	using Functions = MyFunctions<scalar_t>;
	using param_t = Functions::param_t;
	param_t param;

	const int N = 10;
	const int n = param.B.rows();
	const int m = param.B.cols();

	LACompiler prob("Test");

	// Order of creation of variables defines their order in the compiled problem
	auto x = prob.variable("x", 2, N);
	auto u = prob.variable("u", 1, N-1);
	auto xss = prob.variable("xss", 2);
	auto uss = prob.variable("uss", 1);

	auto equalities = prob.function("equalities");
	for(int i=0; i<N-1; i++)
		equalities->add(Functions::dynamics_eq(), {x[i+1], x[i], u[i]});

	equalities->add(Functions::dynamics_ss(), {x[N-2], u[N-2]});
	equalities->add(Functions::test_func(), {x[3], uss});
	equalities->add(Functions::dynamics_ss(), {xss, uss});

	auto inequalities = prob.function("inequalities");
	for(int i=0; i<N-1; i++)
		inequalities->add(Functions::dynamics(), {xss, u[i]});

	auto objective = prob.function("objective");
	for(int i=0; i<N-1; i++)
		objective->add(Functions::stage_cost(), {x[i], u[i], xss, uss});

	std::cout << prob;

	LAProblem<scalar_t, param_t> compiled(prob);	

	std::ofstream f;
	f.open("/Users/cnjones/git/lampc/examples/test.compiled.hpp", std::ios::trunc);
	compiled.toFile(f);
	f.close();

	return 0;
};