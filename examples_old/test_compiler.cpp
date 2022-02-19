// #include <>
#include "la_compiler.hpp"


// The functions that will be used in defining the problem
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

	// std::cout << ""

// 	LACompiler prob("Test");
// 	auto x = prob.variable("x", 2, N);
// 	auto u = prob.variable("u", 1, N-1);
// 	auto xss = prob.variable("xss", 2);
// 	auto uss = prob.variable("uss", 1);

// auto equalities = prob.function();

// for(int i=0; i<N-1; i++)
// 	equalities.add(function<Function::dynamics>(), x(i+1), x(i), u(i));

// auto inequalities = prob.function();

// for(int i=0; i<N-1; i++)
// 	inequalities.add(function<Function::dynamics>(), x(i+1), x(i), u(i), lb, ub);

// ...

// // Method 1
// prob.compile("filename.hpp");

// ... 

// #include "filename.hpp"

// ClassName prob;
// auto var = prob.make_variable();

// x = prob.get_variable("x", var);  <- returns a segment
// x = {1, 2};



// // Method 2... later?
// prob.compile();

// x(i).set(...)
// u(i).set(...)
// out = equalities.eval()



	return 0;
};