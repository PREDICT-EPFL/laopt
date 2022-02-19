/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include "bsmatrix.hpp"
#include "bsmaker.hpp"
#include "lampc_function.hpp"

// #include <iostream>
// #include <Eigen/Eigenvalues> 
#include <type_traits>

using namespace lampc;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;
static time_point get_time()
{
    /** OS dependent */
#ifdef __APPLE__
    return std::chrono::system_clock::now();
#elif defined _WIN32 || defined _WIN64 || defined _MSC_VER
    return std::chrono::system_clock::now();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}

// #include "gtest/gtest.h"


// TEST(BSTest, FunctionMatrix) {

template<int size_>
struct Variable
{
	static constexpr int size = size_;
};

template<int size_, typename... vars>
struct Constraint
{
	static constexpr int size = size_;
	using varTuple = std::tuple<vars...>;

	// Total number of scalar variables
	static constexpr std::size_t totalVariables = BS::detail::sum_int_template<vars::size...>();

	// Number of variable blocks
	static constexpr std::size_t numVariables = sizeof...(vars);

	// Convert the variables into indices into the Variables list
	template<typename Variables>
	static constexpr Eigen::Vector<int, sizeof...(vars)> get_indices()
	{
		Eigen::Vector<int, sizeof...(vars)> indices;
		int i=0;
		auto l = {(
			indices[i++] = BS::detail::get_index<vars>(Variables()),
			0
			)...};
		return indices;
	}
};


template<typename Scalar>
struct Functions
{
    struct param_t
    {
        Eigen::Matrix<Scalar, 2, 1> x0 {-1, -2};
        const Eigen::Matrix<Scalar, 2, 2> A {{1.0, 2.0}, {3.0, 4.0}};
        Eigen::Matrix<Scalar, 2, 1> B {10, 20};
        Eigen::Matrix<Scalar, 1, 1> ref {3};

        Eigen::Matrix<Scalar, 2, 1> q {1, 1e3}; // Stage-cost weights
        Eigen::Matrix<Scalar, 1, 1> r {1e-3}; // Stage-cost weights
    };

    FUNCTION(dynamics, Scalar, param_t, (xplus, 2), (x, 2), (u, 1))
    {
        xplus = p.A.template cast<T>() * x + p.B.template cast<T>() * u;
    }

    FUNCTION(dynamics_eq, Scalar, param_t, (out, 2), (xplus, 2), (x, 2), (u, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - xplus;
    }

    FUNCTION(test_func, Scalar, param_t, (out, 2), (x, 2), (u, 1))
    {
        out << x(0) + x(1) + u(0), 
        	   2*x(0) + 3*x(1) + 4*u(0);
        // out << x(0)*x(1)+x(0)*x(0)*10.0,
        //     u(0)*x(0)*x(1)+u(0)*x(0)*x(0)*10.0;
    }
};

template<typename Scalar, std::size_t N, typename range>
struct BSTest;

template<typename Scalar, std::size_t N, std::size_t... ind>
struct BSTest<Scalar, N, std::integer_sequence<std::size_t, ind...>>
{
 	using param_t = typename Functions<Scalar>::param_t;
 	using dynamics_eq = typename Functions<Scalar>::dynamics_eq;
 	using test_func = typename Functions<Scalar>::test_func;

 	Variables var;
 	x = var.add(Variable("x", 2, N));  // <- Geneates indices
 	xss = var.add(Variable("xss", 2));


 	Constraints eq;
 	for(i=0; i<N-1; i++)
 	{
 		eq.add(function<dynamics_eq>(), x(i), x(i+1), u(i));
 		eq.add(function<steady_state>(), xss, u(i));
 	}

 	Constraints inequalities;
 	for(i=0; i<N-1; i++)
 		inequalities.add(function<dynamics_eq>(), x(i), x(i+1), u(i), lb, ub);


    template<int i>
    struct x : Variable<2> {static constexpr const char* name="x";};

    template<int i>
    struct u : Variable<1> {static constexpr const char* name="u";};

    using variables = std::tuple<x<ind>..., u<ind>..., x<N>>;

    template<int i>
    struct con_dynamics : Constraint<dynamics_eq::num_outputs, x<i+1>, x<i>, u<i>> {using Function = dynamics_eq;};
    template<int i>
    struct con_test : Constraint<test_func::num_outputs, x<i+3>, u<i+1>> {using Function = test_func;};

    using eq = std::tuple<con_test<0>, con_dynamics<ind>..., con_test<1>, con_test<7>, con_test<2>>;
    using ineq = std::tuple<con_test<3>, con_test<4>>;
    using obj = std::tuple<con_test<3>, con_dynamics<2>, con_test<4>>;

	using problemMaker = typename BS::ProblemCompiler<Scalar, param_t, variables, eq, ineq, obj>;
	using problem = typename problemMaker::problem_t;
};



int main() 
{
    using Scalar = double;
    const std::size_t N = 10;

    using Test = BSTest<Scalar, N, std::make_integer_sequence<std::size_t, N>>;
    using param_t = Test::param_t;

	param_t param;

    auto f = BS::FunctionFactory<Scalar, param_t, Test::eq, Test::variables>::make();

    decltype(f)::variable_t x;
    decltype(f)::output_t out;

    x.array() = 1;
    x.segment(4,2).array() = 2;


    const std::size_t NUM_EXP = 1000000;
    time_point start = get_time();
    for(int i = 0; i < NUM_EXP; ++i)
    {
	    f.eval(param, x, out);
    }
    time_point stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eval time " << std::setprecision(9)
              << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";


    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "out = " << out.transpose() << std::endl;

    start = get_time();
    for(int i = 0; i < NUM_EXP; ++i)
    {
	    f.jacobian(param, x, out);
    }
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Jacobian eval time " << std::setprecision(9)
              << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";

	IOFormat CleanFmt(2, 0, " ", "\n", "", "");
    std::cout << "f.jacobian = \n" << Eigen::MatrixX<Scalar>(f.get_jacobian()).format(CleanFmt) << std::endl;


    // std::cout << "type(f) = " << type_name<decltype(f)>() << std::endl;


{
	using Problem = Test::problem;
    Problem prob = Test::problemMaker::make();

    Problem::equalities_t::output_t eq;
    prob.equalities.eval(param, x, eq);

    std::cout << "eq = " << eq.transpose() << std::endl;

    std::cout << type_name<Test::x<0>>() << std::endl;

    // std::cout << "type(equalities) = " << type_name<Problem::equalities_t>() << std::endl;
    return 0;
};
}
// }