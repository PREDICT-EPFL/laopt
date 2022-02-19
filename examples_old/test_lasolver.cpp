// This would be generated code
#include "la_functionset.hpp"
#include "myfunctions.hpp"

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

#include "test.compiled.hpp"

int main()
{
	using scalar_t = double;
	using equalities_t = Test::equalities_t;
	equalities_t equalities;

	equalities_t::input_t x;
	equalities_t::output_t y;
	Test::param_t param;

	x.array() = 1;

	equalities.eval(param, x, y);

	equalities.eval_jacobian(param, x, y);

	// std::cout << "y = " << y.transpose() << std::endl;
	// std::cout << "jacobian = \n" << Eigen::MatrixX<scalar_t>(equalities.jacobian) << std::endl;

    const std::size_t NUM_EXP = 1000000;
    time_point start = get_time();
    for(int i = 0; i < NUM_EXP; ++i)
    {
	    equalities.eval(param, x, y);
    }
    time_point stop = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Eval time " << std::setprecision(9)
              << static_cast<scalar_t>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";


    // std::cout << "x = " << x.transpose() << std::endl;
    // std::cout << "y = " << y.transpose() << std::endl;

    start = get_time();
    for(int i = 0; i < NUM_EXP; ++i)
    {
			equalities.eval_jacobian(param, x, y);
    }
    stop = get_time();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Jacobian eval time " << std::setprecision(9)
              << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";

	// IOFormat CleanFmt(2, 0, " ", "\n", "", "");
 //    std::cout << "f.jacobian = \n" << Eigen::MatrixX<scalar_t>(equalities.jacobian).format(CleanFmt) << std::endl;

	return 0;
};