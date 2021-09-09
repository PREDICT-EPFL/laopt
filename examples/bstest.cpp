/** Static memory implementation of a variable block-matrix type.
 */

#include "bsmatrix.hpp"
#include "lampc_function.hpp"

using namespace lampc;


template<typename Scalar, std::size_t N, typename range>
struct BSTest;

template<typename Scalar, std::size_t N, std::size_t... ind>
struct BSTest<Scalar, N, std::integer_sequence<std::size_t, ind...>>
{
    struct param_t
    {
        Eigen::Matrix<Scalar, 2, 1> x0 {-1, -2};
        const Eigen::Matrix<Scalar, 2, 2> A {{1.0, 0.0}, {0.1, 1.0}};
        Eigen::Matrix<Scalar, 2, 1> B {0.1, 0.005};
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
        Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - xplus;
    }


	template<int i>
	struct x : Variable<2> {static constexpr const char* name="x";};

	template<int i>
	struct u : Variable<1> {static constexpr const char* name="u";};

	using variables = std::tuple<u<ind>..., x<ind>..., x<N>>;

	template<int i>
	struct con_dynamics : DenseConstraint<2, x<i+1>, x<i>, u<i>> {};

	using constraints = std::tuple<con_dynamics<ind>...>;

	BSMatrix<double, BSTest::constraints, BSTest::variables> mat;

	BSTest()
	{
		mat.finalize();

		std::cout << "mat = \n" << Eigen::MatrixX<double>(mat) << std::endl;

		// Eigen::Matrix<double, con2::size, z::size> B;
		// B.array() = 1;

		// Eigen::Matrix<double, con1<1>::size, x<1>::size> B1;
		// B1.array() = 2;
		
  //   	time_point start = get_time();
		// for(int i=0; i<1000000; i++)
		// {
		// 	// B.array() += i;
		// 	mat.template set<con2, z>(B);
		// 	mat.template set<con1<1>, x<1>>(B1);
		// 	// B.array() -= i;
		// }
	 //    time_point stop = get_time();

	 //    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	 //    std::cout << "Time " << std::setprecision(9)
	 //              << static_cast<double>(duration.count()) / 1000000 << " [microseconds]" << "\n";

		// std::cout << "mat = \n" << Eigen::MatrixX<double>(mat) << std::endl;

	}
};

int main()
{
	const std::size_t N = 20;

	using Test = BSTest<double, N, std::make_integer_sequence<std::size_t, N>>;
	Test test;



	Test::param_t param;
	Eigen::Vector<double, 2> xp;
	Eigen::Vector<double, 2> x;
	Eigen::Vector<double, 1> u;

	xp << 4, 5;
	x << 1, 2;
	u << 3;

	auto ret = Test::dynamics_eq::jac(param, xp, x, u);
	std::cout << "ret.jacobian = \n" << ret.jacobian << std::endl;

	test.mat.template set<Test::con_dynamics<1>, Test::x<2>>(ret.jacobian.block(0, 0, 2, 2));
	test.mat.template set<Test::con_dynamics<1>, Test::x<1>>(ret.jacobian.block(0, 2, 2, 2));
	test.mat.template set<Test::con_dynamics<1>, Test::u<1>>(ret.jacobian.block(0, 4, 2, 1));

	std::cout << "mat = \n" << Eigen::MatrixX<double>(test.mat) << std::endl;

	// std::cout << "BS::is_member<int, int, float> = " << BS::is_member<int, int, float>() << std::endl;
	// std::cout << "BS::is_member<bool, int, float> = " << BS::is_member<bool, int, float>() << std::endl;

	// BSTest test;

	// struct x : Variable<3> {};
	// struct y : Variable<5> {};

	// Block<x> blk;
	// for(int i=0; i<x::size; i++)
	// 	blk.column_length[i] = i;
	
	// for(int i=0; i<x::size; i++)
	// 	std::cout << "blk.getColumnNonZeros<x>(" << i << ") = "
	// 			  << blk.getColumnNonZeros<x>(i) << std::endl;

	// for(int i=0; i<y::size; i++)
	// 	std::cout << "blk.getColumnNonZeros<y>(" << i << ") = "
	// 			  << blk.getColumnNonZeros<y>(i) << std::endl;

	return 0;
}