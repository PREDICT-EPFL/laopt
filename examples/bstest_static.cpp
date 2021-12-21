#include <iostream>
#include <tuple>
#include <utility>

#include "bsmatrix.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"

using namespace BS;

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


template<typename Scalar, template<typename, int, typename...> class RowType>
struct BSTest
{
	// Define column tags
	struct x : Column<Scalar, 2> {};
	struct y : Column<Scalar, 3> {};
	struct z : Column<Scalar, 4> {};

	struct con1 : RowType<Scalar, 3, x, y> {};
	struct con2 : RowType<Scalar, 2, x> {};
	struct con3 : RowType<Scalar, 4, z, y> {};

	using mat_t = BSMatrix<Scalar, std::tuple<x, y, z>, std::tuple<con1, con2, con3>>;
	mat_t mat;

	void print_blocktypes()
	{
		std::cout << "type(con1, x) = " << type_name<typename mat_t::template block_type_t<con1, x>>() << std::endl;
		std::cout << "type(con1, y) = " << type_name<typename mat_t::template block_type_t<con1, y>>() << std::endl;
		std::cout << "type(con1, z) = " << type_name<typename mat_t::template block_type_t<con1, z>>() << std::endl;

		std::cout << "type(con2, x) = " << type_name<typename mat_t::template block_type_t<con2, x>>() << std::endl;
		std::cout << "type(con2, y) = " << type_name<typename mat_t::template block_type_t<con2, y>>() << std::endl;
		std::cout << "type(con2, z) = " << type_name<typename mat_t::template block_type_t<con2, z>>() << std::endl;

		std::cout << "type(con3, x) = " << type_name<typename mat_t::template block_type_t<con3, x>>() << std::endl;
		std::cout << "type(con3, y) = " << type_name<typename mat_t::template block_type_t<con3, y>>() << std::endl;
		std::cout << "type(con3, z) = " << type_name<typename mat_t::template block_type_t<con3, z>>() << std::endl;
	}

	void print_blocksparsity()
	{
		std::cout << "mat_t::is_nonzero<con1, x> = " << mat_t::template is_nonzero<con1, x> << std::endl;
		std::cout << "mat_t::is_nonzero<con1, y> = " << mat_t::template is_nonzero<con1, y> << std::endl;
		std::cout << "mat_t::is_nonzero<con1, z> = " << mat_t::template is_nonzero<con1, z> << std::endl;

		std::cout << "mat_t::is_nonzero<con2, x> = " << mat_t::template is_nonzero<con2, x> << std::endl;
		std::cout << "mat_t::is_nonzero<con2, y> = " << mat_t::template is_nonzero<con2, y> << std::endl;
		std::cout << "mat_t::is_nonzero<con2, z> = " << mat_t::template is_nonzero<con2, z> << std::endl;

		std::cout << "mat_t::is_nonzero<con3, x> = " << mat_t::template is_nonzero<con3, x> << std::endl;
		std::cout << "mat_t::is_nonzero<con3, y> = " << mat_t::template is_nonzero<con3, y> << std::endl;
		std::cout << "mat_t::is_nonzero<con3, z> = " << mat_t::template is_nonzero<con3, z> << std::endl;
	}

	void test_todense()
	{
		test_todense_impl(con1{});
	}

	void test_todense_impl(DenseRow<Scalar, 3, x, y>)
	{
		std::cout << "Dense implementation\n";
		auto& q = mat.template get<con1, x>().m_matrix;
		q.array() = 1;
		mat.template get<con1, y>().m_matrix.array() = 2;
		mat.template get<con2, x>().m_matrix.array() = 3;
		mat.template get<con3, z>().m_matrix.array() = 4;
		mat.template get<con3, y>().m_matrix.array() = 5;

		Eigen::Matrix<Scalar, mat_t::num_rows, mat_t::num_cols> D;
		mat.toDense(D);

		std::cout << "D = \n" << D << std::endl;
	}

	void test_todense_impl(SparseRow<Scalar, 3, x, y>)
	{
		std::cout << "Sparse implementation\n";
		// auto& q = mat.template get<con1, x>().m_matrix;
		// q.array() = 1;
		// mat.template get<con1, y>().m_matrix.array() = 2;
		// mat.template get<con2, x>().m_matrix.array() = 3;
		// mat.template get<con3, z>().m_matrix.array() = 4;
		// mat.template get<con3, y>().m_matrix.array() = 5;

		// Eigen::Matrix<Scalar, mat_t::num_rows, mat_t::num_cols> D;
		// mat.toDense(D);

		// std::cout << "D = \n" << D << std::endl;
	}


	void test_tosparse()
	{
		Eigen::SparseMatrix<Scalar> S(mat_t::num_rows, mat_t::num_cols);
		mat.reserve(S);
		S.makeCompressed();

		for(int i=0; i<1000000; i++)
			mat.toSparse(S);

		std::cout << "S = \n" << S << std::endl;
	}

	void run_test()
	{
		mat.info();
		std::cout << "\n\n======================\n\n" << std::endl;
		print_blocktypes();
		std::cout << "\n\n======================\n\n" << std::endl;
		print_blocksparsity();
		std::cout << "\n\n======================\n\n" << std::endl;
		test_todense();
		std::cout << "\n\n======================\n\n" << std::endl;
		test_tosparse();
		std::cout << "\n\n======================\n\n" << std::endl;

	// BS::mat_t::block_type_t<BS::con1, BS::x>& B = bs.mat.get<BS::con1, BS::x>();
	// std::cout << "B = \n" << B.m_matrix << std::endl;
	}
};

template<typename ... input_t>
using tuple_cat_t=
decltype(std::tuple_cat(
    std::declval<input_t>()...
));

template<typename Scalar, std::size_t N, typename range>
struct Prob;

template<typename Scalar, std::size_t N, std::size_t... ind>
struct Prob<Scalar, N, std::integer_sequence<std::size_t, ind...>>
{
	struct param_t
	{
		Eigen::Vector<Scalar, 2> x0;
	};

	template<std::size_t i>
	struct x : Column<Scalar, 2> {};

	template<std::size_t i>
	struct u : Column<Scalar, 1> {};

	template<std::size_t i>
	struct dynamics : DenseRow<Scalar, 2, u<i>, x<i>, x<i+1>> {};

	using variables = tuple_cat_t<std::tuple<x<ind>,  u<ind>>..., std::tuple<x<N>>>;
	using constraints = std::tuple<dynamics<ind>...>;
	using Jacobian_t = BSMatrix<Scalar, variables, constraints>;
	Jacobian_t jac;

	// Prob()
	// {
	// 	std::cout << "variables = " << type_name<variables>() << std::endl;
	// }

	void test_write_constraints()
	{
    	// Copy into the sparse matrix in block-row order
	    (void)std::initializer_list<int>{
	        (
	        	jac.template get<dynamics<ind>, u<ind>>().m_matrix.array() = ind,
	        	jac.template get<dynamics<ind>, x<ind>>().m_matrix.array() = 10*ind,
	        	jac.template get<dynamics<ind>, x<ind+1>>().m_matrix.array() = 100*ind,
	        	0
	        )...
	    };

{
		Eigen::Matrix<Scalar, Jacobian_t::num_rows, Jacobian_t::num_cols> D;

        time_point start = get_time();
        for(int i = 0; i < 10000; ++i)
        {
			jac.toDense(D);
        }
        time_point stop = get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "D = \n" << D << std::endl;
        std::cout << "Microseconds per dense conversion " << static_cast<double>(duration.count())/10000 << " [microseconds]" << "\n";
}

{
		Eigen::SparseMatrix<Scalar> S(Jacobian_t::num_rows, Jacobian_t::num_cols);
		jac.reserve(S);
		S.makeCompressed();

        time_point start = get_time();
        for(int i = 0; i < 10000; ++i)
        {
			jac.toSparse(S);
        }
        time_point stop = get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "S = \n" << S << std::endl;
        std::cout << "Microseconds per sparse conversion " << static_cast<double>(duration.count())/10000 << " [microseconds]" << "\n";
}

	}

};

int main()
{
	BSTest<double, DenseRow>().run_test();

	// BSTest<double, SparseRow>().run_test();

	const std::size_t N = 10;
	Prob<double, N, std::make_integer_sequence<std::size_t, N>> prob;
	prob.test_write_constraints();

	return 0;
}
