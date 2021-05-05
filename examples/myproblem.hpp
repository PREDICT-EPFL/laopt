#include <math.h>
#include <map>
#include "Eigen/Dense"
#include <Eigen/Sparse>
#include "unsupported/Eigen/AutoDiff"

#include "polygen_helper.hpp"

using namespace Eigen;

template<int nOutputs_, int... inSizes_>
struct FunctionTraits
{
	static constexpr int nOutputs = nOutputs_;
	static constexpr int nInputs = sizeof...(inSizes_);
	// static constexpr std::array<int, nInputs> inSizes = {inSizes_...};
};

template<typename>
struct xcon_t;

template<int nOutputs_, int... inSizes>
struct xcon_t< FunctionTraits<nOutputs_, inSizes...> >
{
	using func_t = FunctionTraits<nOutputs_, inSizes...>;
	static constexpr int nOutputs = func_t::nOutputs;
	static constexpr int nInputs = func_t::nInputs;

	xcon_t()
	{
		std::cout << "nOutputs = " << nOutputs << std::endl;
        (void)std::initializer_list<int>{ 
            (
				std::cout << "input size = " << inSizes << std::endl,
				0
			)...
		};
	}
    // static constexpr std::size_t len      = len_;      // Length of a constraint of this type
    // static constexpr std::size_t num_constraints = num_constraints_; // Number of constraints in this set

    // static constexpr std::size_t nnz = nnz_; // Number of non-zeros in a single constraint of this type

    // // // Offsets into the constraint jacobian for each input
    // // Eigen::Matrix<Eigen::Index, num_constraints, num_inputs> jacobian_offsets;

    // static constexpr std::size_t offset = Prev::next;               // Offset into the main constraint
    // static constexpr std::size_t next = offset + len * num_constraints;  // Offset where next constraint should go
    // static constexpr std::size_t constraint_index = Prev::constraint_index + Prev::num_constraints; // Variable index

    // // Shorthand to return offset
    // EIGEN_STRONG_INLINE constexpr std::size_t operator()(int ind = 0) const
    // {
    //     assert(ind < num);
    //     return offset + ind * len;
    // }
};


template<typename scalar_t>
struct LOpt
{
	// Define variable accessors and ordering
	using xss_t = var_t<2, 1>;
	using uss_t = var_t<1, 1, xss_t>;
	using x_t = var_t<2, 5, uss_t>;
	using u_t = var_t<1, 4, x_t>;
	
	static constexpr auto xss = xss_t();
	static constexpr auto uss = uss_t();
	static constexpr auto x = x_t();
	static constexpr auto u = u_t();

	// Declare the sizes of all functons
	using func1_t = FunctionTraits<2, 1, 2>;
	using func3_t = FunctionTraits<2, 1, 2, 2>;
	using func2_t = FunctionTraits<2, 1, 2>;
	using dynamics_t = FunctionTraits<2, 2, 1>;
	
	// Define constraint accessors and ordering
	// using eq1_t = con_t<func1_t, 1>;
	// using eq2_t = con_t<func2_t, 1, eq1_t>;
	// using eq3_t = con_t<func3_t, 1, eq2_t>;
	// using eq4_t = con_t<func3_t, 1, 10, eq3_t>;
	// using eq5_t = con_t<func3_t, 1, 10, eq4_t>;

	using eq1_t = con_t<2, 1, 6>;
	using eq2_t = con_t<2, 1, 6, eq1_t>;
	using eq3_t = con_t<2, 1, 10, eq2_t>;
	using eq4_t = con_t<2, 1, 10, eq3_t>;
	using eq5_t = con_t<2, 1, 10, eq4_t>;

	eq1_t eq1;
	eq2_t eq2;
	eq3_t eq3;
	eq4_t eq4;
	eq5_t eq5;

	xcon_t<func1_t> xeq1;

	// static constexpr auto eq1 = eq1_t();
	// static constexpr auto eq2 = eq2_t();
	// static constexpr auto eq3 = eq3_t();
	// static constexpr auto eq4 = eq4_t();
	// static constexpr auto eq5 = eq5_t();
	
	// Convenient user-accessors
	// std::array<con_slow_t, 5> constraint_list = {con_slow_t{eq1, "eq1"}, {eq2, "eq2"}, {eq3, "eq3"}, {eq4, "eq4"}, {eq5, "eq5"}};
	
	// Define NLP sizes
	enum
	{
		NUM_VARS = u_t::next,
		NUM_CON   = eq5_t::next,
		nnz_constraints_jacobian = eq1_t::nnz + eq2_t::nnz + eq3_t::nnz + eq4_t::nnz + eq5_t::nnz
	};
	
	// Short names to pass constant vectors and writable vectors
	template<typename T, std::size_t n>
	using cVec = const Eigen::Ref<const Eigen::Matrix<T, n, 1>>;
	template<typename T, std::size_t n>
	using Vec = Eigen::Ref<Eigen::Matrix<T, n, 1>>;
	
	// NLP variable types
	using variable_t            = Matrix<scalar_t, NUM_VARS, 1>;
	using constraint_t          = Matrix<scalar_t, NUM_CON, 1>;
	using constraint_jacobian_t = Matrix<scalar_t, NUM_CON,  NUM_VARS>;
	using obj_gradient_t        = Matrix<scalar_t, 1, NUM_VARS>;
	using obj_hessian_t         = Matrix<scalar_t, NUM_VARS, NUM_VARS>;
	using obj_t                 = scalar_t;
	
	// Set non-zeros of J to match the sparsity structure of the constraint Jacobian
	EIGEN_STRONG_INLINE void constraints_sparse_initialize(SparseMatrix<scalar_t>& J)
	{
		std::vector<Eigen::Triplet<scalar_t>> trip;
		
		set_equation_sparsity(trip, eq1.info(), {u.info(0), x.info(1)});
		set_equation_sparsity(trip, eq2.info(), {uss.info(), xss.info()});
		set_equation_sparsity(trip, eq3.info(), {u.info(1), x.info(1), x.info(2)});
		set_equation_sparsity(trip, eq4.info(), {u.info(2), x.info(2), x.info(3)});
		set_equation_sparsity(trip, eq5.info(), {u.info(3), x.info(3), x.info(4)});
		
		J.setFromTriplets(trip.begin(), trip.end());
		J.makeCompressed();
	}
	// Forwarding calls for dense and sparse jacobians. Will be moved to parent class later.
	EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, Ref<constraint_t> constraints) noexcept
		{this->template constraints_impl<int>(var, constraints, 0);}
	EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, Ref<constraint_t> constraints, Ref<constraint_jacobian_t> jacobian) noexcept
		{this->template constraints_impl<Ref<constraint_jacobian_t>>(var, constraints, jacobian);}
	EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, Ref<constraint_t> constraints, Ref<SparseMatrix<scalar_t>> jacobian) noexcept
		{this->template constraints_impl<Ref<SparseMatrix<scalar_t>>>(var, constraints, jacobian);}
	
	template<typename jacobian_t>
	EIGEN_STRONG_INLINE void constraints_impl(const Ref<const variable_t>& var,
	                                          Ref<constraint_t> constraints,
	                                          jacobian_t jacobian) noexcept
	{

		// eq1(con) = func1(eq1.J<0>(J))

		// eq1(con) = func1(u(v_, 0), x(v_, 1), eq1(J, 0), eq1(J, 1));

		// eq1(func1(u(0, var), x(1, var)), c, J);
		// eq2(func2(uss(var), xss(var)), c, J);
		// for (int i=1, ind=0; i<N; i+=1, ind++)
		// 	eq3(func3(u(i), x(i), x(i+1)), c, J, ind);

		func1({u(0), x(1)}, {0, 0}, eq1(), var, constraints, jacobian);
		func2({uss(), xss()}, {0, 0}, eq2(), var, constraints, jacobian);
		func3({u(1), x(1), x(2)}, {0, 2, 0}, eq3(), var, constraints, jacobian);
		func3({u(2), x(2), x(3)}, {0, 2, 0}, eq4(), var, constraints, jacobian);
		func3({u(3), x(3), x(4)}, {0, 2, 0}, eq5(), var, constraints, jacobian);
	}
	
	// Evaluates the upper and lower bounds for the optimization variable into x_l and x_u
	// Can access the resulting bounds with the macros var_get(x_l), where var is the variable
	EIGEN_STRONG_INLINE void variable_bounds(Ref<variable_t> x_l, 
	                                         Ref<variable_t> x_u) noexcept
	{
		xss.get(x_l).array() = -2e+20;
		xss.get(x_u).array() = 2e+20;
		uss.get(x_l).array() = -2e+20;
		uss.get(x_u).array() = 2e+20;
		x.get_matrix(x_l).colwise() = Const14;
		x.get_matrix(x_u).array() = 2e+20;
		u.get_matrix(x_l).array() = -2e+20;
		u.get_matrix(x_u).array() = 2e+20;
	}
	
	// Evaluates the upper and lower bounds for the constraints variable into g_l and g_u
	// Can access the resulting bounds with the macros con_get(g_l), where con is the variable
	EIGEN_STRONG_INLINE void constraint_bounds(Ref<constraint_t> g_l, 
	                                           Ref<constraint_t> g_u) noexcept
	{
		eq1.get(g_l).array() = 0.0;
		eq1.get(g_u).array() = 0.0;
		eq2.get(g_l).array() = 0.0;
		eq2.get(g_u).array() = 0.0;
		eq3.get(g_l).array() = 0.0;
		eq3.get(g_u).array() = 0.0;
		eq4.get(g_l).array() = 0.0;
		eq4.get(g_u).array() = 0.0;
		eq5.get(g_l).array() = 0.0;
		eq5.get(g_u).array() = 0.0;
	}
	
	EIGEN_STRONG_INLINE void objective_sparse_initialize(SparseMatrix<scalar_t>& H)
	{
		set_nonzero_blocks<scalar_t>(H, {BlockInfo
		{0, 0, 2, 2}, {0, 2, 2, 1}, {0, 3, 2, 2}, {0, 5, 2, 2}, {0, 7, 2, 2}, {0, 9, 2, 2},
		{0, 13, 2, 1}, {0, 14, 2, 1}, {0, 15, 2, 1}, {0, 16, 2, 1}, {2, 0, 1, 2}, {2, 2, 1, 1},
		{2, 3, 1, 2}, {2, 5, 1, 2}, {2, 7, 1, 2}, {2, 9, 1, 2}, {2, 13, 1, 1}, {2, 14, 1, 1},
		{2, 15, 1, 1}, {2, 16, 1, 1}, {3, 0, 2, 2}, {3, 2, 2, 1}, {3, 3, 2, 2}, {3, 13, 2, 1},
		{5, 0, 2, 2}, {5, 2, 2, 1}, {5, 5, 2, 2}, {5, 14, 2, 1}, {7, 0, 2, 2}, {7, 2, 2, 1},
		{7, 7, 2, 2}, {7, 15, 2, 1}, {9, 0, 2, 2}, {9, 2, 2, 1}, {9, 9, 2, 2}, {9, 16, 2, 1},
		{13, 0, 1, 2}, {13, 2, 1, 1}, {13, 3, 1, 2}, {13, 13, 1, 1}, {14, 0, 1, 2}, {14, 2, 1, 1},
		{14, 5, 1, 2}, {14, 14, 1, 1}, {15, 0, 1, 2}, {15, 2, 1, 1}, {15, 7, 1, 2}, {15, 15, 1, 1},
		{16, 0, 1, 2}, {16, 2, 1, 1}, {16, 9, 1, 2}, {16, 16, 1, 1}
		});
	}
	
	// Locations to store the computed hessians
	const Matrix<Eigen::Index, 4, 4> H_offset_0 = {{98, 41, 50, 11}, {95, 32, 47, 2}, {96, 33, 48, 3}, {93, 30, 45, 0}};
	const Matrix<Eigen::Index, 4, 4> H_offset_1 = {{104, 42, 62, 12}, {101, 32, 59, 2}, {102, 35, 60, 5}, {99, 30, 57, 0}};
	const Matrix<Eigen::Index, 4, 4> H_offset_2 = {{110, 43, 74, 13}, {107, 32, 71, 2}, {108, 37, 72, 7}, {105, 30, 69, 0}};
	const Matrix<Eigen::Index, 4, 4> H_offset_3 = {{116, 44, 86, 14}, {113, 32, 83, 2}, {114, 39, 84, 9}, {111, 30, 81, 0}};
	
	EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var) noexcept
	{
		scalar_t val = 0.0;
		
		val += func7({u(0), uss(), x(0), xss()}, var);
		val += func7({u(1), uss(), x(1), xss()}, var);
		val += func7({u(2), uss(), x(2), xss()}, var);
		val += func7({u(3), uss(), x(3), xss()}, var);
		return val;
	}
	
	EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var,
	                                       Ref<obj_gradient_t> gradient) noexcept
	{
		scalar_t val = 0.0;
		gradient.setZero();
		
		val += func7({u(0), uss(), x(0), xss()}, var, gradient, true);
		val += func7({u(1), uss(), x(1), xss()}, var, gradient, true);
		val += func7({u(2), uss(), x(2), xss()}, var, gradient, true);
		val += func7({u(3), uss(), x(3), xss()}, var, gradient, true);
		return val;
	}
	
	EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var,
	                                       Ref<obj_gradient_t> gradient,
	                                       SparseMatrix<scalar_t>& hessian) noexcept
	{
		scalar_t val = 0.0;
		gradient.setZero();
		// Set non-zero elements of hessian to zero, but don't change nnz
		Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>> (hessian.valuePtr(), hessian.nonZeros(), 1).array() = 0.0;
		
		val += func7({u(0), uss(), x(0), xss()}, var, gradient, H_offset_0, hessian, true);
		val += func7({u(1), uss(), x(1), xss()}, var, gradient, H_offset_1, hessian, true);
		val += func7({u(2), uss(), x(2), xss()}, var, gradient, H_offset_2, hessian, true);
		val += func7({u(3), uss(), x(3), xss()}, var, gradient, H_offset_3, hessian, true);
		return val;
	}
	
	template<typename T>
	EIGEN_STRONG_INLINE void _func1(cVec<T, 1>& var0_, cVec<T, 2>& var1_, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 2, 1> tmp1;
		_dynamics<T>(x0.template cast<T>(), var0_, tmp1);
		out = var1_ - tmp1;
	}
	MAKE_JACOBIAN(func1, LOpt<scalar_t>, 2, 1, 2);
	
	template<typename T>
	EIGEN_STRONG_INLINE void _func3(cVec<T, 1>& var0_, cVec<T, 2>& var1_, cVec<T, 2>& var2_, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 2, 1> tmp2;
		_dynamics<T>(var1_, var0_, tmp2);
		out = var2_ - tmp2;
	}
	MAKE_JACOBIAN(func3, LOpt<scalar_t>, 2, 1, 2, 2);
	
	template<typename T>
	EIGEN_STRONG_INLINE void _func2(cVec<T, 1>& var0_, cVec<T, 2>& var1_, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 2, 1> tmp3;
		_dynamics<T>(var1_, var0_, tmp3);
		out = var1_ - tmp3;
	}
	MAKE_JACOBIAN(func2, LOpt<scalar_t>, 2, 1, 2);
	
	template<typename T>
	EIGEN_STRONG_INLINE void _func7(cVec<T, 1>& var0_, cVec<T, 1>& var1_, cVec<T, 2>& var2_, cVec<T, 2>& var3_, T& out) const noexcept
	{
		out = (q.template cast<T>().template segment<1>(0) * (var2_ - var3_).template segment<1>(0) * (var2_ - var3_).template segment<1>(0) + q.template cast<T>().template segment<1>(1) * (var2_ - var3_).template segment<1>(1) * (var2_ - var3_).template segment<1>(1) + (var0_ - var1_).template segment<1>(0) * (var0_ - var1_).template segment<1>(0))[0];
	}
	MAKE_HESSIAN(LOpt, func7);
	func7Hessian<1, 1, 2, 2> func7;
	
	const Matrix<scalar_t, 2, 1> Const14 = {-1.0, -2e+20};
	
	Matrix<scalar_t, 2, 1> q;
	
	template<typename T>
	EIGEN_STRONG_INLINE void _dynamics(cVec<T, 2>& x, cVec<T, 1>& u, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 4, 1> tmp4;
		tmp4.template block<2, 1>(0, 0) = Matrix<T, 2, 1>::Constant(1) * sin((x.template segment<1>(0) + x.template segment<1>(1)).array()).matrix();
		tmp4.template block<1, 1>(2, 0) = cos((u).array()).matrix();
		tmp4.template block<1, 1>(3, 0) = x.template segment<1>(1);
		out = A.template cast<T>() * tmp4 + B.template cast<T>() * u;
	}
	
	Matrix<scalar_t, 2, 1> x0;
	
	const Matrix<scalar_t, 2, 4> A = {{0, 0, 1, 2}, {1, 0, 3, 4}};
	
	Matrix<scalar_t, 2, 1> B;
	

	struct Bob
	{	
		template<int num_outputs, int num_inputs>
		Eigen::Map<Eigen::Matrix<scalar_t, num_outputs, num_inputs>, 0, Eigen::Stride<Eigen::Dynamic, 1>>
			 get_jacobian_block(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jacobian, 
	 							int col, int valueStart)
		{
			using Map_t = Eigen::Map<Eigen::Matrix<scalar_t, num_outputs, num_inputs>, 0, Eigen::Stride<Eigen::Dynamic, 1>>;
	        using Stride_t = Eigen::Stride<Eigen::Dynamic, 1>;

	        int nnz = jacobian.outerIndexPtr()[col+1] - jacobian.outerIndexPtr()[col];
	        scalar_t* data = jacobian.valuePtr() + valueStart;
	        return Map_t(data, Stride_t(nnz, 1));
		}

		template<int blah>
		void test()
		{
			std::cout << "Hello from test\n";
		}
	};

	LOpt() :
		func1(this),
		func3(this),
		func7(this),
		func2(this)
	{
		q.array() = 1;
		x0.array() = 0.0;
		B << 1, 0;

		SparseMatrix<scalar_t> J(NUM_CON, NUM_VARS);
		constraints_sparse_initialize(J);

		// Set the offset sequence
        scalar_t* values = J.valuePtr();
		for(int i=0; i<J.nonZeros(); i++)
			values[i] = i;

        // nnz = jacobian.outerIndexPtr()[var_offsets[ind]+1] - jacobian.outerIndexPtr()[var_offsets[ind]],
        // data = jacobian.valuePtr() + jacobian.outerIndexPtr()[var_offsets[ind]] + col_offsets[ind],
        // Map_t(data, num_outputs, input_sizes, Stride_t(nnz, 1)).row(i) = deriv.template segment<input_sizes>(offset),
        // offset += input_sizes,

        // Get the offset location for each jacobian block
        std::cout << "J.coeff(eq3(), x(1)) = " << (Eigen::Index)(J.coeff(eq3(), x(1))) << std::endl;
        std::cout << "J.coeff(eq3(), x(1)) = " << (Eigen::Index)(J.coeff(eq3(), x(1))) << std::endl;

        Bob bob;
        // bob.template test<4>();
        std::cout << "J(eq3(), x(1)) = \n" << bob.template get_jacobian_block<eq3_t::len, x_t::len>(J, x(1), (int)(J.coeff(eq3(), x(1)))) << std::endl;
        std::cout << "J(eq3(), x(1)) = \n" << bob.template get_jacobian_block<eq3_t::len, x_t::len>(J, x(1), (int)(J.coeff(eq3(), x(1)))) << std::endl;

		// eq1.info(), {u.info(0), x.info(1)});
		// set_equation_sparsity(trip, eq2.info(), {uss.info(), xss.info()});
		// set_equation_sparsity(trip, eq3.info(), {u.info(1), x.info(1), x.info(2)});
		// set_equation_sparsity(trip, eq4.info(), {u.info(2), x.info(2), x.info(3)});
		// set_equation_sparsity(trip, eq5.info(), {u.info(3), x.info(3), x.info(4)});


		std::cout << "J = \n" << J << std::endl;

		// func3({u(1), x(1), x(2)}, {0, 2, 0}, eq3(), var, constraints, jacobian);

		variable_t var;
		for(int i=0; i<NUM_VARS; i++) var[i] = i;
		auto ret = func3(u(var, 1), x(var, 1), x(var, 2));

		std::cout << "ret.val = " << ret.val.transpose() << std::endl;
		std::cout << "ret.jacobian = \n" << ret.jacobian << std::endl;
	}

};
