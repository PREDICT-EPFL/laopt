#include <math.h>
#include <map>
#include "Eigen/Dense"
#include <Eigen/Sparse>
#include "unsupported/Eigen/AutoDiff"

#include "polygen_helper.hpp"

using namespace Eigen;

template<typename scalar_t>
struct LOpt
{
	// Define variable accessors and ordering
	using xss_t = var_t<2, 1>;
	using x_t = var_t<2, 5, xss_t>;
	using u_t = var_t<1, 4, x_t>;
	using uss_t = var_t<1, 1, u_t>;
	
	static constexpr auto xss = xss_t();
	static constexpr auto x = x_t();
	static constexpr auto u = u_t();
	static constexpr auto uss = uss_t();
	
	// Define constraint accessors and ordering
	using eq1_t = con_t<2, 1, 6>;
	using eq2_t = con_t<2, 1, 6, eq1_t>;
	using eq3_t = con_t<2, 3, 10, eq2_t>;
	
	static constexpr auto eq1 = eq1_t();
	static constexpr auto eq2 = eq2_t();
	static constexpr auto eq3 = eq3_t();
	
	// Convenient user-accessors
	std::array<con_slow_t, 3> constraint_list = {con_slow_t{eq1, "eq1"}, {eq2, "eq2"}, {eq3, "eq3"}};
	
	// Define NLP sizes
	enum
	{
		NUM_VARS = uss.next,
		NUM_CON   = eq3.next,
		nnz_constraints_jacobian = eq1.nnz + eq2.nnz + eq3.nnz
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
		for(int i=1, eq_ind=0; i<4; i+=1, eq_ind++)
			set_equation_sparsity(trip, eq3.info(eq_ind), {u.info(i), x.info((i+1)), x.info(i)});
		
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
		func1({u(0), x(1)}, {0, 0}, eq1(), var, constraints, jacobian);
		func2({uss(), xss()}, {0, 0}, eq2(), var, constraints, jacobian);
	template<typename T>
	EIGEN_STRONG_INLINE void _func1(cVec<T, 1>& var0_, cVec<T, 2>& var1_, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 2, 1> tmp1;
		_dynamics<T>(x0.template cast<T>(), var0_, tmp1);
		out = var1_ - tmp1;
	}
	MAKE_JACOBIAN(func1, LOpt<scalar_t>, 2, 1, 2);
	
	template<typename T>
	EIGEN_STRONG_INLINE void _func2(cVec<T, 1>& var0_, cVec<T, 2>& var1_, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 2, 1> tmp2;
		_dynamics<T>(var1_, var0_, tmp2);
		out = var1_ - tmp2;
	}
	MAKE_JACOBIAN(func2, LOpt<scalar_t>, 2, 1, 2);
	
	template<typename T>
	EIGEN_STRONG_INLINE void _func3(cVec<T, 1>& var0_, cVec<T, 2>& var1_, cVec<T, 2>& var2_, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 2, 1> tmp3;
		_dynamics<T>(var2_, var0_, tmp3);
		out = var1_ - tmp3;
	}
	MAKE_JACOBIAN(func3, LOpt<scalar_t>, 2, 1, 2, 2);
	
	Matrix<scalar_t, 2, 1> x0;
	
	template<typename T>
	EIGEN_STRONG_INLINE void _dynamics(cVec<T, 2>& x, cVec<T, 1>& u, Vec<T, 2> out) const noexcept
	{
		Matrix<T, 4, 1> tmp4;
		tmp4.template block<2, 1>(0, 0) = Matrix<T, 2, 1>::Constant(1) * sin((x.template segment<1>(0) + x.template segment<1>(1)).array()).matrix();
		tmp4.template block<1, 1>(2, 0) = cos((u).array()).matrix();
		tmp4.template block<1, 1>(3, 0) = x.template segment<1>(1);
		out = A.template cast<T>() * tmp4 + B.template cast<T>() * u;
	}
	
	Matrix<scalar_t, 2, 1> B;
	
	const Matrix<scalar_t, 2, 4> A = {{0, 0, 1, 2}, {1, 0, 3, 4}};
	
	LOpt() :
		func1(this),
		func2(this),
		func3(this)
	{
		x0.array() = 0.0;
		B << 1, 0;
	}

};
