#include <iostream>
#include <numeric>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <type_traits>

#include <functional>

#include "lampc.hpp"


/**
 * General RK4 integrator
 * 
 * Takes a pointer to the ODE and a step size, and returns the RK4 integration.
 */
template<typename diff_t, int n, int m>
using SysFunc_ptr = std::add_pointer_t<Eigen::Vector<diff_t, n>(
		const Eigen::Ref<const Eigen::Vector<diff_t, n>>&, 
		const Eigen::Ref<const Eigen::Vector<diff_t, m>>&)>;

template<typename scalar_t=double, typename diff_t=scalar_t, int n, int m>
EIGEN_STRONG_INLINE 
Eigen::Vector<diff_t, n> 
rk4(SysFunc_ptr<diff_t,n,m> ode, 
		const Eigen::Ref<const Eigen::Vector<diff_t, n>>& x, 
		const Eigen::Ref<const Eigen::Vector<diff_t, m>>& u,
		const scalar_t _h) noexcept
{
	diff_t h = static_cast<diff_t>(_h);
  Eigen::Vector<diff_t, n> k1 = ode(x,       u);
  Eigen::Vector<diff_t, n> k2 = ode(x+h/static_cast<diff_t>(2.0)*k1,u);
  Eigen::Vector<diff_t, n> k3 = ode(x+h/static_cast<diff_t>(2.0)*k2,u);
  Eigen::Vector<diff_t, n> k4 = ode(x+h*k3,  u);
  return x + h/static_cast<diff_t>(6.0) * (k1 + static_cast<diff_t>(2.0)*k2 + static_cast<diff_t>(2.0)*k3 + k4);
}

template<typename scalar_t, typename diff_t=scalar_t, typename O, typename XArg, typename... Args>
EIGEN_STRONG_INLINE auto rk4(O ode, const scalar_t _h, XArg& x, Args&... args) noexcept
{
	diff_t h = static_cast<diff_t>(_h);
  auto k1 = ode(x,       												 args...);
  auto k2 = ode(x+h/static_cast<diff_t>(2.0)*k1, args...);
  auto k3 = ode(x+h/static_cast<diff_t>(2.0)*k2, args...);
  auto k4 = ode(x+h*k3,  												 args...);
  return x + h/static_cast<diff_t>(6.0) * (k1 + static_cast<diff_t>(2.0)*k2 + static_cast<diff_t>(2.0)*k3 + k4);
}


template<typename scalar_t>
struct Test
{
  Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
  Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};

	template<typename diff_t=scalar_t>
	EIGEN_STRONG_INLINE Vector<diff_t, 2> 
		sys(const Ref<const Matrix<diff_t, 2, 1>>& x, const Ref<const Vector<diff_t, 1>>& u) noexcept
	{
	  return ((((x(0)) * (A.template cast<diff_t>()).array()).matrix()) * (x)) + ((B.template cast<diff_t>()) * (u));
	}

	template<typename diff_t=scalar_t>
	EIGEN_STRONG_INLINE Vector<diff_t, 2> dsys(
			const Ref<const Vector<diff_t, 2>>& xp, 
			const Ref<const Vector<diff_t, 2>>& x, 
			const Ref<const Vector<diff_t, 1>>& u) noexcept
	{
		auto self=this;
		return xp - rk4<scalar_t, diff_t>([self](auto&... args){return self->template sys<diff_t>(args...);}, 0.1, x, u);
	}

  make_differentiable(dsys, 2, 2,2,1);

  lampc::Problem<scalar_t> prob;
  lampc::Variable<scalar_t>& x;
  lampc::Variable<scalar_t>& u;

  Eigen::VectorX<scalar_t> var; // The problem variable

  template<typename D, typename T>
	void shoot(T& tape)
	{
		for(int i=0; i<x().cols()-1; i++)
			std::tie(std::ignore, tape(-1,{x[i],u[i],x[i+1]})) = dsys(D(),x(i+1),u(i),x(i));
	};

	SparseMatrix<scalar_t>& shoot()
	{
		shoot<lampc::Jacobian>(mat);
		return S;
	}

	lampc::BSMatrix<scalar_t> mat;
	SparseMatrix<scalar_t> S;

  Test(int N)	: x(prob.variable(2,N)), u(prob.variable(1,N)), var(prob.size)
  {
		prob.set_variable(var);

		// Create the tape to record the copy sequence
		lampc::BSMatrixTape<scalar_t> tape;
		shoot<lampc::Jacobian>(tape);
		tape.finalize_structure();
		shoot<lampc::Jacobian>(tape);

		std::cout << "Sparsity structure = \n" << Eigen::MatrixX<int>(tape.get_sparsity_structure()) << std::endl;
		std::cout << "Copies = " << tape.copy_sequence << std::endl;
		std::cout << "\n\n";

		// We now replace the tape with a block-sparse matrix
		// which loads the copy-sequence from the tape (or from file)
		mat.initialize_from_tape(tape);
		mat.initialize_matrix(S);
  }
};

int main()
{
	Test<double> t(10);

	std::cout << "t.shoot() = \n" << t.shoot() << std::endl;

	// Eigen::Vector<double,2> x;
	// Eigen::Vector<double,1> u;

	// x << 1,2;
	// u << 3;

	// t.test_function<double>(x,u);
	
	// Eigen::Vector<double,2> val;
	// Eigen::Matrix<double,2,3> J;
	// std::tie(val,J) = t.test_function<double>(lampc::Jacobian(), x,u);
	// std::cout << "val = " << val.transpose() << std::endl;
	// std::cout << "J = \n" << J << std::endl;

	// using scalar_t = float;
	// using Problem = lampc::Problem<scalar_t>; 

	// Problem prob;

	// constexpr int N = 5;
	// auto x = prob.variable(2, N);
	// auto u = prob.variable(1, N-1);

	// std::cout << "type(x) = " << type_name<decltype(x)>() << std::endl;
	// std::cout << "&x = " << &x << std::endl;

	// Eigen::VectorX<scalar_t> var(prob.size);
	// prob.set_variable(var);

	// for(int i=0; i<x().cols(); i++)	x(i).array() = i;
	// for(int i=0; i<u().cols(); i++)	u(i).array() = i;

	// std::cout << "x = \n" << x() << std::endl;
	// std::cout << "u = \n" << u() << std::endl;

	// using DSYS = lampc::Function<scalar_t, 2, 2,2,1>;
	// auto dsys = make_function(DSYS, _dsys);

	// std::cout << "dsys(x, u) = " << dsys(x(1),x(0), u(0)).transpose() << std::endl;
	// std::cout << "jac dsys(x, u) = \n" << std::get<1>(dsys(x(1),x(0), u(0), lampc::Jacobian())) << std::endl;
	// std::cout << "hess dsys(x, u) = \n" << std::get<2>(dsys(x(1),x(0), u(0), lampc::Hessian()))[0] << std::endl;

	// auto shoot = [&](auto& tape)
	// {
	// 	for(int i=0; i<x().cols()-1; i++)
	// 		std::tie(std::ignore, tape(-1,{x[i],u[i],x[i+1]})) = dsys(x(i+1),u(i),x(i),lampc::Jacobian());
	// };

	// // Create the tape to record the copy sequence
	// lampc::BSMatrixTape<scalar_t> tape;
	// shoot(tape);
	// tape.finalize_structure();
	// shoot(tape);

	// std::cout << "Sparsity structure = \n" << Eigen::MatrixX<int>(tape.get_sparsity_structure()) << std::endl;
	// std::cout << "Copies = " << tape.copy_sequence << std::endl;
	// std::cout << "\n\n";

	// // We now replace the tape with a block-sparse matrix
	// // which loads the copy-sequence from the tape (or from file)
	// lampc::BSMatrix<scalar_t> mat(tape);
	// SparseMatrix<scalar_t> S;
	// mat.initialize_matrix(S);

	// // Now we call the function and it just plays back the copy sequence
	// shoot(mat);
	// std::cout << "S = \n" << Eigen::MatrixX<scalar_t>(S).format(Eigen::IOFormat(2)) << std::endl;

	// constexpr int NUM_EXP = 1000;
 //  auto start = std::chrono::steady_clock::now();
 //  for(int i = 0; i < NUM_EXP; ++i)
 //  {
	// 	x(2)(0)++;
	// 	shoot(mat);
 //  }
 //  auto end = std::chrono::steady_clock::now();

 //  std::cout << "Time to compute the block-sparse jacobian: "
 //      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)NUM_EXP
 //      << " us" << std::endl;

	return 0;
}
