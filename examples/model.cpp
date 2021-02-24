#include <tuple>
#include <functional>
#include <math.h>
#include <iostream>

#include <Eigen/Dense>
#include "unsupported/Eigen/AutoDiff"

// #include "ADJacobian.h"

#include "problemBase.hpp"
#include "test.hpp"



// User-written class
struct MyNLP : public MyNLPBase<MyNLP>
{
	nlp_variable_t bob;
	scalar_t y;

	template<typename T>
    EIGEN_STRONG_INLINE void sys(const Eigen::Ref<const x_t<T>>& xp, 
    							 const Eigen::Ref<const x_t<T>>& x, 
    							 const Eigen::Ref<const u_t<T>>& u,
    							 Eigen::Ref<x_t<T>> out) const noexcept
    {
    	out(0) = xp.sum() + 2*x.sum() + 3*u.sum();
    	out(1) = xp.sum() + 2*x.sum() + 3*u.sum();
    	out(2) = xp.sum() + 2*x.sum() + 3*u.sum();
    	// out(3) = xp.sum() + 2*x.sum() + 3*u.sum(); 
    	// std::cout << "out = " << out.transpose() << std::endl;

    	// std::cout << "IN SYS" << std::endl;
    }

	template<typename T>
	EIGEN_STRONG_INLINE scalar_t stage_cost(const Eigen::Ref<const x_t<T>>& x,
								     		const Eigen::Ref<const u_t<T>>& u)
	{
		return x.sum() + 2*u.sum();
	}
};



template<typename Functor>
EIGEN_STRONG_INLINE double test_call(Functor functor)
{
	return functor();
}



template<typename scalar_t>
void test_function(
		const Eigen::Ref<const Eigen::Matrix<scalar_t, 3, 1>>& xp,
		const Eigen::Ref<const Eigen::Matrix<scalar_t, 3, 1>>& x,
		const Eigen::Ref<const Eigen::Matrix<scalar_t, 2, 1>>& u,
		Eigen::Ref<Eigen::Matrix<scalar_t, 3, 1>> out
	)
{
	out(0) = xp.sum();
	out(1) = 2*x.sum();
	out(2) = 3*u.sum();

	std::cout << "I'VE BEEN CALLED" << std::endl;
}


int main(void)
{

 //    // using Solver = SQPSolver<MyMPC>;
 //    // Solver solver;

    MyNLP mpc;
	using scalar_t = typename MyNLP::scalar_t;

    for (int i=0; i<MyNLP::VAR_SIZE; i++) mpc.bob(i) = (double)i;
    std::cout << "mpc.bob = " << mpc.bob.transpose() << std::endl;

	for (int i=0; i<MyNLP::x_numvecs; i++)
		std::cout << "X(" << i << ") = " << mpc.x(mpc.bob, i).transpose() << std::endl;

	std::cout << "uss = " << mpc.uss(mpc.bob).transpose() << std::endl;
	std::cout << "mpc.x(2)= " << mpc.x(mpc.bob, 2).transpose() << std::endl;

	MyNLP::x_t<scalar_t> err;
	
	mpc.sys<scalar_t>(
			mpc.x(mpc.bob, 1), 
			mpc.x(mpc.bob, 0),
			mpc.u(mpc.bob, 0), 
			err);

	std::cout << "sys(xp,x,u) = " << err.transpose() << std::endl;

	std::cout << std::endl << std::endl;
	std::cout << "---------------------------------------" << std::endl;
	std::cout << std::endl << std::endl;

	MyNLP::nlp_variable_t var;
	for(int i=0; i<MyNLP::VAR_SIZE; i++) var(i) = i;
	MyNLP::nlp_constraints_t equalities;
	mpc.equalities(var, equalities);

	std::cout << "var = " << var.transpose() << std::endl;
	std::cout << "xss = " << mpc.xss(var).transpose() << std::endl;
	std::cout << "equalities = " << equalities.transpose() << std::endl;

	std::cout << std::endl << std::endl;
	std::cout << "---------------------------------------" << std::endl;
	std::cout << std::endl << std::endl;

	MyNLP::x_t<scalar_t> xp;
	MyNLP::x_t<scalar_t> x;
	MyNLP::u_t<scalar_t> u;
	Eigen::Matrix<scalar_t, MyNLP::n, 1> out;

	Eigen::Matrix<scalar_t, MyNLP::n, MyNLP::x_t_size> J_xp;
	Eigen::Matrix<scalar_t, MyNLP::n, MyNLP::x_t_size> J_x;
	Eigen::Matrix<scalar_t, MyNLP::n, MyNLP::u_t_size> J_u;

	xp << 1,2,3;
	x << 5,6,7;
	u << 9,10;

    mpc.sys_jacobian(xp, x, u, out, J_xp, J_x, J_u);

	std::cout << "out = " << out.transpose() << std::endl;
	std::cout << "J_xp = " << std::endl << J_xp << std::endl;
	std::cout << "J_x = " << std::endl << J_x << std::endl;
	std::cout << "J_u = " << std::endl << J_u << std::endl;

	std::cout << std::endl << std::endl;
	std::cout << "---------------------------------------" << std::endl;
	std::cout << std::endl << std::endl;

	{

	// std::cout << "TESTING" << std::endl;

	MyNLP::nlp_variable_t var;
	MyNLP::nlp_constraints_t equalities;
	MyNLP::nlp_eq_jacobian_t jacobian;

	equalities.setZero();
	jacobian.setZero();

	for (int i=0; i<var.rows(); i++) var(i) = i;
	mpc.equalities_linearised(var, equalities, jacobian);

	std::cout << "equalities = " << equalities.transpose() << std::endl;
	std::cout << "jacobian = " << std::endl;
	std::cout << jacobian << std::endl;

	}

	std::cout << std::endl << std::endl;
	std::cout << "---------------------------------------" << std::endl;
	std::cout << std::endl << std::endl;

	{
	using scalar_t = double;
	Eigen::Matrix<scalar_t, 3, 1> xp;
	Eigen::Matrix<scalar_t, 3, 1> x;
	Eigen::Matrix<scalar_t, 2, 1> u;
	Eigen::Matrix<scalar_t, 3, 1> out;

	xp << 1,2,3;
	x << 4,5,6;
	u << 7,9;
	out.setZero();

	std::cout << "Calling test_function directly";
	test_function<scalar_t>(xp, x, u, out);
	std::cout << out.transpose() << std::endl;

// template<int num_outputs, int ...n, typename scalar_t>
// void make_jacobian(

	Eigen::Matrix<scalar_t, 3, 3> J_xp;
	Eigen::Matrix<scalar_t, 3, 3> J_x;
	Eigen::Matrix<scalar_t, 3, 2> J_u;

	J_x << 1,2,3,4,5,6,7,8,9;

	std::cout << "Calling through make_jacobian";
	// using AD_test_function = test_function<AutoDiffScalar<Matrix<scalar_t, 8, 1>>>;


	// jacobian<scalar_t, 3, 8, 3,3,2>(test_function<AD_scalar<scalar_t, 8>>,
	// 	out,
	// 	xp, x, u,
	// 	J_xp, J_x, J_u);
	std::cout << out.transpose() << std::endl;

	Eigen::Matrix<scalar_t, 5, 1> outxx;

	Eigen::Matrix<scalar_t, 10,1> var;
	var << 1,2,3,4,5,6,7,8,9,10;

	using ad_sys_t = AD_scalar<scalar_t, 8>;
	auto ad_sys = [&mpc](
		const Eigen::Ref<const MyNLP::x_t<ad_sys_t>>& xp, 
    	const Eigen::Ref<const MyNLP::x_t<ad_sys_t>>& x, 
    	const Eigen::Ref<const MyNLP::u_t<ad_sys_t>>& u,
    	Eigen::Ref<MyNLP::x_t<ad_sys_t>> out
		){mpc.sys<ad_sys_t>(xp,x,u,out);};

	// for (int j=0; j<10; j++)
	// for (int i=0; i<2e6; i++)
	call_jacobian<scalar_t, 3,8, 3,3,2>(ad_sys,
		var.template segment<3>(4), x, u,
		outxx.template segment<3>(1),
		J_xp, J_x, J_u);


	std::cout << "Jacobians" << std::endl;
	std::cout << "J_xp" << std::endl;
	std::cout << J_xp << std::endl;
	std::cout << "J_x" << std::endl;
	std::cout << J_x << std::endl;
	std::cout << "J_u" << std::endl;
	std::cout << J_u << std::endl;
	}


	{

	// std::cout << "TESTING" << std::endl;

	MyNLP::nlp_variable_t var;
	MyNLP::nlp_constraints_t equalities;
	MyNLP::nlp_eq_jacobian_t jacobian;

	equalities.setZero();
	jacobian.setZero();

	for (int i=0; i<var.rows(); i++) var(i) = i;

	for (int j=0; j<10; j++)
	for (int i=0; i<2e6; i++)
	mpc.equalities_linearised(var, equalities, jacobian);

	std::cout << "equalities = " << equalities.transpose() << std::endl;
	std::cout << "jacobian = " << std::endl;
	std::cout << jacobian << std::endl;

	}

}
