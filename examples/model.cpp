// #include <tuple>
// #include <functional>
#include <math.h>
#include <iostream>
#include <string>
#include <type_traits>

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

// #include "problemBase.hpp"
// #include "test.hpp"

#include "polygen_helper.hpp"


using namespace Eigen;
using namespace std;

// TODO:
// 1. Function with scalar output
// 2. First structured function (sum of functions)


// // User-written class
// struct MyNLP : public ThisIsMyNLPBase<MyNLP>
// {
// 	float scl = 1.0;

// 	template<typename T>
//     EIGEN_STRONG_INLINE void sys(const Ref<const x_t<T>>& xp, 
//     							 const Ref<const x_t<T>>& x, 
//     							 const Ref<const u_t<T>>& u,
//     							 Ref<x_t<T>> out) const noexcept
//     {
//     	out(0) = scl*(xp.sum());// + 2*x.sum() + 3*u.sum());
//     	out(1) = scl*(2*x.sum());
//     	out(2) = scl*(3*u.sum());
//     }

// 	template<typename T>
// 	EIGEN_STRONG_INLINE T stage_cost(const Eigen::Ref<const x_t<T>>& x,
// 								     const Eigen::Ref<const u_t<T>>& u)
// 	{
// 		return x.sum() + 2*u.sum();
// 	}
// };


template<typename scalar_t>
struct Test
{
	using Matrix_A = Eigen::Matrix<scalar_t, 3, 3>;
	Matrix_A A = (Matrix_A() << 1,2,3,4,5,6,7,8,9).finished();
	using Matrix_B = Eigen::Matrix<scalar_t, 3, 2>;
	Matrix_B B = (Matrix_B() << 1,3,5,2,4,6).finished();
	scalar_t h = 0.1;

template<typename T>
EIGEN_STRONG_INLINE void _sys(
	const Eigen::Ref<const Eigen::Matrix<T, 3, 1>>& x, 
	const Eigen::Ref<const Eigen::Matrix<T, 2, 1>>& u, 
	Eigen::Ref<Eigen::Matrix<T, 3, 1>> dx) const noexcept
{
	dx = A * x + B * u;
}
MAKE_JACOBIAN(sys, Test<scalar_t>, 3, 3, 2);
template<typename T>
EIGEN_STRONG_INLINE void _rk4(
	const Eigen::Ref<const Eigen::Matrix<T, 3, 1>>& x, 
	const Eigen::Ref<const Eigen::Matrix<T, 2, 1>>& u, 
	Eigen::Ref<Eigen::Matrix<T, 3, 1>> out) const noexcept
{
	Eigen::Matrix<T, 3, 1> tmp1;
	_sys<T>(x, u, tmp1);
	Eigen::Matrix<T, 3, 1> tmp2;
	_sys<T>((x + h * 0.5 * tmp1), u, tmp2);
	Eigen::Matrix<T, 3, 1> tmp3;
	_sys<T>((x + h * 0.5 * tmp2), u, tmp3);
	Eigen::Matrix<T, 3, 1> tmp4;
	_sys<T>((x + h * tmp3), u, tmp4);
	out = x + h * 0.1667 * (tmp1 + 2 * tmp2 + 2 * tmp3 + tmp4);
}
MAKE_JACOBIAN(rk4, Test<scalar_t>, 3, 3, 2);

    Test() : sys(this), rk4(this) {};
};


using scalar_t = double;
Test<scalar_t> t;

// #define SIM_EXPORT __declspec(dllexport)

#if defined(__cplusplus)
extern "C" {
#endif

	// SIM_EXPORT 
	void test() {
		std::cout << "HELLO WORLD" << std::endl;
	}

	void test2(double* ptr) {
		std::cout << "HELLO WORLD 2 :" << ptr[0] << std::endl;
	}

	// void callme(double* _x, double* _u, double* _xp, double* _Jx, double* _Ju) {
	void callme(double* _x, double* _u, double* _xp) {
		// std::cout << "HELLO WORLD 2 :" << _x[0] << std::endl;

		Map<Matrix<double, 3, 1>> x(_x);
		Map<Matrix<double, 2, 1>> u(_u);

		// Map<Eigen::Matrix<scalar_t, 3, 3>> Jx(_Jx);
		// Map<Eigen::Matrix<scalar_t, 3, 2>> Ju(_Ju);
		Map<Eigen::Matrix<scalar_t, 3, 1>> xp(_xp);

		// std::cout << "x = " << x.transpose() << std::endl;
		// t.rk4(x, u, xp, Jx, Ju);
		t.rk4(x, u, xp);

		// std::cout << "xp = " << xp.transpose() << std::endl;
		// std::cout << "Jx = \n" << Jx << std::endl;
	}


    // SIM_EXPORT Detector *DetectorNew() { return new Detector(); }
    // SIM_EXPORT void DetectorProcess(Detector *pDet, int *pIn, int *pOut, int n) {
    //     pDet->process(pIn, pOut, n);
    // }
    // SIM_EXPORT void DetectorDelete(Detector *pDet) { delete pDet; }

#if defined(__cplusplus)
}
#endif


#include <dlfcn.h>

extern "C" {
#include "gen.h"
}

int main(void)
{
	Eigen::Matrix<scalar_t, 3, 1> x;
	Eigen::Matrix<scalar_t, 3, 1> dx;
	Eigen::Matrix<scalar_t, 2, 1> u;

	x << 1,2,3;
	u << 1,2;

	t.sys(x, u, dx);

	std::cout << "x = " << x.transpose() << std::endl;
	std::cout << "u = " << u.transpose() << std::endl;
	std::cout << "dx = " << dx.transpose() << std::endl;

	Eigen::Matrix<scalar_t, 3, 3> Jx;
	Eigen::Matrix<scalar_t, 3, 2> Ju;

	t.sys(x, u, dx, Jx, Ju);

	std::cout << "==================================\n";
	std::cout << "dx = " << dx.transpose() << std::endl;
	std::cout << "Jx = \n" << Jx << std::endl;
	std::cout << "Ju = \n" << Ju << std::endl;


	std::cout << "==================================\n";
	Eigen::Matrix<scalar_t, 3, 1> xp;

	t.rk4(x, u, xp);
	t.rk4(x, u, xp, Jx, Ju);

	std::cout << "xp = " << xp.transpose() << std::endl;
	std::cout << "Jx = \n" << Jx << std::endl;
	std::cout << "Ju = \n" << Ju << std::endl;

	using clock = std::chrono::system_clock;
	// using ms = std::chrono::duration<double, std::milli>;
	using sec = std::chrono::duration<double>;

	const auto before = clock::now();
	t.h = 0.001;
	for(int i=0; i<100000; i++)
		t.rk4(x, u, xp);
	const sec duration = clock::now() - before;

	std::cout << "It took " << (duration.count()) / (double)(100000) << "sec" << std::endl;



	// t.rk4(x, u, xp, Jx, Ju);

	// std::cout << "xp = " << xp.transpose() << std::endl;
	// std::cout << "Jx = \n" << Jx << std::endl;
	// std::cout << "Ju = \n" << Ju << std::endl;

	// MyNLP mpc;

	// MyNLP::nlp_variable_t var;
	// MyNLP::nlp_constraints_t equalities;
	// MyNLP::nlp_eq_jacobian_t jacobian;

	// for(int i=0; i<var.rows(); i++) var(i) = ((MyNLP::scalar_t)i)/10.0;
	// jacobian.setZero();
	// equalities.setZero();

	// mpc.equalities(var, equalities);
	// cout << "Equalities = " << equalities.transpose() << endl;

	// // for(int i=0; i<1e6; i++)
	// 	mpc.equalities_linearised(var, equalities, jacobian);
	// cout << "Equalities = " << equalities.transpose() << endl;
	// cout << "Jacobian = \n" << jacobian << endl;


	// cout << "EIGEN version : " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

	// // cout << "=============================================\n";
	// // mpc.scl = 2.0;
	// // mpc.equalities(var, equalities);
	// // cout << "Equalities = " << equalities.transpose() << endl;

	// // mpc.equalities_linearised(var, equalities, jacobian);
	// // cout << "Equalities = " << equalities.transpose() << endl;
	// // cout << "Jacobian = \n" << jacobian << endl;

	return 0;
}
