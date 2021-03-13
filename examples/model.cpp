#include <tuple>
#include <functional>
#include <math.h>
#include <iostream>
#include <string>
#include <type_traits>

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "problemBase.hpp"
#include "test.hpp"

using namespace Eigen;
using namespace std;

// TODO:
// 1. Function with scalar output
// 2. First structured function (sum of functions)


// User-written class
struct MyNLP : public ThisIsMyNLPBase<MyNLP>
{
	float scl = 1.0;

	template<typename T>
    EIGEN_STRONG_INLINE void sys(const Ref<const x_t<T>>& xp, 
    							 const Ref<const x_t<T>>& x, 
    							 const Ref<const u_t<T>>& u,
    							 Ref<x_t<T>> out) const noexcept
    {
    	out(0) = scl*(xp.sum());// + 2*x.sum() + 3*u.sum());
    	out(1) = scl*(2*x.sum());
    	out(2) = scl*(3*u.sum());
    }

	template<typename T>
	EIGEN_STRONG_INLINE T stage_cost(const Eigen::Ref<const x_t<T>>& x,
								     const Eigen::Ref<const u_t<T>>& u)
	{
		return x.sum() + 2*u.sum();
	}
};


int main(void)
{
	MyNLP mpc;

	MyNLP::nlp_variable_t var;
	MyNLP::nlp_constraints_t equalities;
	MyNLP::nlp_eq_jacobian_t jacobian;

	for(int i=0; i<var.rows(); i++) var(i) = ((MyNLP::scalar_t)i)/10.0;
	jacobian.setZero();
	equalities.setZero();

	mpc.equalities(var, equalities);
	cout << "Equalities = " << equalities.transpose() << endl;

	// for(int i=0; i<1e6; i++)
		mpc.equalities_linearised(var, equalities, jacobian);
	cout << "Equalities = " << equalities.transpose() << endl;
	cout << "Jacobian = \n" << jacobian << endl;


	cout << "EIGEN version : " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

	// cout << "=============================================\n";
	// mpc.scl = 2.0;
	// mpc.equalities(var, equalities);
	// cout << "Equalities = " << equalities.transpose() << endl;

	// mpc.equalities_linearised(var, equalities, jacobian);
	// cout << "Equalities = " << equalities.transpose() << endl;
	// cout << "Jacobian = \n" << jacobian << endl;

	return 0;
}
