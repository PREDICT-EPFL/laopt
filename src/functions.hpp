#ifndef __LAMPC__FUNCTIONS_HPP
#define __LAMPC__FUNCTIONS_HPP

/**
 * Provides a class of common functions for defining OCPs
 */

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

namespace lampc {
	/**
	 * Simple general RK4 integrator
	 */
	template<typename ODE, template<typename,typename> class DiffType, typename scalar_t, size_t num_states, size_t... param_sizes>
	struct RK4 : public DiffType<RK4<ODE,DiffType,scalar_t,num_states,param_sizes...>, lampc::FuncInfo<scalar_t, num_states, num_states, param_sizes...>>
	{
	  ODE& ode; // Reference to the callable ode object
	  scalar_t h; // Step size

	  RK4(ODE& ode, scalar_t h) : ode(ode), h(h) {}

	  template<typename diff_t>
	  EIGEN_STRONG_INLINE Eigen::Vector<diff_t, num_states> 
	  impl(const Eigen::Ref<const Eigen::Vector<diff_t, num_states>>& x,
	       const Eigen::Ref<const Eigen::Vector<diff_t, param_sizes>>&... params) noexcept
	  {
	    diff_t _h = static_cast<diff_t>(h);
	    auto k1 = ode.template impl<diff_t>(x,                               params...);
	    auto k2 = ode.template impl<diff_t>(x+_h/static_cast<diff_t>(2.0)*k1, params...);
	    auto k3 = ode.template impl<diff_t>(x+_h/static_cast<diff_t>(2.0)*k2, params...);
	    auto k4 = ode.template impl<diff_t>(x+_h*k3,                          params...);
	    return x + _h/static_cast<diff_t>(6.0) * (k1 + static_cast<diff_t>(2.0)*k2 + static_cast<diff_t>(2.0)*k3 + k4);
	  }
	};

	// /**
	//  * Identity function
	//  */
	// template<int n, typename V>
	// EIGEN_STRONG_INLINE auto id(lampc::Eval, const std::pair<lampc::Segment, Eigen::Ref<V>> x) noexcept
	// {
	// 	assert(x.second.rows() == n && "Invalid size passed to id");
	// 	using scalar_t = typename V::Scalar;
	// 	using Func = lampc::DFunction<scalar_t, n, n>;
	// 	typename lampc::CallTape<Func> ret{x.first};

	// 	ret.value = x.second;
	// 	return ret;
	// }

	// template<int n, typename V>
	// EIGEN_STRONG_INLINE auto id(lampc::Eval, Eigen::Ref<V> x) noexcept
	// {
	// 	assert(x.rows() == n && "Invalid size passed to id");
	// 	using scalar_t = typename V::Scalar;
	// 	using Func = lampc::DFunction<scalar_t, n, n>;
	// 	typename lampc::Call<Func> ret;

	// 	ret.value = x;
	// 	return ret;
	// }

	// // Tape version
	// template<int n, typename V>
	// EIGEN_STRONG_INLINE auto id(lampc::Jacobian, const std::pair<lampc::Segment, Eigen::Ref<V>> x) noexcept
	// {
	// 	assert(x.second.rows() == n && "Invalid size passed to id");
	// 	using scalar_t = typename V::Scalar;
	// 	using Func = lampc::DFunction<scalar_t, n, n>;
	// 	typename lampc::JacobianTapeCall<Func> ret{x.first};

	// 	ret.value = x.second;
	// 	ret.jacobian = Eigen::Matrix<scalar_t, n, n>::Identity();

	// 	return ret;
	// }

	// // Deployment version
	// template<int n, typename V>
	// EIGEN_STRONG_INLINE auto id(lampc::Jacobian, Eigen::Ref<V> x) noexcept
	// {
	// 	assert(x.rows() == n && "Invalid size passed to id");
	// 	using scalar_t = typename V::Scalar;
	// 	using Func = lampc::DFunction<scalar_t, n, n>;
	// 	typename lampc::JacobianCall<Func> ret;

	// 	ret.value = x;
	// 	ret.jacobian = Eigen::Matrix<scalar_t, n, n>::Identity();

	// 	return ret;
	// }

	// // TODO: Create a method to produce hessians for functions where the number of rows is not known at compile-time...
	// // template<typename V>
	// // EIGEN_STRONG_INLINE Eigen::MatrixX<V::Scalar> id(lampc::Hessian, const Eigen::Ref<const V>& x) noexcept
	// // {
	// // 	std::vector<Eigen::MatrixX<V::Scalar> out = Eigen::MatrixX<V::Scalar>::identity(x.rows(), x.rows());
	// // 	return out;
	// // }

};

#endif