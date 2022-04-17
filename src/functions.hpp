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
	template<typename scalar_t, typename diff_t=scalar_t, 
	         int n, // Dimension of x
	         typename O, typename... Parameters>
	EIGEN_STRONG_INLINE Eigen::Vector<diff_t, n> rk4(O ode, const scalar_t _h, const Eigen::Ref<const Eigen::Vector<diff_t, n>>& x, 
	                             const Parameters&... params) noexcept
	{
	  diff_t h = static_cast<diff_t>(_h);
	  auto k1 = ode(x,                               params...);
	  auto k2 = ode(x+h/static_cast<diff_t>(2.0)*k1, params...);
	  auto k3 = ode(x+h/static_cast<diff_t>(2.0)*k2, params...);
	  auto k4 = ode(x+h*k3,                          params...);
	  return x + h/static_cast<diff_t>(6.0) * (k1 + static_cast<diff_t>(2.0)*k2 + static_cast<diff_t>(2.0)*k3 + k4);
	}

	/**
	 * Identity function
	 */
	template<typename V>
	EIGEN_STRONG_INLINE V id(lampc::Eval, const Eigen::Ref<const V>& x) noexcept
	{
		V y(x.rows());
		y = x;
		return y;
	}

	// Tape version
	template<typename V>
	EIGEN_STRONG_INLINE auto id(lampc::Jacobian, const std::pair<lampc::Segment, Eigen::Ref<V>> x) noexcept
	{
		using scalar_t = typename V::Scalar;
		int rows = x.second.rows();

		V value(rows);
		value = x.second;

		Eigen::MatrixX<scalar_t> jac = Eigen::MatrixX<scalar_t>::Identity(rows, rows);

		return std::make_pair(std::vector<lampc::Segment>{x.first},	std::make_pair(value, jac));
	}

	// Deployment version
	template<typename V>
	EIGEN_STRONG_INLINE std::pair<V, Eigen::MatrixX<typename V::Scalar>> id(lampc::Jacobian, Eigen::Ref<V> x) noexcept
	{
		using scalar_t = typename V::Scalar;

		V value(x.rows());
		value = x;

		Eigen::MatrixX<scalar_t> jac = Eigen::MatrixX<scalar_t>::Identity(x.rows(), x.rows());

		return std::make_pair(value, jac);
	}

	// TODO: Create a method to produce hessians for functions where the number of rows is not known at compile-time...
	// template<typename V>
	// EIGEN_STRONG_INLINE Eigen::MatrixX<V::Scalar> id(lampc::Hessian, const Eigen::Ref<const V>& x) noexcept
	// {
	// 	std::vector<Eigen::MatrixX<V::Scalar> out = Eigen::MatrixX<V::Scalar>::identity(x.rows(), x.rows());
	// 	return out;
	// }

};

#endif