#ifndef LAOPT_INVERTED_PENDULUM_SIMPLE_OCP_HPP
#define LAOPT_INVERTED_PENDULUM_SIMPLE_OCP_HPP

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"

class InvertedPendulumSimpleOcp : public laopt_tools::ControlProblemBase<
									 /*Numeric*/ double, /*NX*/ 2, /*NU*/ 1, /*NP*/ 0,
									 /*NG*/ 0, /*NG0*/ 0, /*NGF*/ 0,
									 /*Options*/ laopt_tools::FixedEndTime>
									 // /*Options*/ laopt_tools::FixedEndTime | laopt_tools::DiscreteDynamics>
{
public:
	Numeric angle_ref_{0};
	Eigen::Matrix<Numeric, NU, NU> R_{{1}};

	template <typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t,
	          typename tau_t, typename T = typename x_t::Scalar> // T is scalar type
	T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
	                     const Eigen::MatrixBase<u_t>& u,
	                     const Eigen::MatrixBase<p_t>& p,
	                     const Eigen::MatrixBase<t0_t>& t0,
	                     const Eigen::MatrixBase<tf_t>& tf,
	                     const tau_t& tau)
	{
		return get_L_x<T>(x) + get_L_u<T>(u);
	}

	template <typename xf_t, typename p_t, typename t0_t, typename tf_t,
	          typename T = typename xf_t::Scalar> // T is scalar type
	T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
	                  const Eigen::MatrixBase<p_t>& p,
	                  const Eigen::MatrixBase<t0_t>& t0,
	                  const Eigen::MatrixBase<tf_t>& tf)
	{
		return 10 * get_L_x<T>(xf);
	}

	template <typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
			  typename T = typename x_t::Scalar> // T is scalar type
	state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
							 const Eigen::MatrixBase<u_t>& u,
							 const Eigen::MatrixBase<p_t>& p,
							 const Eigen::MatrixBase<t0_t>& t0,
							 const Eigen::MatrixBase<tf_t>& tf,
							 const tau_t& tau)
	{
		const double g = 9.81, l = 0.5, m = 0.15, b = 0.1;

		// Setup states and controls
		T theta = x(0); // Angle
		T theta_dot = x(1); // Angular velocity
		T torque = u(0); // Torque

		// Dynamics
		state_t<T> x_dot;
		x_dot << theta_dot,
				 (m * g * l * sin(theta) - b * theta_dot + torque) / (m * l * l);
		return x_dot;
	}

	// const double h_discr = 0.1;
	// template <typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
	// 	      typename T = typename x_t::Scalar> // T is scalar type
	// state_t<T> discrete_dynamics_impl(const Eigen::MatrixBase<x_t>& x,
	// 								  const Eigen::MatrixBase<u_t>& u,
	// 								  const Eigen::MatrixBase<p_t>& p,
	// 								  const Eigen::MatrixBase<t0_t>& t0,
	// 								  const Eigen::MatrixBase<tf_t>& tf,
	// 								  const tau_t& tau)
	// {
	// 	const double g = 9.81, l = 0.5, m = 0.15, b = 0.1;
	//
	// 	// Setup states and controls
	// 	T theta = x(0); // Angle
	// 	T theta_dot = x(1); // Angular velocity
	// 	T torque = u(0); // Torque
	//
	// 	// Dynamics
	// 	state_t<T> x_dot;
	// 	x_dot << theta_dot,
	// 			 (m * g * l * sin(theta) - b * theta_dot + torque) / (m * l * l);
	//
	// 	// Discretize dynamics
	// 	state_t<T> x_next;
	// 	x_next = x + h_discr * x_dot;
	// 	return x_next;
	// }

	// ---- Helpers ----
	template <typename T>
	T get_L_x(const Eigen::Ref<const state_t<T>>& x)
	{
		T angle_err = angle_ref_ - x(0);
		return 10 * angle_err * angle_err;
	}

	template <typename T>
	T get_L_u(const Eigen::Ref<const input_t<T>>& u)
	{
		return u.dot(R_ * u);
	}

	// template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
	// 	     typename T = typename x_t::Scalar> // T is scalar type
	// ineq_constr_t<T> inequality_constraints_impl(const Eigen::MatrixBase<x_t>& x,
	// 											 const Eigen::MatrixBase<u_t>& u,
	// 											 const Eigen::MatrixBase<p_t>& p,
	// 											 const Eigen::MatrixBase<t0_t>& t0,
	// 											 const Eigen::MatrixBase<tf_t>& tf,
	// 											 const tau_t& tau)
	// {
	// 	ineq_constr_t<T> ineq_constr;
	// 	ineq_constr(0) = (-2.789 - u(0));
	// 	return ineq_constr;
	// }
};

#endif //LAOPT_INVERTED_PENDULUM_SIMPLE_OCP_HPP
