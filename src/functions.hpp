#ifndef __LAMPC__FUNCTIONS_HPP
#define __LAMPC__FUNCTIONS_HPP

/**
 * Provides a class of common functions for defining OCPs
 */

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

namespace lampc {
	namespace functions {
	/**
	 * Simple general RK4 integrator
	 */
	template<typename ODE, typename scalar_t, size_t num_states, size_t... param_sizes>
	struct RK4 : public MakeDifferentiable<RK4<ODE,scalar_t,num_states,param_sizes...>, scalar_t, num_states, num_states, param_sizes...>
	{
	  ODE& ode; // Reference to the callable ode objectp
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

	/**
	 * Identity function
	 */
	template<typename scalar_t, size_t n>
	struct id : public MakeDifferentiable<id<scalar_t,n>, scalar_t, n,n>
	{
		using MakeDifferentiable<id<scalar_t,n>, scalar_t, n, n>::operator();

    static constexpr size_t num_inputs = n;
		static constexpr size_t num_outputs = n;

    template<typename OutValue>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref< const Eigen::Vector<scalar_t, n> >& x,
				     	 OutValue&& outvalue) noexcept
    {
        outvalue = x;
    }

    template<typename OutValue, typename OutJacobian>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref< const Eigen::Vector<scalar_t, n> >& x,
		    		   OutValue&& outvalue, OutJacobian&& outjacobian) noexcept

    {
    	outvalue = x;
			for(int i=0; i<n; i++)
				outjacobian(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(1);
    }

    template<typename OutValue, typename OutJacobian, typename OutHessianArray>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>>& x,
    		   OutValue&& outvalue, OutJacobian&& outjacobian, OutHessianArray&& outhessian) noexcept
    {
    	operator()(x, outvalue, outjacobian);
      for(int i=0; i<num_outputs; i++)
      	outhessian[i] = 0;
    }

	};

	/**
	 * An equation of the form g(x,args...) = f(args...) - x
	 * 
	 * We assume that x is not in args
	 */
	template<typename F, size_t num_outputs_, size_t... input_sizes>
	struct eq : public MakeDifferentiable<eq<F,num_outputs_,input_sizes...>, typename F::scalar_t, num_outputs_,num_outputs_,input_sizes...>
	{
		using MakeDifferentiable<eq<F,num_outputs_,input_sizes...>, typename F::scalar_t, num_outputs_,num_outputs_,input_sizes...>::operator();

		// We assume F is a callable with tags for Eval, Jacobian and Hessian
		F& f;

		static constexpr size_t num_outputs = num_outputs_;

		using scalar_t = typename F::scalar_t;
		static constexpr size_t num_inputs = lampc::meta::sum_template<input_sizes...>();  // Total number of inputs
   
    eq(F& f) : f(f) {}

    template<typename OutValue>
    EIGEN_STRONG_INLINE void
    operator()(
    	const Eigen::Ref< const Eigen::Vector<scalar_t, num_outputs> >& x,
    	const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args,
    	OutValue&& outvalue) noexcept
    {
        outvalue = f(lampc::Eval(), args...) - x;
    }

    template<typename OutValue, typename OutJacobian>
    EIGEN_STRONG_INLINE void
    operator()(
    	const Eigen::Ref< const Eigen::Vector<scalar_t, num_outputs> >& x,
    	const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args,
			OutValue&& outvalue, OutJacobian&& outjacobian) noexcept
    {
    	// Jacobian is [-I jac_f]

			// Set jac_f part
			f(args..., outvalue, outjacobian(all,seqN(num_outputs,fix<num_inputs>)));

			// Add the -x part
			outvalue -= x;
			// outjacobian(all,seqN(0,fix<num_outputs>)) = -Eigen::Matrix<scalar_t,num_outputs,num_outputs>::Identity();
			for(int i=0; i<num_outputs; i++)
				outjacobian(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(-1);
    }

    template<typename OutValue, typename OutJacobian, typename OutHessianArray>
    EIGEN_STRONG_INLINE void
    operator()(
    	const Eigen::Ref< const Eigen::Vector<scalar_t, num_outputs> >& x,
    	const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args,
      OutValue&& outvalue, OutJacobian&& outjacobian, OutHessianArray&& outhessian) noexcept
    {
    	// Hessian is hessian of f
    	// Jacobian is [-I jac_f]

			// Set jac_f part
			f(args..., outvalue, outjacobian(all,seqN(num_outputs,fix<num_inputs>)), outhessian);

			// Add the -x part
			outvalue -= x;
			// outjacobian(all,seqN(0,fix<num_outputs>)) = -Eigen::Matrix<scalar_t,num_outputs,num_outputs>::Identity();
			for(int i=0; i<num_outputs; i++)
				outjacobian(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(-1);
    }


	};

};
};

#endif