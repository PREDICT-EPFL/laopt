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
	template<typename ODE, typename scalar_t>
	struct RK4 : public MakeDifferentiable<RK4<ODE,scalar_t>>
	{
	  ODE& ode; // Reference to the callable ode objectp
	  scalar_t h; // Step size

	  RK4(ODE& ode, scalar_t h) : ode(ode), h(h) {}

	  template<typename X, typename... PARAMS>
	  EIGEN_STRONG_INLINE Eigen::Vector<typename Eigen::MatrixBase<X>::Scalar, Eigen::MatrixBase<X>::RowsAtCompileTime>
	  impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<PARAMS>&... params) noexcept
	  {
	    // diff_t _h = static_cast<diff_t>(h);
	    auto k1 = ode.impl(x,          params...);
	    auto k2 = ode.impl(x+h*0.5*k1, params...);
	    auto k3 = ode.impl(x+h*0.5*k2, params...);
	    auto k4 = ode.impl(x+h*k3,     params...);
	    return x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
	  }
	};

	/**
	 * Identity function
	 */
	struct id : public MakeDifferentiable<id>
	{
		using MakeDifferentiable<id>::operator();

		// Used to determine output size at compile time
    template<typename X>
    EIGEN_STRONG_INLINE X
    impl( const Eigen::MatrixBase<X>& x) noexcept        
    {
      return NULL;
    }

    template<typename OutValue, typename X>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Eval,
    					 OutValue&& outvalue,
							 const Eigen::MatrixBase<X>& x) noexcept
    {
        outvalue = x;
    }

    template<typename OutValue, typename OutJacobian, typename X>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Jacobian,
    					 OutValue&& outvalue, OutJacobian&& outjacobian,
							 const Eigen::MatrixBase<X>& x) noexcept
    {
    	using scalar_t = typename Eigen::MatrixBase<X>::Scalar;
    	outvalue = x;
			for(int i=0; i<outvalue.rows(); i++)
				outjacobian(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(1);
    }

    template<typename OutValue, typename OutJacobian, typename OutHessian, size_t num_outputs, typename X>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Hessian,
    					 OutValue&& outvalue, OutJacobian&& outjacobian, std::array<OutHessian,num_outputs>&& outhessian,
							 const Eigen::MatrixBase<X>& x) noexcept
    {
    	operator()(x, outvalue, outjacobian);
      for(int i=0; i<num_outputs; i++)
      	outhessian[i] = 0;
    }

	};

	/**
	 * An equation of the form g(x,args...) = - x + f(args...)
	 * 
	 * We assume that x is not in args
	 */
	template<typename F>
	struct eq : public MakeDifferentiable<eq<F>>
	{
		using MakeDifferentiable<eq<F>>::operator();

		// We assume F is a callable with tags for Eval, Jacobian and Hessian
		F& f;

    eq(F& f) : f(f) {}

		// Used to determine output size at compile time
    template<typename X, typename... Args>
    EIGEN_STRONG_INLINE X
    impl(const Eigen::MatrixBase<X>& x,
         const Eigen::MatrixBase<Args>&... args) noexcept
    {
    	return NULL;
    }

    template<typename OutValue, typename X, typename... Args>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Eval, 
               OutValue&& outvalue, 
               const Eigen::MatrixBase<X>& x,
               const Eigen::MatrixBase<Args>&... args) noexcept
    {
    	f(lampc::Eval(), outvalue);
    	outvalue -= x;
    }

    template<typename OutValue, typename OutJacobian, typename X, typename... Args>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Jacobian,
               OutValue&& outvalue, OutJacobian&& outjacobian,
               const Eigen::MatrixBase<X>& x,
               const Eigen::MatrixBase<Args>&... args) noexcept
    {
    	// Jacobian is [-I jac_f]

			// Set jac_f part
			f(lampc::Jacobian(),
				outvalue, outjacobian(all,seqN(outvalue.rows(),fix<meta::sum_template<Args::RowsAtCompileTime...>()>)),
				args...);

			// Add the -x part
			outvalue -= x;
			using scalar_t = typename Eigen::MatrixBase<X>::Scalar;
			// constexpr int num_outputs = X::RowsAtCompileTime;
			// outjacobian(all,seqN(0,fix<num_outputs>)) = -Eigen::Matrix<scalar_t,num_outputs,num_outputs>::Identity();
			for(int i=0; i<outvalue.rows(); i++)
				outjacobian(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(-1);
    }

   //  template<typename OutValue, typename OutJacobian, typename OutHessian, size_t len, typename X, typename... Args>
   //  EIGEN_STRONG_INLINE void
   //  operator()(
   //  	lampc::Hessian,
			// OutValue&& outvalue, OutJacobian&& outjacobian, std::array<OutHessian, len>&& outhessian,
			// const Eigen::MatrixBase<X>& x,
			// const Eigen::MatrixBase<Args>&... args) noexcept
   //  {
   //  	// Hessian is hessian of f
   //  	// Jacobian is [-I jac_f]

			// // Set jac_f part
			// f(args..., outvalue, outjacobian(all,seqN(num_outputs,fix<F::num_inputs>)), outhessian);

			// // Add the -x part
			// outvalue -= x;

			// // Dense version
			// // outjacobian(all,seqN(0,fix<num_outputs>)) = -Eigen::Matrix<scalar_t,num_outputs,num_outputs>::Identity();

			// // Sparse version
			// for(int i=0; i<num_outputs; i++)
			// 	outjacobian(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(-1);
   //  }


	};

};
};

#endif