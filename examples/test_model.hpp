#include "polygen_helper.hpp"
#include <functional>

// Define traits class
template<typename Derived>
struct MyNLPBase;

template<typename Derived>
struct nlp_traits<MyNLPBase<Derived>>
{
	using scalar_t = double;
	enum { NX = 32, NE = 18, NI = 0, NP = 0};
};

template <typename Derived>
struct MyNLPBase : public ProblemBase<MyNLPBase<Derived>>
{
	using Base = ProblemBase<MyNLPBase<Derived>>;
	using typename Base::scalar_t;
	using typename Base::nlp_variable_t;
	using typename Base::nlp_constraints_t;
	using typename Base::nlp_eq_jacobian_t;
	
	/** problem dimensions */
	using Base::VAR_SIZE;
	using Base::NUM_EQ;
	using Base::NUM_INEQ;
	using Base::NUM_BOX;
	using Base::DUAL_SIZE;
	
	enum {
		N = 5,
		n = 3,
		m = 2,
	};
	
	DECLARE_VAR_TYPE(x_t, 3);
	DECLARE_VAR_TYPE(u_t, 2);
	DECLARE_VAR_TYPE(y_t, 1);
	
	DECLARE_VAR(x0, 0, x_t_size);
	DECLARE_VAR(x, 3, x_t_size, 5);
	DECLARE_VAR(u, 18, u_t_size, 4);
	DECLARE_VAR(xss, 26, x_t_size);
	DECLARE_VAR(uss, 29, u_t_size);
	DECLARE_VAR(ref, 31, y_t_size);

	DECLARE_CONSTRAINT(eq_steady_state, 0, 3, 1);
	DECLARE_CONSTRAINT(eq_dynamics, 3, 3, 4);
	DECLARE_CONSTRAINT(eq_testx, 15, 3, 1);

	DECLARE_FUNCTION(sys, 3, 3,3,2);

	MyNLPBase() : sys(this) {}


	EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, 
	                                    Eigen::Ref<nlp_constraints_t> _equalities) noexcept
	{
		sys(xss(var), xss(var), uss(var), eq_steady_state(_equalities, 0));
		for (int myIndex=0, _con_ind=0; myIndex<4; myIndex+=1, _con_ind++)
		sys(x(var, (myIndex+1)), x(var, myIndex), u(var, myIndex), eq_dynamics(_equalities, _con_ind));
		sys(x(var, 0), xss(var), u(var, 2), eq_testx(_equalities, 0));
	}


	EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
	                                               Eigen::Ref<nlp_constraints_t> equalities,
	                                               Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
	{
		sys(
			xss(var), xss(var), uss(var), // Inputs
			eq_steady_state(equalities, 0), // Output
			jacobian.template block<eq_steady_state_size, x_t_size>(eq_steady_state_offset(0), xss_offset()),
			jacobian.template block<eq_steady_state_size, x_t_size>(eq_steady_state_offset(0), xss_offset()),
			jacobian.template block<eq_steady_state_size, u_t_size>(eq_steady_state_offset(0), uss_offset()));

		sys(
			xss(var), xss(var), uss(var), // Inputs
			eq_steady_state(equalities, 0), // Output
			jacobian.template block<eq_steady_state_size, x_t_size>(eq_steady_state_offset(0), xss_offset()),
			jacobian.template block<eq_steady_state_size, x_t_size>(eq_steady_state_offset(0), xss_offset()),
			jacobian.template block<eq_steady_state_size, u_t_size>(eq_steady_state_offset(0), uss_offset()));
		
		for (int myIndex=0, _con_index=0; myIndex<4; myIndex+=1, _con_index++)
		sys(
			x(var, (myIndex+1)), x(var, myIndex), u(var, myIndex), // Inputs
			eq_dynamics(equalities, _con_index), // Output
			jacobian.template block<eq_dynamics_size, x_t_size>(eq_dynamics_offset(_con_index), x_offset((myIndex+1))),
			jacobian.template block<eq_dynamics_size, x_t_size>(eq_dynamics_offset(_con_index), x_offset(myIndex)),
			jacobian.template block<eq_dynamics_size, u_t_size>(eq_dynamics_offset(_con_index), u_offset(myIndex)));
		
		sys(
			x(var, 0), xss(var), u(var, 2), // Inputs
			eq_testx(equalities, 0), // Output
			jacobian.template block<eq_testx_size, x_t_size>(eq_testx_offset(0), x_offset(0)),
			jacobian.template block<eq_testx_size, x_t_size>(eq_testx_offset(0), xss_offset()),
			jacobian.template block<eq_testx_size, u_t_size>(eq_testx_offset(0), u_offset(2)));
		
	}



};
