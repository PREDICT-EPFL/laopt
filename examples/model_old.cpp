#define EIGEN_NO_MALLOC

// #include "lampc.hpp"

/*[[[cog 
	from lampc import *
	nlp = NLP("MyNLP", "class Scalar")

	N = nlp.const("N", 5) # Prediction horizon
	n = nlp.const("n", 2) # State dimension
	m = nlp.const("m", 1) # Input dimension

	X   = nlp.var("X",   n, N)
	U   = nlp.var("U",   m, N-1)
	xss = nlp.var("xss", n)
	uss = nlp.var("uss", m)

	# Declare function signatures
	sys = nlp.function("sys", n, ("xp", n), ("x", n), ("u", m))

    i = Index(range(1, N-1))
    sys0(X[0], U[0]) == 0
    sys(X[i+1], X[i], U[i+1]) == 0

    sys(xss, xss, uss) == 0
    equal(X[N-1], xss) == 0

    # i = Index(range(1, N))
    # lb <= out_bnd(X[i], U[i]) <= "ub_func(i)"
]]]*/

//[[[end]]]

POLYMPC_FORWARD_NLP_DECLARATION(MyNLP, /*NX*/ 2, /*NE*/1, /*NI*/0, /*NP*/0, /*Type*/double);
template<class Impl>
class MyNLP : public ProblemBase<MyNLP, Impl>
{
	// Declare user constants
	constexpr int N = 5;
	constexpr int n = 2;
	constexpr int m = 1;

	// Declare user types
    template<typename T>
    using X_t = Eigen::Matrix<T, n, N>;
    template<typename T>
    using U_t = Eigen::Matrix<T, m, N-1>;
    template<typename T>
    using xss_t = Eigen::Matrix<T, n, 1>;
    template<typename T>
    using uss_t = Eigen::Matrix<T, m, 1>;

    // Declare variables
    #DECLARE_VAR(X, offset, n, N);

    template<typename T>
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {

    }
	
    // EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
    //                                         scalar_t &_cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept;

    // EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
    //                                                scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept;
}



// template <Derived>
// struct EqualityConstraint_diff<Derived>
// {
// 	// Generated functions to evaluate
// 	void operator()(RVec<Scalar,num_constraints> out, RCVec<Scalar,num_vars> x)
// 	{
// 		sys(out.block(blah blah), x.seg(asdf), x.seg(asdf));
// 		sys(out.block(blah blah), x.seg(asdf), x.seg(asdf));
// 		bob(out.block(blah blah), x.seg(asdf), x.seg(asdf));
// 	}

// 	void operator()(RVec<Scalar,num_constraints> out, RCVec<Scalar,num_vars> x, RCVec<Jacobian> J)
// 	{
//         m_ad_var = var;
//         sys<ad_scalar_t>(m_ad_var.seg, p, m_ad_eq...);
//         for (int ...)
//         {
//         	equalities(i) = m_ad_eq(i).value(i);
//         }

//         sys<ad_scalar_t>(m_ad_var.seg, p, m_ad_eq...);
//         bob<ad_scalar_t>(m_ad_var.seg, p, m_ad_eq...);        
// 	}
// }


struct EqualityConstraint : public EqualityConstraint_diff<EqualityConstraint>
{
	Vec<Scalar, 2> p;

	template<typename T>
	inline void sys(RVec<T,n> out, RCVec<T,n> xp, RCVec<T,n> x, RCVec<T,m> u)
	{
	    out(0) = p(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + p(0);
	    out(1) = p(2) * xp(1) - cos(x(0))*u(0) + p(1);
	}

    template<typename T>
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {

    }


}



int main()
{
    cout << "Hello world" << endl;

    param_t param;
    Vec<Scalar, n> out;
    Vec<Scalar, n> xp;
    Vec<Scalar, n> x;
    Vec<Scalar, m> u;

    xp << 1,2;
    x << 3,4;
    u << 5;

    _sys<Scalar, param_t>(param, out, xp, x, u);
    _sys<Scalar, param_t>(param, out, xp, param.x0, u);

    // sys(param, out, xp, x, u);

    cout << "x = " << x.transpose() << " xp = " << xp.transpose() << " u = " << u.transpose() << endl;
    cout << "out = " << out.transpose() << endl;

    Matrix<Scalar, n, n> J_xp;
    Matrix<Scalar, n, n> J_x;
    Matrix<Scalar, n, m> J_u;

    sys(param, out, xp, x, u, J_xp, J_x, J_u);

    cout << "J_xp = " << endl << J_xp << endl;
    cout << "J_x = " << endl << J_x << endl;
    cout << "J_u = " << endl << J_u << endl;

    EqualityConstraint<Scalar> 


    // // Final goal
    // // - number of vars
    // // - number of constraints
    // // - scalar type
    // ADMM<2, 1, Scalar> prob;

    // // Compute hessian and linear term
    // // Compute 'A' and upper lower bounds on A and x
    // prob.solve(H,h,A,al,au,xl,xu);
    // Eigen::Vector2f sol = prob.primal_solution();

    return 0;
}