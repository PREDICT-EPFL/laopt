#define EIGEN_NO_MALLOC

#include "lampc.hpp"

using namespace std::placeholders;

template <typename Scalar, typename Traits>
struct MyNLP : public NLP< MyNLP<Scalar, Traits> >
{
    VEC(Scalar, 2) x0; // Initial state

    MyNLP()
    {
        x0.setZero();
    }

    /*[[[cog
    from lampc import *
    nlp = NLP()

    # Simple linear MPC example
    N = 5
    X = nlp.var("X", 2, N, lb=(-5, -3), ub=10)
    U = nlp.var("U", 1, N-1, lb="lb_func")

    i = Index(range(N-1))
    nlp.equality("dynamics", 2, (X[i+1], X[i], U[i+1]), i)
    
    nlp.equality("initial_state", 2, (X[0], ))
    
    xss = nlp.var("xss", 2)
    uss = nlp.var("uss", 1)
    nlp.equality("dynamics", 2, (xss, xss, uss))
    nlp.equality("equal", 2, (X[N-1], xss))
    
    # i = Index(range(1, N))
    # nlp.inequality("out_bnd", 3, (X[i], U[i]), index=i, lb=0, ub="ub_func")

    nlp.generate()
    ]]]*/
    using Base = NLP< MyNLP<Scalar, Traits> >;
    using Base::x;
    using Base::J;
    using Base::g;
    constexpr auto X(int col) {return x.SEG(2, 0 + 2 * col);};
    constexpr auto U(int col) {return x.SEG(1, 10 + 1 * col);};
    constexpr auto xss() {return x.SEG(2, 14);};
    constexpr auto uss() {return x.SEG(1, 16);};
    //[[[end]]]

    /*[[[cog 
    nlp.constraints.gen_eval() 
    nlp.constraints.gen_jacobian() 
    ]]]*/
    inline void eval()
    {
    	for(int i=0; i<4; i++)
    		dynamics<Scalar>(X((i+1)), X(i), U((i+1)), g.SEG(2,0+i*2));
    	initial_state<Scalar>(X(0), g.SEG(2,2));
    	dynamics<Scalar>(xss(), xss(), uss(), g.SEG(2,4));
    	equal<Scalar>(X(4), xss(), g.SEG(2,6));
    }

    inline void eval_jacobian()
    {
    	for(int i=0; i<4; i++)
    		J_dynamics(x.SEG(2, ((i+1)*2+0)), x.SEG(2, (i*2+0)), x.SEG(1, ((i+1)*1+10)),
    		           g.SEG(2, 0+i*2),
    		           J.BLK(2,2,0+i*2,((i+1)*2+0)), J.BLK(2,2,0+i*2,(i*2+0)), J.BLK(2,1,0+i*2,((i+1)*1+10)));
    	J_initial_state(x.SEG(2, 0),
    	                g.SEG(2, 8),
    	                J.BLK(2,2,8,0));
    	J_dynamics(x.SEG(2, 14), x.SEG(2, 14), x.SEG(1, 16),
    	           g.SEG(2, 10),
    	           J.BLK(2,2,10,14), J.BLK(2,2,10,14), J.BLK(2,1,10,16));
    	J_equal(x.SEG(2, 8), x.SEG(2, 14),
    	        g.SEG(2, 12),
    	        J.BLK(2,2,12,8), J.BLK(2,2,12,14));
    }

    //[[[end]]]

    /*[[[cog nlp.gen_func("dynamics", ("xp", "x", "u")) ]]]*/
    inline void J_dynamics(const Ref<const Matrix<Scalar, 2, 1>> xp,
                           const Ref<const Matrix<Scalar, 2, 1>> x,
                           const Ref<const Matrix<Scalar, 1, 1>> u,
                           Ref<Matrix<Scalar, 2, 1>> val,
                           Ref<Matrix<Scalar, 2, 2>> J_xp,
                           Ref<Matrix<Scalar, 2, 2>> J_x,
                           Ref<Matrix<Scalar, 2, 1>> J_u)
    {
    	using input_t = Matrix<Scalar, 5, 1>;
    	using ADScalar = AutoDiffScalar<input_t>;
    	Matrix<ADScalar, 5, 1> _x;
    	Matrix<ADScalar, 2, 1> _out;
    	// Copy current value into dual variables
    	_x.SEG(2,0) = xp;
    	_x.SEG(2,2) = x;
    	_x.SEG(1,4) = u;
    	// Compute the Jacobian
    	AD_seed(_x);
    	this->dynamics<ADScalar>(_x.SEG(2,0), _x.SEG(2,2), _x.SEG(1,4), _out);
    	for(int i=0; i<2; i++) // Copy Jacobian into output variables
    	{
    		val(i) = _out[i].value();
    		Ref<input_t> deriv = _out[i].derivatives();
    		J_xp.row(i) = deriv.SEG(2,0);
    		J_x.row(i) = deriv.SEG(2,2);
    		J_u.row(i) = deriv.SEG(1,4);
    	}
    };

    template <typename T>
    inline void dynamics_contiguous(const Ref<const Matrix<T, 5, 1>> _x,
                                       Ref<Matrix<T, 2, 1>> out)
    {dynamics<T>(_x.SEG(2,0), _x.SEG(2,2), _x.SEG(1,4), out);};

    template <typename T, typename N>
    inline void dynamics(const Ref<const Matrix<T, 2, 1>> xp,
                         const Ref<const Matrix<T, 2, 1>> x,
                         const Ref<const Matrix<T, 1, 1>> u,
                         Ref<Matrix<T, 2, 1>> out)
    //[[[end]]]
    {
        out(0) = x0(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + x0(0);
        out(1) = x0(0) * xp(1) - cos(x(0))*u(0) + x0(1);
    };

    /*[[[cog nlp.gen_func("equal", ("a", "b")) ]]]*/
    inline void J_equal(const Ref<const Matrix<Scalar, 2, 1>> a,
                        const Ref<const Matrix<Scalar, 2, 1>> b,
                        Ref<Matrix<Scalar, 2, 1>> val,
                        Ref<Matrix<Scalar, 2, 2>> J_a,
                        Ref<Matrix<Scalar, 2, 2>> J_b)
    {
    	using input_t = Matrix<Scalar, 4, 1>;
    	using ADScalar = AutoDiffScalar<input_t>;
    	Matrix<ADScalar, 4, 1> _x;
    	Matrix<ADScalar, 2, 1> _out;
    	// Copy current value into dual variables
    	_x.SEG(2,0) = a;
    	_x.SEG(2,2) = b;
    	// Compute the Jacobian
    	AD_seed(_x);
    	this->equal<ADScalar>(_x.SEG(2,0), _x.SEG(2,2), _out);
    	for(int i=0; i<2; i++) // Copy Jacobian into output variables
    	{
    		val(i) = _out[i].value();
    		Ref<input_t> deriv = _out[i].derivatives();
    		J_a.row(i) = deriv.SEG(2,0);
    		J_b.row(i) = deriv.SEG(2,2);
    	}
    };

    template <typename T>
    inline void equal(const Ref<const Matrix<T, 2, 1>> a,
                      const Ref<const Matrix<T, 2, 1>> b,
                      Ref<Matrix<T, 2, 1>> out)
    //[[[end]]]
    {
        out = a - b;
    };

    /*[[[cog nlp.gen_func("initial_state", ("x")) ]]]*/
    inline void J_initial_state(const Ref<const Matrix<Scalar, 2, 1>> x,
                                Ref<Matrix<Scalar, 2, 1>> val,
                                Ref<Matrix<Scalar, 2, 2>> J_x)
    {
    	using input_t = Matrix<Scalar, 2, 1>;
    	using ADScalar = AutoDiffScalar<input_t>;
    	Matrix<ADScalar, 2, 1> _x;
    	Matrix<ADScalar, 2, 1> _out;
    	// Copy current value into dual variables
    	_x.SEG(2,0) = x;
    	// Compute the Jacobian
    	AD_seed(_x);
    	this->initial_state<ADScalar>(_x.SEG(2,0), _out);
    	for(int i=0; i<2; i++) // Copy Jacobian into output variables
    	{
    		val(i) = _out[i].value();
    		Ref<input_t> deriv = _out[i].derivatives();
    		J_x.row(i) = deriv.SEG(2,0);
    	}
    };

    template <typename T>
    inline void initial_state(const Ref<const Matrix<T, 2, 1>> x,
                              Ref<Matrix<T, 2, 1>> out)
    //[[[end]]]
    {
        out = x - x0;
    };
};


/*[[[cog
nlp.generate_traits()
]]]*/
struct MyTraits
{
    enum {
        num_vars = 17,
        num_eq = 14
    };
};
//[[[end]]]

MyNLP<double, MyTraits> nlp;

template<typename _Scalar, typename _param_t, int _NumOutputs, int... n>
struct ADTraits_t
{
    using Scalar = _Scalar;
    static constexpr int input_len()
    {
        int NumInputs = 0;
        (void)initializer_list<int>{ (NumInputs += n, 0)... };
        return NumInputs;
    };

    enum 
    {
        NumInputs = input_len(),
        NumOutputs = _NumOutputs
    };

    using contiguous_input_t = Matrix<Scalar, NumInputs, 1>;
    using ADScalar = AutoDiffScalar<contiguous_input_t>;

    // Tuples of input and jacobian blocks
    using TplInputs = tuple<const Ref<const Matrix<Scalar, n, 1>>...>;
    using TplJacobians = tuple<Ref<Matrix<Scalar, NumOutputs, n>>...>;

    using AD_input_t = tuple<Matrix<ADScalar, n, 1>...>;
    static constexpr auto input_sizes = make_tuple(n...);
    using AD_output_t = Matrix<ADScalar, NumOutputs, 1>;

    using param_t = _param_t;
};

template<typename ADTraits, typename ADFunctor>
struct Jacobian
{
    using ADScalar = typename ADTraits::ADScalar;
    using param_t = typename ADTraits::param_t;
    using Scalar = typename ADTraits::Scalar;
    using AD_input_t = typename ADTraits::AD_input_t;
    using AD_output_t = typename ADTraits::AD_output_t;
    using contiguous_input_t = typename ADTraits::contiguous_input_t;

    enum {
        NumInputs = ADTraits::NumInputs,
        NumOutputs = ADTraits::NumOutputs
    };

    using TplInputs = typename ADTraits::TplInputs;
    using TplJacobians = typename ADTraits::TplJacobians;

    param_t& _param;
    ADFunctor f;
    AD_input_t _x;    // Tuple of dual variables as inputs
    AD_output_t _out; // Vector of dual variables as outputs


    Jacobian(param_t& param) : _param(param) {};

    template<int... n>
    void operator()(Ref<Matrix<Scalar,NumOutputs,1>> val, 
                  const Ref<const Matrix<Scalar, n, 1>>... inputs,
                  Ref<Matrix<Scalar, NumOutputs, n>>... J)
    {
        jacobian_impl(forward<decltype(val)>(val), 
                      forward_as_tuple(inputs...), 
                      forward_as_tuple(J...),
                      std::make_index_sequence<sizeof...(n)>{});
    }

    template <typename vec>
    constexpr int AD_Seed(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++) {
            x[i].derivatives().coeffRef(i + offset) = 1;
        }
        return offset + x.rows();
    }

    template<std::size_t... Is>
    void jacobian_impl(Ref<Matrix<Scalar,NumOutputs,1>> val, 
                  TplInputs inputs,
                  TplJacobians J,
                  std::index_sequence<Is...> ind
                  )
    {
        // Copy the current value of the inputs into the AD variable
        // and set derivative equal to identity
        int offset = 0;
        (void)initializer_list<int>{ 
            (
                get<Is>(_x) = get<Is>(inputs),          // Copy inputs
                offset = AD_Seed(get<Is>(_x), offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        f(std::get<Is>(_x)..., _out, _param);

        // Copy Jacobian into output variables
        for(int i=0; i<NumOutputs; i++)
        {
            val(i) = _out[i].value();
            Ref<contiguous_input_t> deriv = _out[i].derivatives();
            
            // Copy gradients to Jacobian matrices
            offset = 0;
            (void)initializer_list<int>{ 
                (
                    get<Is>(J).row(i) = deriv.template segment<get<Is>(ADTraits::input_sizes)>(offset), 
                    offset += get<Is>(ADTraits::input_sizes),
                    0
                )... 
            };
        }
    }

//     void operator()(const Ref<const Matrix<Scalar, n, 1>>... inputs,
//                     Ref<Matrix<Scalar,NumOutputs,1>> val)
//     {
//         f(inputs..., val);
//     }
};

template<typename param_t, typename _Scalar>
struct dynamics_t {
    void operator()(
        const Ref<const Matrix<_Scalar, 2, 1>> xp,
        const Ref<const Matrix<_Scalar, 2, 1>> x,
        const Ref<const Matrix<_Scalar, 1, 1>> u,
        Ref<Matrix<_Scalar, 2, 1>> out,
        param_t& param)
    {
        out(0) = param.x0(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + param.x0(0);
        out(1) = param.x0(0) * xp(1) - cos(x(0))*u(0) + param.x0(1) + 4.56*u(0);
    }
};


template<template <typename, typename> typename name_t, typename param_t, int NumOutputs, int... NumInputs>
// template<typename name_t, typename param_t, int NumOutputs, int... NumInputs>
using Jacobian_t = 
Jacobian<ADTraits_t<typename param_t::Scalar, param_t, NumOutputs, NumInputs...>, 
         name_t<param_t, 
                typename ADTraits_t<typename param_t::Scalar, param_t, NumOutputs, NumInputs...>::ADScalar>>;

Jacobian_t<dynamics_t, decltype(nlp), 2, 2, 2, 1> J_dynamics(nlp);


// template<template <typename, typename> class Functor, int NumOutputs, int... n>
// struct JJacobian_t
// {
//     Functor<int, double> f1;
//     Functor<double, int> f2;
//     void eval()
//     {
//         f1();
//         f2();
//     }
// };

// template<typename A, typename B>
// struct func_t
// {
//     void operator()()
//     {
//         cout << "Here!" << endl;
//     }
// };

// JJacobian_t<func_t, 2, 2, 2, 1> test_jac;

int main()
{
    cout << "Hello world\n";

    nlp.x.setRandom();

    Matrix<double, 2, 1> xp;
    Matrix<double, 2, 1> x;
    Matrix<double, 1, 1> u;

    Matrix<double, 2, 2> J_xp;
    Matrix<double, 2, 2> J_x;
    Matrix<double, 2, 1> J_u;

    Matrix<double, 2, 1> val;

    // auto dynamics = Dynamics<decltype(nlp), 2, 2, 2, 1>(nlp);
    // auto d = Jacobian<decltype(dynamics), decltype(nlp), 2, 2, 2, 1>(dynamics, nlp);

    xp.setConstant(1);
    x.setConstant(2);
    u.setConstant(3);
    nlp.x0.setConstant(4);
    J_xp.setConstant(-1);
    J_x.setConstant(-1);
    J_u.setConstant(-1);

    cout << "nlp.x0 = " << nlp.x0.transpose() << endl;
    // cout << "dynamics.nlp.x0 = " << dynamics.nlp.x0.transpose() << endl;

    // dynamics(xp, x.tail<2>(), u, val);
    cout << "val = " << val.transpose() << endl;

    J_dynamics.operator()<2,2,1>(val, xp, x, u, J_xp, J_x, J_u);   

    cout << "J_xp = \n" << J_xp << endl;
    cout << "J_x = \n" << J_x << endl;
    cout << "J_u= \n" << J_u << endl;


    cout << "*******************************\n";

    // test_jac.eval();


    return 0;

}