#define EIGEN_NO_MALLOC

#include "lampc.hpp"

using namespace std::placeholders;

template <typename Scalar, typename Traits>
struct MyNLP : public NLP< MyNLP<Scalar, Traits> >
{
    VEC(Scalar, 10) x0; // Initial state

    MAT(Scalar, 10, 15) M;
    VEC(Scalar, 12) v;

    MyNLP()
    {
        x0.setZero();
        M.setConstant(4.5);
        v.setConstant(12.3);
    }

    // template <typename Derived>
    // void testBlock(Eigen::MatrixBase<Derived>& m)
    // {
    //     cout << "CALLING THIS HELPER\n";
    //     testBlock(m.BLK(Derived::RowsAtCompileTime,Derived::ColsAtCompileTime,0,0));}

    // template <typename Derived>
    // void testBlock(Eigen::MatrixBase<Derived>&& m)
    // template <typename Scalar>
    void testBlock(Ref<Matrix<Scalar, 3, 4>> m, const Ref<Matrix<Scalar, 3, 1>> v)
    {
        // static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 4);
        cout << "testBlock" << endl;
        cout << "----- m -----\n";
        cout << m << endl;
        m(0,0) = 1;
        m(1,0) = 2;
        m(2,0) = 3;
        cout << "----- m -----\n";
        cout << m << endl;

        cout << "RowsAtCompileTime = " << m.RowsAtCompileTime << endl;
        cout << "ColsAtCompileTime = " << m.ColsAtCompileTime << endl;

        cout << "m.InnerStrideAtCompileTime = " << m.InnerStrideAtCompileTime << endl;
        cout << "m.innerStride = " << m.innerStride() << endl;
        cout << "m.OuterStrideAtCompileTime = " << m.OuterStrideAtCompileTime << endl;
        cout << "m.outerStride = " << m.outerStride() << endl;

        cout << "v = " << v.transpose() << endl;
        // m.transpose().setConstant(1.098);
    }

    void test()
    {
        cout << Base::x.transpose() << endl;
        cout << "X(1) = " << X(1).transpose() << endl;
        cout << "xss = " << xss().transpose() << endl;

        // cout << x0.transpose() << endl;
        // cout << "Should NOT call helper\n";
        // testBlock(x0.SEG(3,4));
        // cout << x0.transpose() << endl;

        // cout << "Should call helper\n";
        // testBlock(x0);
        // cout << x0.transpose() << endl;
        
        cout << "============== M ============\n"; 
        cout << M << endl;

        auto q = M.BLK(3,4,4,5);
        cout << "Should call helper\n";
        testBlock(q, v.template segment<3>(2));
        cout << M << endl;

        cout << "Should NOT call helper\n";
        testBlock(M.BLK(3,4,4,5), v.template segment<3>(5));
        cout << M << endl;


        cout << "========== J ==========\n";
        cout << Base::J << endl;
        cout << "=======================\n";
        cout << "Calling Jacobian" << endl;
        Base::x.setConstant(1.2);
        // Base::x_d = Base::x;

        eval_jacobian();

        cout << "========== J ==========\n";
        cout << Base::J << endl;
        cout << "=======================\n";

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
    
    i = Index(range(1, N))
    nlp.inequality("out_bnd", 3, (X[i], U[i]), index=i, lb=0, ub="ub_func")

    nlp.generate()
    ]]]*/
    using Base = NLP< MyNLP<Scalar, Traits> >;
    constexpr auto X(int col) {return Base::x.template segment<2>(0 + 2 * col);};
    constexpr auto U(int col) {return Base::x.template segment<1>(10 + 1 * col);};
    constexpr auto xss() {return Base::x.template segment<2>(14);};
    constexpr auto uss() {return Base::x.template segment<1>(16);};
    //[[[end]]]

    inline void eval_jacobian()
    {
        J_dynamics(Base::x.SEG(2,0), Base::x.SEG(2,2), Base::x.SEG(1,4),
            Base::g.SEG(2,0), 
            Base::J.BLK(2,2,0,0), Base::J.BLK(2,2,0,2), Base::J.BLK(2,1,0,4));
    }

    /*[[[cog nlp.gen_func("dynamics", ("xp", "x", "u")) ]]]*/
    inline void J_dynamics(RVEC(Scalar, 2) xp,
                           RVEC(Scalar, 2) x,
                           RVEC(Scalar, 1) u,
                           RVEC(Scalar, 2) val,
                           RVEC(Scalar, 2, 2) J_xp,
                           RVEC(Scalar, 2, 2) J_x,
                           RVEC(Scalar, 2, 1) J_u)
    {
    	using input_t = Matrix<Scalar, 5, 1>;
    	using ADScalar = AutoDiffScalar<input_t>;
    	VEC(ADScalar, 5) _x;
    	VEC(ADScalar, 2) _out;
    	// Copy current value into dual variables
    	_x.SEG(2,0) = xp;
    	_x.SEG(2,2) = x;
    	_x.SEG(1,4) = u;
    	// Compute the Jacobian
    	AD_seed(_x);
    	this->dynamics<ADScalar>(_x.SEG(2,0), _x.SEG(2,2), _x.SEG(1,4), _out);
    	val = _out.value();
    	for(int i=0; i<2; i++) // Copy Jacobian into output variables
    	{
    		Ref<input_t> deriv = _out[i].derivatives().transpose();
    		J_xp.row(i) = deriv.SEG(2,0);
    		J_x.row(i) = deriv.SEG(2,2);
    		J_u.row(i) = deriv.SEG(1,4);
    	}
    };

    template <typename T>
    inline void dynamics(RVEC(T, 2) xp, RVEC(T, 2) x, RVEC(T, 1) u, RVEC(T, 2) out)
    //[[[end]]]
    {
        out(0) = xp(0) - (-sin(x(1)) + x(1)*x(0));
        out(1) = xp(1) - cos(x(0))*u(0);
    };
};


/*[[[cog
nlp.generate_traits()
]]]*/
struct MyTraits
{
    enum {
        num_vars = 17,
        num_eq = 26
    };
};
//[[[end]]]

MyNLP<double, MyTraits> nlp;



// template<int nvars, int nout, typename Scalar, typename Functor>
// void jacobian(const Functor f, 
//               const Matrix<Scalar, nvars, 1> x, 
//               Matrix<Scalar, nout, nvars>& J) // Accepts only block
// {
//     using ADScalar = Eigen::AutoDiffScalar<Matrix<double, nvars, 1>>;
//     using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

//     // AD variables
//     Matrix<ADScalar, nvars, 1> _x;
//     Matrix<ADScalar, nout, 1> _out;

//     f(_x, _out);

//     for(int i=0; i<_out.rows(); i++)
//     {
//         Eigen::Ref<MatX> deriv = _out[i].derivatives().transpose();
//         J.row(i) = deriv;
//     }
// }



// template <short N_IN, short N_OUT>
// struct VectorFunction {
//     using InputType = Eigen::Matrix<double, N_IN, 1>;
//     using ValueType = Eigen::Matrix<double, N_OUT, 1>;
   
//     // Vector function
//     template <typename T>
//     void operator()
//     (
//       const Eigen::Matrix<T, N_IN, 1>& vIn, 
//       Eigen::Matrix<T, N_OUT, 1>* vOut
//     ) const
//     {
//       vOut->operator()(0) = vIn(0);
//       vOut->operator()(1) = 2*vIn(1) + 5*vIn(0);
//       vOut->operator()(2) = 3*vIn(2);
//     }
// };

// template <short N_IN, short N_OUT>
// struct TestFunction {
//     using InputType = Eigen::Matrix<double, N_IN, 1>;
//     using ValueType = Eigen::Matrix<double, N_OUT, 1>;
   
//     // Vector function
//     template <typename T>
//     void operator()
//     (
//       const Eigen::Matrix<T, N_IN, 1>& vIn, 
//       Eigen::Matrix<T, N_OUT, 1>* vOut
//     ) const
//     {
//       vOut->operator()(0) = vIn(0);
//       vOut->operator()(1) = 2*vIn(1) + 5*vIn(0);
//       vOut->operator()(2) = 3*vIn(2);
//     }
// };


// template <typename Scalar>
// void testFunc(const VEC(Scalar, 2)& x, const VEC(Scalar, 3)& y, VEC(Scalar, 3)& out)
// {
//     out(0) = x(1) * y(2) + 3 * x(0);
//     out(1) = 4 * y(1) + x(0) * 17;
//     out(2) = x(0) * x(1) * y(0) * y(1) * y(2);
// }



int main()
{
    cout << "Hello world\n";
    nlp.test();

//     {    
//     Eigen::Matrix<double, 3, 1> vIn;
//     Eigen::Matrix<double, 3, 1> vOut;
//     Eigen::Matrix<double, 3, 3> mJacobian;

//     Eigen::AutoDiffJacobian< VectorFunction<3, 3> > vectorFunAD;

//     for(int i=0; i<3; i++)
//         vIn(i) = i;

//     vectorFunAD(vIn, &vOut, &mJacobian); // Voila! jacobian is in mJacobian.

//     // vectorFunAD(x1, x2, x3, &vOut, &J_x1, &J_x2, &J_x3); // Voila! jacobian is in mJacobian.

//     cout << "vIn = " << vIn.transpose() << endl;
//     cout << "vOut = " << vOut.transpose() << endl;
//     cout << "mJacobian = \n" << mJacobian << endl;
//     };

//     {
//     cout << "\n\n\n";
//     Eigen::Matrix<double, 2, 1> x;
//     Eigen::Matrix<double, 3, 1> y;

//     x.setConstant(1);
//     y.setConstant(4);
    
//     Eigen::Matrix<double, 3, 1> out;

//     testFunc(x, y, out);
//     cout << "out = " << out.transpose() << endl;

//     using ADScalarX = Eigen::AutoDiffScalar<Matrix<double, 2, 1>>;
//     using ad_var_x_t = Eigen::Matrix<ADScalarX, 2, 1>;
//     using ad_var_y_t = Eigen::Matrix<ADScalarX, 3, 1>;

//     Matrix<ADScalarX, 3, 1> out_x;
//     // using ad_var_y_t = Eigen::Matrix<Eigen::AutoDiffScalar<Matrix<double, 3, 1>>, 3, 1>;

//     ad_var_x_t _x = x;
//     ad_var_y_t _y = y;
    
//     AD_seed(_x);
//     AD_clear(_y);

//     testFunc(_x, _y, out_x);

//     cout << "out_x = ";
//     for(int i=0; i<3; i++)
//     {
//         cout << out_x[i].value() << ", ";
//     }
//     cout << endl;
//     cout << "derivatives = \n";
//     for(int i=0; i<3; i++)
//     {
//         cout << out_x[i].derivatives().transpose() << "\n";
//     }
//     cout << endl;

//     // for (int i = 0; i < ad_eq.rows(); i++) {
//     //     b_eq[i] = ad_eq[i].value();
//     //     Eigen::Ref<MatX> deriv = ad_eq[i].derivatives().transpose();
//     //     A_eq.row(i) = deriv;
//     // }

//     }

//     cout << "\n\n\n";
//     Eigen::Matrix<double, 2, 1> x;
//     Eigen::Matrix<double, 3, 1> y;

//     x.setConstant(1);
//     y.setConstant(4);
    
//     // Eigen::Matrix<double, 3, 1> out;

//    Eigen::Matrix<double, 3, 2> J_x;
//    Eigen::Matrix<double, 3, 3> J_y;

// // template<int nvars, int nout, typename Scalar, typename Functor>
// // void jacobian(const Functor& f, 
// //                 const Matrix<Scalar, nvars, 1> x, 
// //                 Matrix<Scalar, nout, nvars>& J)

//     testFunc<double>(x, y, J);


//     auto f = [y](const Matrix<double, 2, 1> x, Matrix<double, 3, 1>& J)
//             {testFunc<double>(x, y, J);};

//     jacobian(f, x, J_x);

    return 0;

}