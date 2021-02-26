#include <tuple>
#include <functional>
#include <math.h>
#include <iostream>
#include <string>
#include <type_traits>

#include <Eigen/Dense>
#include "unsupported/Eigen/AutoDiff"

#include "problemBase.hpp"
#include "test.hpp"

using namespace Eigen;

// User-written class
struct MyNLP : public MyNLPBase<MyNLP>
{
	nlp_variable_t bob;
	scalar_t y;

	template<typename T>
	void test(T x)
	{
		auto y = x*2;
		std::cout << "IN TEST x = " << x << std::endl;
	}

	template<typename T>
    EIGEN_STRONG_INLINE void sys(const Ref<const x_t<T>>& xp, 
    							 const Ref<const x_t<T>>& x, 
    							 const Ref<const u_t<T>>& u,
    							 Ref<x_t<T>> out) const noexcept
    {
    	out(0) = xp.sum() + 2*x.sum() + 3*u.sum();
    	out(1) = xp.sum() + 2*x.sum() + 3*u.sum();
    	out(2) = xp.sum() + 2*x.sum() + 3*u.sum();

    	// std::cout << "IN SYS" << std::endl;
    }

	template<typename T>
	EIGEN_STRONG_INLINE scalar_t stage_cost(const Eigen::Ref<const x_t<T>>& x,
								     		const Eigen::Ref<const u_t<T>>& u)
	{
		return x.sum() + 2*u.sum();
	}
};



template<typename Functor>
EIGEN_STRONG_INLINE double test_call(Functor functor)
{
	return functor();
}


// template < typename... T, typename... O,
//            typename = typename std::enable_if<sizeof...(T) == sizeof...(O)>::type>
// void make_same(T... t, O... o) 
// {
// 	std::cout << "IN HERE" << std::endl;
// }


template<typename... T, typename... L, typename O>
void test_arg_size(MatrixBase<O> const & _out, const MatrixBase<T>&... x, const MatrixBase<L>&... y)
{
	// auto out = const_cast< MatrixBase<O>& >(_out);

	std::cout << "In test_arg_size" << std::endl;
	std::cout << "Number of arguments T = " << sizeof...(T) << std::endl;
	std::cout << "Number of arguments L = " << sizeof...(L) << std::endl;

   //  // Copy gradients to Jacobian matrices
    int offset = 0;
    (void)std::initializer_list<int>{ 
        (
			std::cout << "x = " << x.transpose() << std::endl,
			// std::cout << "type_name<decltype(x)> = " << type_name<decltype(x)>() << std::endl,
			// std::cout << "type_name<T> = " << type_name<T>() << std::endl,
			std::cout << "n = " << T::RowsAtCompileTime << std::endl,
			const_cast< MatrixBase<O>& >(_out)[offset] = x[0],
			offset++,
            // offset += input_sizes,
            0
        )... 
    };
}

// template<typename scalar_t, size_t n, template <typename scalar_t, size_t n, size_t m> class T>
// template<size_t n> // , size_t n=T::RowsAtCompileTime()>
// void test_arg_size(const Eigen::Ref<const Eigen::Matrix<double, n, 1>>& x)
// {
// 	std::cout << "In test_arg_size" << std::endl;
// 	std::cout << "type_name<decltype(x)> = " << type_name<decltype(x)>() << std::endl;
// 	std::cout << "type_name<T> = " << type_name<T>() << std::endl;
// 	std::cout << "n = " << T::RowsAtCompileTime << std::endl;
// }



template<typename scalar_t>
void test_function(
		const Eigen::Ref<const Eigen::Matrix<scalar_t, 3, 1>>& xp,
		const Eigen::Ref<const Eigen::Matrix<scalar_t, 3, 1>>& x,
		const Eigen::Ref<const Eigen::Matrix<scalar_t, 2, 1>>& u,
		Eigen::Ref<Eigen::Matrix<scalar_t, 3, 1>> out
	)
{
	out(0) = xp.sum();
	out(1) = 2*x.sum();
	out(2) = 3*u.sum();

	std::cout << "I'VE BEEN CALLED" << std::endl;
}


template<typename... Args>
auto get_eigen_sizes(const Args... args)
{
	double out = 0;
    (void)std::initializer_list<int>{ 
        (
            // std::cout << "Getting sizes = " << decltype(args)::RowsAtCompileTime << std::endl,
            out += 2*args.sum(),
            // std::cout << "type_name<args>() = " << type_name<Args>() << std::endl,
            0
        )...
    };    
    return out;
}

auto test_best(
	const Eigen::Ref<const Eigen::Matrix<double, 5, 1>> a,
	const Eigen::Ref<const Eigen::Matrix<double, 3, 1>> b,
	const Eigen::Ref<const Eigen::Matrix<double, 4, 1>> c
	)
{
	return 2*a.sum() + 2*b.sum() + 2*c.sum();
}

auto test_best2(
	const Eigen::Matrix<double, 5, 1> a,
	const Eigen::Matrix<double, 3, 1> b,
	const Eigen::Matrix<double, 4, 1> c
	)
{
	return 2*a.sum() + 2*b.sum() + 2*c.sum();
}


template<int... S>
constexpr int sum_template() {
    int result = 0;
    for(auto s : { S... }) result += s;
    return result;
}


template<typename scalar_t, int m, int... n>
void testJac(
	// const Matrix<scalar_t, n, 1>&... in,
	Ref<Matrix<scalar_t, m, n>>... J,
	const Ref<const Matrix<scalar_t, n, 1>>&... in)
{
	std::cout << "In testJac" << std::endl;	
	std::cout << "sizeof...(n) = " << sizeof...(n) << std::endl;	
	std::cout << "num inputs = " << sum_template<n...>() << std::endl;

    (void)std::initializer_list<int>{ 
        (
			J.setOnes(),
			J.row(1) = in,			
            0
        )... 
    };
}

// template<typename... Args>
// void testJac2(Args&&... args)
// {
// 	testJac(std::forward<decltype(Ref<Args>(args))>(Ref<Args>(args))...);
// }

int main(void)
{

	Eigen::Matrix<double, 5, 1> a;
	Eigen::Matrix<double, 3, 1> b;
	Eigen::Matrix<double, 10, 1> c;

	a << 1,2,3,4,5;
	b << 1,2,3;
	c << 1,2,3,4,5,6,7,8,9,10;

	Eigen::Matrix<double, 3,5> Ja;
	Eigen::Matrix<double, 3,3> Jb;
	Eigen::Matrix<double, 5,20> Jc;

	Jc.setZero();
	std::cout << "Jc = \n" << Jc << std::endl;

	// testJac<double, 3, 5, 3, 10>(a,b,c,Ja,Jb,Jc);	
	testJac<double,3,5,3,10>(Ja,Jb,Jc.template block<3,10>(2,5),a.template segment<5>(0),b,3*c);

	std::cout << "c = " << c.transpose() << std::endl;
	std::cout << "Jc = \n" << Jc << std::endl;

	// volatile double out;
	// for(int i=0; i<1e6; i++)
	// 	// for(int j=0; j<1e4; j++)
	// 	for(int j=0; j<1e3; j++)
	// 		out = get_eigen_sizes(2*a+3.4*a,2*b,2*c.template segment<4>(2));
	// 		// out = test_best(2*a+3.4*a,2*b,2*c.template segment<4>(2));
	// std::cout << "out = " << out << std::endl;

	// auto d = 2*a.template segment<3>(0) + 0.34*b;
	// std::cout << "d = " << d.transpose() << std::endl;
	// std::cout << "type_name<decltype(d)>() = " << type_name<decltype(d)>() << std::endl;
	// test_arg_size(c, d, 2.23*d, (3.3537*d).template segment<2>(0));
	// std::cout << "d = " << d.transpose() << std::endl;
	// std::cout << "c = " << c.transpose() << std::endl;


	// std::cout << "\n\n\n\n\n";

return 0;

 //    // using Solver = SQPSolver<MyMPC>;
 //    // Solver solver;

    MyNLP mpc;
	using scalar_t = typename MyNLP::scalar_t;

//     for (int i=0; i<MyNLP::VAR_SIZE; i++) mpc.bob(i) = (double)i;
//     std::cout << "mpc.bob = " << mpc.bob.transpose() << std::endl;

// 	for (int i=0; i<MyNLP::x_numvecs; i++)
// 		std::cout << "X(" << i << ") = " << mpc.x(mpc.bob, i).transpose() << std::endl;

// 	std::cout << "uss = " << mpc.uss(mpc.bob).transpose() << std::endl;
// 	std::cout << "mpc.x(2)= " << mpc.x(mpc.bob, 2).transpose() << std::endl;

// 	MyNLP::x_t<scalar_t> err;
	
// 	mpc.sys<scalar_t>(
// 			mpc.x(mpc.bob, 1), 
// 			mpc.x(mpc.bob, 0),
// 			mpc.u(mpc.bob, 0), 
// 			err);

// 	std::cout << "sys(xp,x,u) = " << err.transpose() << std::endl;

// 	std::cout << std::endl << std::endl;
// 	std::cout << "---------------------------------------" << std::endl;
// 	std::cout << std::endl << std::endl;

// 	MyNLP::nlp_variable_t var;
// 	for(int i=0; i<MyNLP::VAR_SIZE; i++) var(i) = i;
// 	MyNLP::nlp_constraints_t equalities;
// 	mpc.equalities(var, equalities);

// 	std::cout << "var = " << var.transpose() << std::endl;
// 	std::cout << "xss = " << mpc.xss(var).transpose() << std::endl;
// 	std::cout << "equalities = " << equalities.transpose() << std::endl;

// 	std::cout << std::endl << std::endl;
// 	std::cout << "---------------------------------------" << std::endl;
// 	std::cout << std::endl << std::endl;

// 	MyNLP::x_t<scalar_t> xp;
// 	MyNLP::x_t<scalar_t> x;
// 	MyNLP::u_t<scalar_t> u;
// 	Eigen::Matrix<scalar_t, MyNLP::n, 1> out;

// 	Eigen::Matrix<scalar_t, MyNLP::n, MyNLP::x_t_size> J_xp;
// 	Eigen::Matrix<scalar_t, MyNLP::n, MyNLP::x_t_size> J_x;
// 	Eigen::Matrix<scalar_t, MyNLP::n, MyNLP::u_t_size> J_u;

// 	xp << 1,2,3;
// 	x << 5,6,7;
// 	u << 9,10;

//     // mpc.sys_jacobian(xp, x, u, out, J_xp, J_x, J_u);

// 	// std::cout << "out = " << out.transpose() << std::endl;
// 	// std::cout << "J_xp = " << std::endl << J_xp << std::endl;
// 	// std::cout << "J_x = " << std::endl << J_x << std::endl;
// 	// std::cout << "J_u = " << std::endl << J_u << std::endl;

// 	std::cout << std::endl << std::endl;
// 	std::cout << "---------------------------------------" << std::endl;
// 	std::cout << std::endl << std::endl;

// 	{

// 	// std::cout << "TESTING" << std::endl;

// 	MyNLP::nlp_variable_t var;
// 	MyNLP::nlp_constraints_t equalities;
// 	MyNLP::nlp_eq_jacobian_t jacobian;

// 	equalities.setZero();
// 	jacobian.setZero();

// 	for (int i=0; i<var.rows(); i++) var(i) = i;
// 	mpc.equalities_linearised(var, equalities, jacobian);

// 	std::cout << "equalities = " << equalities.transpose() << std::endl;
// 	std::cout << "jacobian = " << std::endl;
// 	std::cout << jacobian << std::endl;

// 	}

// 	std::cout << std::endl << std::endl;
// 	std::cout << "---------------------------------------" << std::endl;
// 	std::cout << std::endl << std::endl;

// 	{
// 	using scalar_t = double;
// 	Eigen::Matrix<scalar_t, 3, 1> xp;
// 	Eigen::Matrix<scalar_t, 3, 1> x;
// 	Eigen::Matrix<scalar_t, 2, 1> u;
// 	Eigen::Matrix<scalar_t, 3, 1> out;

// 	xp << 1,2,3;
// 	x << 4,5,6;
// 	u << 7,9;
// 	out.setZero();

// 	std::cout << "Calling test_function directly";
// 	test_function<scalar_t>(xp, x, u, out);
// 	std::cout << out.transpose() << std::endl;

// // template<int num_outputs, int ...n, typename scalar_t>
// // void make_jacobian(

// 	Eigen::Matrix<scalar_t, 3, 3> J_xp;
// 	Eigen::Matrix<scalar_t, 3, 3> J_x;
// 	Eigen::Matrix<scalar_t, 3, 2> J_u;

// 	J_x << 1,2,3,4,5,6,7,8,9;

// 	std::cout << "Calling through make_jacobian";
// 	// using AD_test_function = test_function<AutoDiffScalar<Matrix<scalar_t, 8, 1>>>;


// 	// jacobian<scalar_t, 3, 8, 3,3,2>(test_function<AD_scalar<scalar_t, 8>>,
// 	// 	out,
// 	// 	xp, x, u,
// 	// 	J_xp, J_x, J_u);
// 	std::cout << out.transpose() << std::endl;

// 	Eigen::Matrix<scalar_t, 5, 1> outxx;

// 	Eigen::Matrix<scalar_t, 10,1> var;
// 	var << 1,2,3,4,5,6,7,8,9,10;

// 	using ad_sys_t = AD_scalar<scalar_t, 8>;
// 	auto ad_sys = [&mpc](
// 		const Eigen::Ref<const MyNLP::x_t<ad_sys_t>>& xp, 
//     	const Eigen::Ref<const MyNLP::x_t<ad_sys_t>>& x, 
//     	const Eigen::Ref<const MyNLP::u_t<ad_sys_t>>& u,
//     	Eigen::Ref<MyNLP::x_t<ad_sys_t>> out
// 		){mpc.sys<ad_sys_t>(xp,x,u,out);};

// 	// for (int j=0; j<10; j++)
// 	// for (int i=0; i<2e6; i++)
// 	// call_jacobian<scalar_t, 3,8, 3,3,2>(ad_sys,
// 	// 	var.template segment<3>(4), x, u,
// 	// 	outxx.template segment<3>(1),
// 	// 	J_xp, J_x, J_u);


// 	std::cout << "Jacobians" << std::endl;
// 	std::cout << "J_xp" << std::endl;
// 	std::cout << J_xp << std::endl;
// 	std::cout << "J_x" << std::endl;
// 	std::cout << J_x << std::endl;
// 	std::cout << "J_u" << std::endl;
// 	std::cout << J_u << std::endl;
// 	}


	{

	// std::cout << "TESTING" << std::endl;

	MyNLP::nlp_variable_t var;
	MyNLP::nlp_constraints_t equalities;
	MyNLP::nlp_eq_jacobian_t jacobian;

	equalities.setZero();
	jacobian.setZero();

	for (int i=0; i<var.rows(); i++) var(i) = i;

	// for (int j=0; j<10; j++)
	// for (int i=0; i<2e6; i++)
	mpc.equalities_linearised(var, equalities, jacobian);

	// std::cout << "equalities = " << equalities.transpose() << std::endl;
	// std::cout << "jacobian = " << std::endl;
	// std::cout << jacobian << std::endl;

	}

	// {

	// using scalar_t = double;
	// Eigen::Matrix<scalar_t, 3, 1> xp;
	// Eigen::Matrix<scalar_t, 3, 1> x;
	// Eigen::Matrix<scalar_t, 2, 1> u;
	// Eigen::Matrix<scalar_t, 3, 1> out;

	// xp << 1,2,3;
	// x << 4,5,6;
	// u << 7,8;

	// // auto c = callback<MyNLP, scalar_t, 3, 3, 3, 2>(&mpc);

	// // for (int j=0; j<1e6; j++)
	// // 	for (int i=0; i<1e6; i++)
	// // 		mpc.template sys<scalar_t>(xp,x,u,out);
	// 		// mpc.c.operator()<scalar_t>(xp,x,u,out);

	// // std::cout << "out = " << out.transpose() << std::endl;
 //   	// Foo f; 
 //    // auto c1 = getCallback<&Foo::bar>(&f);
 //    // size_t x1; c1(x1, "123"); std::cout << "c1:" << x1 << "\n";

	// }
}
