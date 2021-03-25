#include <iostream>
#include <fmt/core.h>
#include <type_traits>
#include <math.h>

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "myproblem.hpp"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"

using namespace Ipopt;
// using namespace Eigen;


template<typename scalar_t, typename Prob>
class NLP_Ipopt: public TNLP, public Prob
{
public:
	NLP_Ipopt() {};

	using typename Prob::variable_t;
	using typename Prob::constraint_t;
	using typename Prob::constraint_jacobian_t;
	using typename Prob::cost_hessian_t;
	using typename Prob::cost_t;


	using Prob::NUM_VARS;
	using Prob::NUM_CON;
	using Prob::nnz_constraints_jacobian;


	// using LOpt<scalar_t>::constraint_t;
	// using LOpt<scalar_t>::variable_t;

	// using variable_t    = Eigen::Matrix<scalar_t, NUM_VARS, 1>;
	// using constraint_t = Eigen::Matrix<scalar_t, NUM_CON, 1>;


	bool get_nlp_info(
		Ipopt::Index&          n,
		Ipopt::Index&          m,
		Ipopt::Index&          nnz_jac_g,
		Ipopt::Index&          nnz_h_lag,
		IndexStyleEnum&        index_style
	)
	{
		n = NUM_VARS;
		m = NUM_CON;
		nnz_jac_g = nnz_constraints_jacobian;
		nnz_h_lag = 10; // Number of nonzero entries in the Hessian
		index_style = TNLP::C_STYLE;
		return true;
	};

	bool get_bounds_info(
		Ipopt::Index   n,
		Number* x_l,
		Number* x_u,
		Ipopt::Index   m,
		Number* g_l,
		Number* g_u
	)
	{
		// n and m are whatever we set in get_nlp_info
		assert(n == NUM_VARS);
		assert(m == NUM_CON);

		// TODO: Compute upper and lower variable bounds
		for( Ipopt::Index i = 0; i < 4; i++ )
		{
			x_l[i] = 1.0;
		}
		for( Ipopt::Index i = 0; i < 4; i++ )
		{
			x_u[i] = 5.0;
		}

		// TODO: Set upper and lower constraint bounds
		g_l[0] = 25;
		g_u[0] = 2e19;  // == infinity
		g_l[1] = g_u[1] = 40.0;  // Equality

		return true;	
	}

	bool get_starting_point(
		Ipopt::Index   n,
		bool    init_x,
		Number* x,
		bool    init_z,
		Number* z_L,
		Number* z_U,
		Ipopt::Index   m,
		bool    init_lambda,
		Number* lambda
	)
	{
		assert(init_x == true);  // We provide initial primal
		assert(init_z == false);  // Not sure what this is
		assert(init_lambda == false);  // Not providing dual

		x[0] = 1.0;
		x[1] = 5.0;
		x[2] = 5.0;
		x[3] = 1.0;

		return true;
	}

	bool eval_f(
		Ipopt::Index         n,
		const Number* x,
		bool          new_x,
		Number&       obj_value
	)
	{
		assert(n == NUM_VARS);

		obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

		return true;
	}

	bool eval_grad_f(
		Ipopt::Index         n,
		const Number* x,
		bool          new_x,
		Number*       grad_f
	)
	{
		assert(n == NUM_VARS);

		grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
		grad_f[1] = x[0] * x[3];
		grad_f[2] = x[0] * x[3] + 1;
		grad_f[3] = x[0] * (x[0] + x[1] + x[2]);

		return true;
	}

	bool eval_g(
		Ipopt::Index         n,
		const Number* x,
		bool          new_x,
		Ipopt::Index         m,
		Number*       g
	)
	{
		assert(n == NUM_VARS);
		assert(m == NUM_CON);

		Eigen::Map<const variable_t > var(x);
		Eigen::Map<constraint_t > constraints(g);
		this->constraints(var, constraints);

		return true;   	
	}

   /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
	bool eval_jac_g(
		Ipopt::Index         n,
		const Number* x,
		bool          new_x,
		Ipopt::Index         m,
		Ipopt::Index         nele_jac,
		Ipopt::Index*        iRow,
		Ipopt::Index*        jCol,
		Number*       values
	)
	{
		assert( n == 4 );
		assert( m == 2 );

		if( values == NULL )
		{
			Eigen::SparseMatrix<scalar_t> jac_g(Prob::NUM_CON, Prob::NUM_VARS);
			this->constraints_sparse_initialize(jac_g);

			int ind = 0;
		    for (int k=0; k < jac_g.outerSize(); ++k)
		    {
		        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(jac_g, k); it; ++it, ind++)
		        {
					iRow[ind] = it.row();
					jCol[ind] = it.col();
		        }
		    }
		}
		else
		{
			Eigen::Map<const variable_t> var(x);
			constraint_t tmp;
			Eigen::Map<Eigen::Matrix<scalar_t, nnz_constraints_jacobian, 1>> J(values);
			this->constraints_sparse_jacobian_values(var, tmp, J); 
		}

		return true;
	}

   /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
	bool eval_h(
		Ipopt::Index         n,
		const Number* x,
		bool          new_x,
		Number        obj_factor,
		Ipopt::Index         m,
		const Number* lambda,
		bool          new_lambda,
		Ipopt::Index         nele_hess,
		Ipopt::Index*        iRow,
		Ipopt::Index*        jCol,
		Number*       values
	)
	{
		assert(n == 4);
		assert(m == 2);

		if( values == NULL )
		{
			// return the structure. This is a symmetric matrix, fill the lower left
			// triangle only.

			// the hessian for this problem is actually dense
			Ipopt::Index idx = 0;
			for( Ipopt::Index row = 0; row < 4; row++ )
			{
				for( Ipopt::Index col = 0; col <= row; col++ )
				{
					iRow[idx] = row;
					jCol[idx] = col;
					idx++;
				}
			}

			assert(idx == nele_hess);
		}
		else
		{
			// return the values. This is a symmetric matrix, fill the lower left
			// triangle only

			// fill the objective portion
			values[0] = obj_factor * (2 * x[3]); // 0,0

			values[1] = obj_factor * (x[3]);     // 1,0
			values[2] = 0.;                      // 1,1

			values[3] = obj_factor * (x[3]);     // 2,0
			values[4] = 0.;                      // 2,1
			values[5] = 0.;                      // 2,2

			values[6] = obj_factor * (2 * x[0] + x[1] + x[2]); // 3,0
			values[7] = obj_factor * (x[0]);                   // 3,1
			values[8] = obj_factor * (x[0]);                   // 3,2
			values[9] = 0.;                                    // 3,3

			// add the portion for the first constraint
			values[1] += lambda[0] * (x[2] * x[3]); // 1,0

			values[3] += lambda[0] * (x[1] * x[3]); // 2,0
			values[4] += lambda[0] * (x[0] * x[3]); // 2,1

			values[6] += lambda[0] * (x[1] * x[2]); // 3,0
			values[7] += lambda[0] * (x[0] * x[2]); // 3,1
			values[8] += lambda[0] * (x[0] * x[1]); // 3,2

			// add the portion for the second constraint
			values[0] += lambda[1] * 2; // 0,0

			values[2] += lambda[1] * 2; // 1,1

			values[5] += lambda[1] * 2; // 2,2

			values[9] += lambda[1] * 2; // 3,3
		}

		return true;
	}   

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
	void finalize_solution(
		SolverReturn               status,
		Ipopt::Index                      n,
		const Number*              x,
		const Number*              z_L,
		const Number*              z_U,
		Ipopt::Index                      m,
		const Number*              g,
		const Number*              lambda,
		Number                     obj_value,
		const IpoptData*           ip_data,
		IpoptCalculatedQuantities* ip_cq
	)
	{
		// here is where we would store the solution to variables, or write to a file, etc
		// so we could use the solution.

		// For this example, we write the solution to the console
		std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
		for( Ipopt::Index i = 0; i < n; i++ )
		{
			std::cout << "x[" << i << "] = " << x[i] << std::endl;
		}

		std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
		for( Ipopt::Index i = 0; i < n; i++ )
		{
			std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
		}
		for( Ipopt::Index i = 0; i < n; i++ )
		{
			std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
		}

		std::cout << std::endl << std::endl << "Objective value" << std::endl;
		std::cout << "f(x*) = " << obj_value << std::endl;

		std::cout << std::endl << "Final value of the constraints:" << std::endl;
		for( Ipopt::Index i = 0; i < m; i++ )
		{
			std::cout << "g(" << i << ") = " << g[i] << std::endl;
		}
	}

private:
   /**@name Methods to block default compiler methods.
    *
    * The compiler automatically generates the following three methods.
    *  Since the default compiler implementation is generally not what
    *  you want (for all but the most simple classes), we usually
    *  put the declarations of these methods in the private section
    *  and never implement them. This prevents the compiler from
    *  implementing an incorrect "default" behavior without us
    *  knowing. (See Scott Meyers book, "Effective C++")
    */
   //@{

   NLP_Ipopt(
      const NLP_Ipopt&
   );

   NLP_Ipopt& operator=(
      const NLP_Ipopt&
   );
   //@}

   // /** @name NLP data */
   // //@{
   // /** Number of variables */
   // Ipopt::Index N_;
   // /** Value of constants in constraints */
   // Number* a_;
   // //@}
};


// template<int... x>
// struct Len {};

// template<template <typename...> class Var_Lengths, template <typename...> class Var_Lengths>
// struct Jacobian_Setter
// {
// 	// const std::initializer_list<int> var_lengths;
// 	// const std::initializer_list<int> var_offsets;

// 	// Jacobian_Setter(std::initializer_list<int> _var_lengths, std::initializer_list<int> _var_offsets) :
// 	// 	var_lengths(_var_lengths), var_offsets(_var_offsets) {}

// 	// template<int len>
// 	// EIGEN_STRONG_INLINE void set_dense(Eigen::Ref<Eigen::Matrix<scalar_t, len, 1> )
// };

// template<int... var_sizes>
// // void foo(std::initializer_list<int> a, std::initializer_list<int> b)
// void foo(const Len<var_sizes...> a, const Len<var_lengths...> b)
// {
// 	std::cout << "A = " << type_name<decltype(a)>() << std::endl;
// 	std::cout << "B = " << type_name<decltype(b)>() << std::endl;

//     int offset = 0;
//     (void)std::initializer_list<int>{ 
//         (
//             J.row(i) = deriv.template segment<input_sizes>(offset), 
//             offset += input_sizes,
//             0
//         )... 
//     };


// 	// for(auto i: a)
// 	// 	std::cout << i << std::endl;
// }

// template
// template<
// EIGEN_STRONG_INLINE make_setter()
// [J](int iRow, Eigen::Ref<Eigen::Matrix<scalar_t, 1, x1_len+x2_len+x3_len+x4_len>> row)
// {
// 	for(auto )

// 	J.row(eq1_offset() + iRow).template segment<x1_len>(x1_offset()) = row.template segment<x1_len>(offset);

// 	J.template block<1, x1_len>(eq1_offset(), x1_offset()),

// using pairlist = std::initializer_list<std::pair<int,int>>;

template<typename V, typename V2, typename scalar1 = typename V::Scalar, typename scalar2 = typename V2::Scalar>
void test_function_impl(Eigen::MatrixBase<V>& v,
	Eigen::MatrixBase<V2>& v2)
{
	static_assert(V::ColsAtCompileTime == 1, "col test");
	static_assert(std::is_same<typename V::Scalar, typename V2::Scalar>::value, "type test");

	std::cout << "v = " << v.transpose() << std::endl;
	std::cout << "type_name(v) = " << type_name<decltype(v)>() << std::endl;
	v[0] = 100.0;

	// std::cout << "data during = " << v.data() << std::endl;
}

// template<typename V>
// void test_function(V&& v)
// {
// 	std::cout << "test_function: type_name(v) = " << type_name<decltype(v)>() << std::endl;
// 	test_function_impl<typename V::Scalar, V::RowsAtCompileTime>(v);
// }


void test_function(std::array<int, 4> x)
{
	std::cout << "x[0] = " << x[0] << std::endl;
}



int main(void)
{
	// test_function({1,2,3,4});


	using scalar_t = double;
	using NLP = LOpt<scalar_t>;

	NLP nlp;
	NLP::variable_t var;
	NLP::constraint_t con;

	// // Eigen::Matrix<scalar_t, 1, 1> x0;
	// // Eigen::Matrix<scalar_t, 1, 1> x1;
	// // Eigen::Matrix<scalar_t, 1, 1> x2;
	// // Eigen::Matrix<scalar_t, 1, 1> x3;
	// // Eigen::Matrix<scalar_t, 1, 1> out;

	// // x0 << 1;
	// // x1 << 2;
	// // x2 << 3;
	// // x3 << 3;

	// // nlp.func2(x0, x1, x2, x3, out);

	// // std::cout << "x = [" << x0 << ", " << x1 << ", " << x2 << ", " << x3 << "]" << std::endl;
	// // std::cout << "out = " << out << std::endl;

	// // Eigen::Matrix<scalar_t, 1, 1> J0;
	// // Eigen::Matrix<scalar_t, 1, 1> J1;
	// // Eigen::Matrix<scalar_t, 1, 1> J2;
	// // Eigen::Matrix<scalar_t, 1, 1> J3;
	// // nlp.func2(x0, x1, x2, x3, out, J0, J1, J2, J3);

	// // std::cout << "J = [" << J0 << ", " << J1 << ", " << J2 << ", " << J3 << "]" << std::endl;

	// // Eigen::Matrix<scalar_t, 10, 10> J;

	// // constexpr int eq1_offset = 1;
	// // // eq1_offset
	// // // auto filler = Fill_Dense<decltype(J)>({0,1,2,3}, {1,1,1,1}, J);
	// // // auto filler = Fill_Dense<decltype(J)>(pairlist{{0,1}, {1,1}, {2,1}, {3,1}}, J);

	// // std::cout << "J = \n" << J << std::endl;

	// // for (long i=0; i<100000000; i++)
	// // {		
	// // 	auto filler = Fill_Dense<decltype(J), 1, 0, 1, 5, 6>(J);
	// // 	x0(0) += 1;
	// // 	nlp.func2(x0, x1, x2, x3, out, filler);
	// // }
	// // std::cout << "J = \n" << J << std::endl;


	for(int i=0; i<NLP::NUM_VARS; i++) var(i) = i;

	std::cout << "var = " << var.transpose() << std::endl;

	nlp.constraints(var, con);
	std::cout << "con = " << con.transpose() << std::endl;
  
	NLP::constraint_jacobian_t J;
	J = decltype(J)::Zero();
	std::cout << "J.shape = " << decltype(J)::RowsAtCompileTime << " x " << decltype(J)::ColsAtCompileTime << std::endl;
	// for (long i=0; i<10000000; i++)
	{
		var(0)++;
		nlp.constraints(var, con, J);
	}
	std::cout << "J = \n" << J << std::endl;

	std::cout << "=========================================\n";

	{
	Eigen::SparseMatrix<scalar_t> J_sparse(NLP::NUM_CON, NLP::NUM_VARS);
	nlp.constraints_sparse_initialize(J_sparse);
	for(int i=0; i<NLP::nnz_constraints_jacobian; i++) J_sparse.valuePtr()[i] = i;
	std::cout << "J_sparse = \n" << Eigen::MatrixXd(J_sparse) << std::endl;
	var(0) = 0;
	// for (long i=0; i<10000000; i++)
	{
		var(0)++;
		nlp.constraints(var, con, J_sparse);		
	}
	std::cout << "J_sparse = \n" << Eigen::MatrixXd(J_sparse) << std::endl;
	}

	std::cout << "=========================================\n";

	// // auto x = var.template segment<2>(2);
	// Eigen::Matrix<double, 10, 20> t;
	// t.array() = 0;
	// std::cout << "t = \n" << t << std::endl;

	// auto x = t.row(3).template segment<2>(2).transpose();
	// std::cout << "x = " << x.transpose() << std::endl;
	// std::cout << "x.shape = (" << x.rows() << ", " << x.cols() << ")\n";
	// std::cout << "type_name(x) = " << type_name<decltype(x)>() << std::endl; 
	// std::cout << "data pre = " << x.data() << std::endl;


	// std::cout << "--------- calling ---------" << std::endl;
	// Eigen::Matrix<double, 3,1> y;
	// test_function_impl(x, y);
	// std::cout << "--------- post ---------" << std::endl;
	
	// std::cout << "x = " << x.transpose() << std::endl;
	// std::cout << "data post = " << x.data() << std::endl;


	// {
	// Eigen::SparseMatrix<scalar_t> J_sparse(NLP::NUM_CON, NLP::NUM_VARS);
	// nlp.constraints_sparse_initialize(J_sparse);
	// Eigen::Map<Eigen::Matrix<scalar_t, NLP::nnz_constraints_jacobian, 1>> Jvalues(J_sparse.valuePtr(), J_sparse.nonZeros(), 1);
	// var(0) = 0;
	// // for (long i=0; i<10000000; i++)
	// {
	// 	var(0)++;
	// 	nlp.constraints_sparse_jacobian_crs(var, con, Jvalues);
	// }
	// std::cout << "J_sparse = \n" << Eigen::MatrixXd(J_sparse) << std::endl;

	// std::cout << "J_sparse.Values = " << Jvalues.transpose() << std::endl;
	// }


 //   SmartPtr<TNLP> mynlp = new NLP_Ipopt<double, NLP>();
 //   Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

 //   app->Options()->SetNumericValue("tol", 1e-7);
 //   app->Options()->SetStringValue("mu_strategy", "adaptive");
 //   app->Options()->SetStringValue("output_file", "ipopt.out");

 //   ApplicationReturnStatus status;
 //   status = app->Initialize();
 //   if( status != Solve_Succeeded )
 //   {
 //      std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
 //      return (int) status;
 //   }

 //   // Ask Ipopt to solve the problem
 //   status = app->OptimizeTNLP(mynlp);

 //   if( status == Solve_Succeeded )
 //   {
 //      std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
 //   }
 //   else
 //   {
 //      std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
 //   }

 //   // As the SmartPtrs go out of scope, the reference count
 //   // will be decremented and the objects will automatically
 //   // be deleted.

 //   return (int) status;
}

