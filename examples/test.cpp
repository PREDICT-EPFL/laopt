#include <ctime>

#include <iostream>
#include <fmt/core.h>
#include <type_traits>
#include <math.h>
#include <array>

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "myproblem.hpp"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"

// #include "Delegate.h"

using namespace Ipopt;
using namespace Eigen;

/*
TODO:
- variable bounds
- bounds info

- cost evaluation, jacobian and hessian
- get starting point
- finalize solution
*/


template<typename scalar_t, typename Prob>
class NLP_Ipopt: public TNLP, public Prob
{
public:
	using typename Prob::variable_t;
	using typename Prob::constraint_t;
	using typename Prob::constraint_jacobian_t;
	using typename Prob::obj_gradient_t;
	using typename Prob::obj_hessian_t;
	using typename Prob::obj_t;


	using Prob::NUM_VARS;
	using Prob::NUM_CON;
	using Prob::nnz_constraints_jacobian;

	variable_t x0;  // Initial iterate

	NLP_Ipopt() :
		jac_g(Prob::NUM_CON, Prob::NUM_VARS),
		hessian_g(Prob::NUM_VARS, Prob::NUM_VARS)
		{
		this->constraints_sparse_initialize(jac_g);  // Compute sparsity structure of constraints
		for(int i=0; i<NUM_VARS; i++)
			x0[i] = 0.0;
	};

	// Sparsity structures
	Eigen::SparseMatrix<scalar_t> jac_g;
	Eigen::SparseMatrix<scalar_t> hessian_g;

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

		this->variable_bounds(Eigen::Map<variable_t>(x_l), Eigen::Map<variable_t>(x_u));
		this->constraint_bounds(Eigen::Map<constraint_t>(g_l), Eigen::Map<constraint_t>(g_u));

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

		for (int i=0; i<n; i++)
			x[i] = x0[i];

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
		obj_value = this->objective(Eigen::Map<const variable_t>(x));
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
		this->objective(Eigen::Map<const variable_t>(x), Eigen::Map<obj_gradient_t>(grad_f));
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
			// Copy the pre-computed jacobian structure into the Ipopt variables
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

			// Map the Ipopt values vectors into our pre-computed sparse matrix structure
			Eigen::Map<Eigen::SparseMatrix<scalar_t>> J(jac_g.rows(), jac_g.rows(), jac_g.nonZeros(), 
													    jac_g.outerIndexPtr(), jac_g.innerIndexPtr(),
													    values);
			constraint_t tmp;
			this->constraints(var, tmp, J); 
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
		return false;
		// assert(n == 4);
		// assert(m == 2);

		// if( values == NULL )
		// {
		// 	// return the structure. This is a symmetric matrix, fill the lower left
		// 	// triangle only.

		// 	// the hessian for this problem is actually dense
		// 	Ipopt::Index idx = 0;
		// 	for( Ipopt::Index row = 0; row < 4; row++ )
		// 	{
		// 		for( Ipopt::Index col = 0; col <= row; col++ )
		// 		{
		// 			iRow[idx] = row;
		// 			jCol[idx] = col;
		// 			idx++;
		// 		}
		// 	}

		// 	assert(idx == nele_hess);
		// }
		// else
		// {
		// 	// return the values. This is a symmetric matrix, fill the lower left
		// 	// triangle only

		// 	// fill the objective portion
		// 	values[0] = obj_factor * (2 * x[3]); // 0,0

		// 	values[1] = obj_factor * (x[3]);     // 1,0
		// 	values[2] = 0.;                      // 1,1

		// 	values[3] = obj_factor * (x[3]);     // 2,0
		// 	values[4] = 0.;                      // 2,1
		// 	values[5] = 0.;                      // 2,2

		// 	values[6] = obj_factor * (2 * x[0] + x[1] + x[2]); // 3,0
		// 	values[7] = obj_factor * (x[0]);                   // 3,1
		// 	values[8] = obj_factor * (x[0]);                   // 3,2
		// 	values[9] = 0.;                                    // 3,3

		// 	// add the portion for the first constraint
		// 	values[1] += lambda[0] * (x[2] * x[3]); // 1,0

		// 	values[3] += lambda[0] * (x[1] * x[3]); // 2,0
		// 	values[4] += lambda[0] * (x[0] * x[3]); // 2,1

		// 	values[6] += lambda[0] * (x[1] * x[2]); // 3,0
		// 	values[7] += lambda[0] * (x[0] * x[2]); // 3,1
		// 	values[8] += lambda[0] * (x[0] * x[1]); // 3,2

		// 	// add the portion for the second constraint
		// 	values[0] += lambda[1] * 2; // 0,0

		// 	values[2] += lambda[1] * 2; // 1,1

		// 	values[5] += lambda[1] * 2; // 2,2

		// 	values[9] += lambda[1] * 2; // 3,3
		// }

		// return true;
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

		// // For this example, we write the solution to the console
		// std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
		// for( Ipopt::Index i = 0; i < n; i++ )
		// {
		// 	std::cout << "x[" << i << "] = " << x[i] << std::endl;
		// }

		// std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
		// for( Ipopt::Index i = 0; i < n; i++ )
		// {
		// 	std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
		// }
		// for( Ipopt::Index i = 0; i < n; i++ )
		// {
		// 	std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
		// }

		std::cout << std::endl << std::endl << "Objective value" << std::endl;
		std::cout << "f(x*) = " << obj_value << std::endl;

		// std::cout << std::endl << "Final value of the constraints:" << std::endl;
		// for( Ipopt::Index i = 0; i < m; i++ )
		// {
		// 	std::cout << "g(" << i << ") = " << g[i] << std::endl;
		// }


		const int N = 5;
		std::cout << "========== Solution =========\n";
		auto X = Eigen::Map<const variable_t>(x);
		std::cout << "x0 = " << Prob::x0_get(X).transpose() << std::endl;
		std::cout << "x1 = " << Prob::x1_get(X).transpose() << std::endl;
		std::cout << "x2 = " << Prob::x2_get(X).transpose() << std::endl;
		std::cout << "x3 = " << Prob::x3_get(X).transpose() << std::endl;
		std::cout << "x4 = " << Prob::x4_get(X).transpose() << std::endl;
		std::cout << "x5 = " << Prob::x5_get(X).transpose() << std::endl;
		std::cout << "x6 = " << Prob::x6_get(X).transpose() << std::endl;
		std::cout << "x7 = " << Prob::x7_get(X).transpose() << std::endl;
		std::cout << "x8 = " << Prob::x8_get(X).transpose() << std::endl;
		std::cout << "x9 = " << Prob::x9_get(X).transpose() << std::endl;

		std::cout << "u0 = " << Prob::u0_get(X).transpose() << std::endl;
		std::cout << "u1 = " << Prob::u1_get(X).transpose() << std::endl;
		std::cout << "u2 = " << Prob::u2_get(X).transpose() << std::endl;
		std::cout << "u3 = " << Prob::u3_get(X).transpose() << std::endl;
		std::cout << "u4 = " << Prob::u4_get(X).transpose() << std::endl;
		std::cout << "u5 = " << Prob::u5_get(X).transpose() << std::endl;
		std::cout << "u6 = " << Prob::u6_get(X).transpose() << std::endl;
		std::cout << "u7 = " << Prob::u7_get(X).transpose() << std::endl;
		std::cout << "u8 = " << Prob::u8_get(X).transpose() << std::endl;
		std::cout << "u9 = " << Prob::u9_get(X).transpose() << std::endl;

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









// template<typename _scalar_t>
// struct Test
// {
// 	// typedef _scalar_t scalar_t;
// 	using scalar_t = _scalar_t;

//     Eigen::Matrix2d Q = {{2, 1}, {1, 4}};


// 	template<typename T>
// 	EIGEN_STRONG_INLINE void _cost(const Eigen::Ref<const Eigen::Matrix<T, 2, 1>> x1, 
// 			  const Eigen::Ref<const Eigen::Matrix<T, 3, 1>> x2, 
// 			  const Eigen::Ref<const Eigen::Matrix<T, 5, 1>> x3, 
// 				T &y) const
// 	{
// 	    y = 0.5*x1.dot(Q.template cast<T>() * x1) + 4 * x2[0] * x3[1] * x2.sum() + 7 * x3.sum();
// 	}
//     MAKE_HESSIAN(Test, cost);
//     costHessian<2,3,5> cost;

// 	Test() : cost(this) {};
// };





int main(void)
{
	// using scalar_t = double;
	// using OPT = LOpt<scalar_t>;
	// OPT opt;

	// Eigen::Matrix<scalar_t, 21, 1> var = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};

	// scalar_t val;
	// Eigen::Matrix<scalar_t, 1, 50> gradient;
	// SparseMatrix<scalar_t> hessian(OPT::NUM_VARS, OPT::NUM_VARS);
	// opt.objective_sparse_initialize(hessian);

	// // val = opt.objective(var.template segment<OPT::NUM_VARS>(1));
	// // val = opt.objective(var.template segment<OPT::NUM_VARS>(1), gradient.segment<OPT::NUM_VARS>(2));
	// val = opt.objective(var.template segment<OPT::NUM_VARS>(1), gradient.segment<OPT::NUM_VARS>(2), hessian);


	// std::cout << "val = " << val << std::endl;
	// std::cout << "gradient = " << gradient << std::endl;
	// std::cout << "hessian = \n" << hessian << std::endl;


#if 1
	using myNLP = NLP_Ipopt<double, LOpt<double>>;
   SmartPtr<myNLP> mynlp = new myNLP();
   Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

   app->Options()->SetNumericValue("tol", 1e-7);
   app->Options()->SetStringValue("mu_strategy", "adaptive");
   app->Options()->SetStringValue("output_file", "ipopt.out");
   app->Options()->SetStringValue("hessian_approximation", "limited-memory");

   ApplicationReturnStatus status;
   status = app->Initialize();
   if( status != Solve_Succeeded )
   {
      std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
      return (int) status;
   }

	mynlp->x0[0] = 1.0;
	mynlp->x0[1] = 5.0;
	mynlp->x0[2] = 5.0;
	mynlp->x0[3] = 1.0;


   // Ask Ipopt to solve the problem
   mynlp->x_initial = {1,2};
   mynlp->q = {1,1};

   std::cout << "x_initial = " << mynlp->x_initial << std::endl;
   std::cout << "q = " << mynlp->q << std::endl;   
   status = app->OptimizeTNLP(mynlp);

   mynlp->q = {100000,100000};

   std::cout << "x_initial = " << mynlp->x_initial << std::endl;
   std::cout << "q = " << mynlp->q << std::endl;   
   status = app->OptimizeTNLP(mynlp);

   // if( status == Solve_Succeeded )
   // {
   //    std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
   // }
   // else
   // {
   //    std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
   // }

   return (int) status;
 #endif
}
