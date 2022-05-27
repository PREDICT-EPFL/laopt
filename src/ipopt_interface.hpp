#ifndef IPOPT_INTERFACE_HPP
#define IPOPT_INTERFACE_HPP

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"

using namespace Ipopt;

struct MyNLP: public TNLP
{
public:

	/* Constructor. */
	MyNLP()
	{ 
		std::cout << "hello" << std::endl;
	}

	bool get_nlp_info(
	   Index&          n,
	   Index&          m,
	   Index&          nnz_jac_g,
	   Index&          nnz_h_lag,
	   IndexStyleEnum& index_style
	)
	{
	   // The problem described in MyNLP.hpp has 2 variables, x1, & x2,
	   n = 2;

	   // one equality constraint,
	   m = 1;

	   // 2 nonzeros in the jacobian (one for x1, and one for x2),
	   nnz_jac_g = 2;

	   // and 2 nonzeros in the hessian of the lagrangian
	   // (one in the hessian of the objective for x2,
	   //  and one in the hessian of the constraints for x1)
	   nnz_h_lag = 2;

	   // We use the standard fortran index style for row/col entries
	   index_style = FORTRAN_STYLE;

	   return true;
	}

	bool get_bounds_info(
	   Index   n,
	   Number* x_l,
	   Number* x_u,
	   Index   m,
	   Number* g_l,
	   Number* g_u
	)
	{
	   // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
	   // If desired, we could assert to make sure they are what we think they are.
	   assert(n == 2);
	   assert(m == 1);

	   // x1 has a lower bound of -1 and an upper bound of 1
	   x_l[0] = -1.0;
	   x_u[0] = 1.0;

	   // x2 has no upper or lower bound, so we set them to
	   // a large negative and a large positive number.
	   // The value that is interpretted as -/+infinity can be
	   // set in the options, but it defaults to -/+1e19
	   x_l[1] = -1.0e19;
	   x_u[1] = +1.0e19;

	   // we have one equality constraint, so we set the bounds on this constraint
	   // to be equal (and zero).
	   g_l[0] = g_u[0] = 0.0;

	   return true;
	}

	bool get_starting_point(
	   Index   n,
	   bool    init_x,
	   Number* x,
	   bool    init_z,
	   Number* z_L,
	   Number* z_U,
	   Index   m,
	   bool    init_lambda,
	   Number* lambda
	)
	{
	   // Here, we assume we only have starting values for x, if you code
	   // your own NLP, you can provide starting values for the others if
	   // you wish.
	   assert(init_x == true);
	   assert(init_z == false);
	   assert(init_lambda == false);

	   // we initialize x in bounds, in the upper right quadrant
	   x[0] = 0.5;
	   x[1] = 1.5;

	   return true;
	}

	bool eval_f(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Number&       obj_value
	)
	{
	   // return the value of the objective function
	   Number x2 = x[1];

	   obj_value = -(x2 - 2.0) * (x2 - 2.0);

	   return true;
	}

	bool eval_grad_f(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Number*       grad_f
	)
	{
	   // return the gradient of the objective function grad_{x} f(x)

	   // grad_{x1} f(x): x1 is not in the objective
	   grad_f[0] = 0.0;

	   // grad_{x2} f(x):
	   Number x2 = x[1];
	   grad_f[1] = -2.0 * (x2 - 2.0);

	   return true;
	}

	bool eval_g(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Index         m,
	   Number*       g
	)
	{
	   // return the value of the constraints: g(x)
	   Number x1 = x[0];
	   Number x2 = x[1];

	   g[0] = -(x1 * x1 + x2 - 1.0);

	   return true;
	}

	bool eval_jac_g(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Index         m,
	   Index         nele_jac,
	   Index*        iRow,
	   Index*        jCol,
	   Number*       values
	)
	{
	   if( values == NULL )
	   {
	      // return the structure of the jacobian of the constraints

	      // element at 1,1: grad_{x1} g_{1}(x)
	      iRow[0] = 1;
	      jCol[0] = 1;

	      // element at 1,2: grad_{x2} g_{1}(x)
	      iRow[1] = 1;
	      jCol[1] = 2;
	   }
	   else
	   {
	      // return the values of the jacobian of the constraints
	      Number x1 = x[0];

	      // element at 1,1: grad_{x1} g_{1}(x)
	      values[0] = -2.0 * x1;

	      // element at 1,2: grad_{x1} g_{1}(x)
	      values[1] = -1.0;
	   }

	   return true;
	}

	bool eval_h(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Number        obj_factor,
	   Index         m,
	   const Number* lambda,
	   bool          new_lambda,
	   Index         nele_hess,
	   Index*        iRow,
	   Index*        jCol,
	   Number*       values
	)
	{
	   if( values == NULL )
	   {
	      // return the structure. This is a symmetric matrix, fill the lower left
	      // triangle only.

	      // element at 1,1: grad^2_{x1,x1} L(x,lambda)
	      iRow[0] = 1;
	      jCol[0] = 1;

	      // element at 2,2: grad^2_{x2,x2} L(x,lambda)
	      iRow[1] = 2;
	      jCol[1] = 2;

	      // Note: off-diagonal elements are zero for this problem
	   }
	   else
	   {
	      // return the values

	      // element at 1,1: grad^2_{x1,x1} L(x,lambda)
	      values[0] = -2.0 * lambda[0];

	      // element at 2,2: grad^2_{x2,x2} L(x,lambda)
	      values[1] = -2.0 * obj_factor;

	      // Note: off-diagonal elements are zero for this problem
	   }

	   return true;
	}

	void finalize_solution(
	   SolverReturn               status,
	   Index                      n,
	   const Number*              x,
	   const Number*              z_L,
	   const Number*              z_U,
	   Index                      m,
	   const Number*              g,
	   const Number*              lambda,
	   Number                     obj_value,
	   const IpoptData*           ip_data,
	   IpoptCalculatedQuantities* ip_cq
	)
	{
		Eigen::Map<const Eigen::VectorX<Number>> sol(x, n);
		std::cout << "Solution = " << sol.transpose() << std::endl;

	   // here is where we would store the solution to variables, or write to a file, etc
	   // so we could use the solution. Since the solution is displayed to the console,
	   // we currently do nothing here.
	}
};
















// template<typename Prob>
// class NLP_Ipopt : public Ipopt::TNLP
// {
// public:
// 	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

// 	using scalar_t = typename Prob::scalar_t;

// 	// static constexpr std::size_t num_variables = Prob::num_variables;
// 	// static constexpr std::size_t num_constraints = Prob::constraints::output_size;

// 	// variable_t x0;  // Initial iterate

// 	// // Weight for the objective function
// 	// typename Prob::objective::weight_t w_obj; 

// 	// // Sparsity structures
// 	// Eigen::SparseMatrix<scalar_t> jac_g;
// 	// Eigen::SparseMatrix<scalar_t> hessian_lagrangian;

// 	// struct sol_t
// 	// {
// 	// 	variable_t primal;
// 	// 	constraint_t dual_g;
// 	// 	variable_t dual_x;
// 	// 	obj_t obj;
// 	// 	obj_gradient_t obj_gradient;
// 	// };
// 	// sol_t sol;

// 	Prob& prob;

// 	Eigen::VectorX<scalar_t> initial_primal;

// 	NLP_Ipopt(Prob& _prob) : prob(_prob)
// 	{
// 		initial_primal.array() = 0;
// 	};

// 	bool get_nlp_info(
// 		Ipopt::Index&          n,
// 		Ipopt::Index&          m,
// 		Ipopt::Index&          nnz_jac_g,
// 		Ipopt::Index&          nnz_h_lag,
// 		IndexStyleEnum&        index_style
// 	)
// 	{
// 		n = prob.num_variables();
// 		m = prob.constraints.value.buffer_rows();
// 		nnz_jac_g = prob.constraints.jacobian.sparsity_structure.nonZeros();
// 		// nnz_h_lag = Prob::lagrangian::hessian_nnz; // Total NNZ - we only want lower part
// 		index_style = TNLP::C_STYLE;

// 		// Compute nnz in the lower-triangular part of the hessian
// 		int ind = 0;
// 		for (int k=0; k < prob.lagrangian.hessian.sparsity_structure.outerSize(); ++k)
// 			for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(prob.lagrangian.hessian.sparsity_structure, k); it; ++it)
// 				if(it.row() >= it.col())  // Store only the lower triangular part
// 					ind++;
// 		nnz_h_lag = ind;
// 		return true;
// 	};

// 	bool get_bounds_info(
// 		Ipopt::Index   n,
// 		Ipopt::Number* x_l,
// 		Ipopt::Number* x_u,
// 		Ipopt::Index   m,
// 		Ipopt::Number* g_l,
// 		Ipopt::Number* g_u
// 	)
// 	{
// 		// n and m are whatever we set in get_nlp_info
// 		assert(n == num_variables);
// 		assert(m == num_constraints);

// 		Eigen::Map<variable_t>(x_l) = prob.x_l;
// 		Eigen::Map<variable_t>(x_u) = prob.x_u;
// 		Eigen::Map<variable_t>(g_l) = prob.constraints.lb;
// 		Eigen::Map<variable_t>(g_u) = prob.constraints.ub;

// 		return true;	
// 	}

// 	bool get_starting_point(
// 		Ipopt::Index   n,
// 		bool    init_x,
// 		Ipopt::Number* x,
// 		bool    init_z,
// 		Ipopt::Number* z_L,
// 		Ipopt::Number* z_U,
// 		Ipopt::Index   m,
// 		bool    init_lambda,
// 		Ipopt::Number* lambda
// 	)
// 	{
// 		assert(init_x == true);  // We provide initial primal
// 		assert(init_z == false);  // Not sure what this is
// 		assert(init_lambda == false);  // Not providing dual

// 		Eigen::Map<Eigen::VectorX<scalar_t>> val(x,n) = initial_primal;

// 		return true;
// 	}

// 	bool eval_f(
// 		Ipopt::Index         n,
// 		const Ipopt::Number* x,
// 		bool                 new_x,
// 		Ipopt::Number&       obj_value
// 	)
// 	{
// 		assert(n == prob.num_variables());

// 		prob.set_decision_variable(x);
// 		prob.eval_constraints(lampc::Eval());

// 		obj_value = Prob::objective::eval(param, w_obj, Eigen::Map<const variable_t>(x));
// 		return true;
// 	}

// 	bool eval_grad_f(
// 		Ipopt::Index         n,
// 		const Ipopt::Number* x,
// 		bool          new_x,
// 		Ipopt::Number*       grad_f
// 	)
// 	{
// 		assert(n == num_variables);
// 		Prob::objective::eval(param, w_obj, Eigen::Map<const variable_t>(x), Eigen::Map<obj_gradient_t>(grad_f));
// 		return true;
// 	}

// 	bool eval_g(
// 		Ipopt::Index         n,
// 		const Ipopt::Number* x,
// 		bool          new_x,
// 		Ipopt::Index         m,
// 		Ipopt::Number*       g
// 	)
// 	{
// 		assert(n == num_variables);
// 		assert(m == num_constraints);

// 		Eigen::Map<const variable_t > var(x);
// 		Eigen::Map<constraint_t > constraints(g);
// 		Prob::constraints::eval(param, var, constraints);

// 		return true;   	
// 	}

//    /** Method to return:
//     *   1) The structure of the jacobian (if "values" is NULL)
//     *   2) The values of the jacobian (if "values" is not NULL)
//     */
// 	bool eval_jac_g(
// 		Ipopt::Index         n,
// 		const Ipopt::Number* x,
// 		bool          new_x,
// 		Ipopt::Index         m,
// 		Ipopt::Index         nele_jac,
// 		Ipopt::Index*        iRow,
// 		Ipopt::Index*        jCol,
// 		Ipopt::Number*       values
// 	)
// 	{
// 		assert(n == num_variables);
// 		assert(m == num_constraints);

// 		if( values == NULL )
// 		{
// 			// Copy the pre-computed jacobian structure into the Ipopt variables
// 			int ind = 0;
// 		    for (int k=0; k < jac_g.outerSize(); ++k)
// 		    {
// 		        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(jac_g, k); it; ++it, ind++)
// 		        {
// 					iRow[ind] = it.row();
// 					jCol[ind] = it.col();
// 		        }
// 		    }
// 		}
// 		else
// 		{
// 			Eigen::Map<const variable_t> var(x);

// 			// Map the Ipopt values vectors into our pre-computed sparse matrix structure
// 			Eigen::Map<Eigen::SparseMatrix<scalar_t>> J(jac_g.rows(), jac_g.rows(), jac_g.nonZeros(), 
// 													    jac_g.outerIndexPtr(), jac_g.innerIndexPtr(),
// 													    values);
// 			constraint_t tmp;
// 			Prob::constraints::eval(param, var, tmp, J); 
// 		}

// 		return true;
// 	}

//    /** Method to return:
//     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
//     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
//     */
// 	bool eval_h(
// 		Ipopt::Index         n,
// 		const Ipopt::Number* x,
// 		bool          new_x,
// 		Ipopt::Number        obj_factor,
// 		Ipopt::Index         m,
// 		const Ipopt::Number* lambda,
// 		bool          new_lambda,
// 		Ipopt::Index         nele_hess,
// 		Ipopt::Index*        iRow,
// 		Ipopt::Index*        jCol,
// 		Ipopt::Number*       values
// 	)
// 	{
// 		assert(n == num_variables);
// 		assert(m == num_constraints);

// 		if( values == NULL )
// 		{
// 			// return the structure. This is a symmetric matrix, fill the lower left
// 			// triangle only.

// 			// Copy the pre-computed hessian structure into the Ipopt variables
// 			int ind = 0;
// 		    for (int k=0; k < hessian_lagrangian.outerSize(); ++k)
// 		    {
// 		        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(hessian_lagrangian, k); it; ++it)
// 		        {
// 		        	if(it.row() <= it.col())  // Store only the lower triangular part
// 		        	{
// 						iRow[ind] = it.row();
// 						jCol[ind] = it.col();
// 						ind++;
// 		        	}
// 		        }
// 		    }

// 			assert(ind == nele_hess);
// 		}
// 		else
// 		{
// 			// // return the values. This is a symmetric matrix, fill the lower left
// 			// // triangle only

// 			// // Map the Ipopt values vectors into our pre-computed sparse matrix structure
// 			// Eigen::Map<Eigen::SparseMatrix<scalar_t>> H(hessian_lagrangian.rows(), hessian_lagrangian.rows(), hessian_lagrangian.nonZeros(), 
// 			// 										    hessian_lagrangian.outerIndexPtr(), hessian_lagrangian.innerIndexPtr(),
// 			// 										    values);

// 			Eigen::Map<const variable_t> var(x);

// 			// Setup the weight vector = [obj_factor*1 lambda] where 1 is the length of the objective
// 			typename Prob::lagrangian::weight_t w;
// 			w.array() = obj_factor;
// 			w.template tail<Prob::constraints::output_size>() = Eigen::Map<const constraint_t>(lambda);

// 			variable_t tmp_grad;
// 			Prob::lagrangian::eval(param, w, var, tmp_grad, hessian_lagrangian);

// 			// Copy the lower-triangular part of the hessian
// 			int ind = 0;
// 		    for (int k=0; k < hessian_lagrangian.outerSize(); ++k)
// 		    {
// 		        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(hessian_lagrangian, k); it; ++it)
// 		        {
// 		        	if(it.row() <= it.col())  // Store only the lower triangular part
// 		        	{
// 		        		values[ind] = it.value();
// 						ind++;
// 		        	}
// 		        }
// 		    }
// 		}

// 		return true;
// 	}   

// 	// // template <class T,
// 	// //          typename std::enable_if_t<std::is_const<T>::value == true>* = nullptr>
// 	// // constexpr auto test_const(T& x) {
// 	// // 	std::cout << "In const" << std::endl;
//  // //        return Eigen::Map<const Eigen::Matrix<scalar_t, 2, 5>>((x).data());
// 	// // }

// 	// // template <class T,
// 	// //          typename std::enable_if_t<std::is_const<T>::value == false>* = nullptr>
// 	// // constexpr auto test_const(T& x) {
// 	// // 	std::cout << "In non-const" << std::endl;
//  // //        return Eigen::Map<Eigen::Matrix<scalar_t, 2, 5>>((x).data());
// 	// // }

//    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
// 	void finalize_solution(
// 		Ipopt::SolverReturn               status,
// 		Ipopt::Index                      n,
// 		const Ipopt::Number*              x,
// 		const Ipopt::Number*              z_L,
// 		const Ipopt::Number*              z_U,
// 		Ipopt::Index                      m,
// 		const Ipopt::Number*              g,
// 		const Ipopt::Number*              lambda,
// 		Ipopt::Number                     obj_value,
// 		const Ipopt::IpoptData*           ip_data,
// 		Ipopt::IpoptCalculatedQuantities* ip_cq
// 	)
// 	{
// 		sol.primal = Eigen::Map<const variable_t>(x);

// 		// TODO: Fill in the rest of the solution struct
// 	}

// private:
//    /**@name Methods to block default compiler methods.
//     *
//     * The compiler automatically generates the following three methods.
//     *  Since the default compiler implementation is generally not what
//     *  you want (for all but the most simple classes), we usually
//     *  put the declarations of these methods in the private section
//     *  and never implement them. This prevents the compiler from
//     *  implementing an incorrect "default" behavior without us
//     *  knowing. (See Scott Meyers book, "Effective C++")
//     */
//    //@{

//    NLP_Ipopt(
//       const NLP_Ipopt&
//    );

//    NLP_Ipopt& operator=(
//       const NLP_Ipopt&
//    );
//    //@}

//    // /** @name NLP data */
//    // //@{
//    // /** Number of variables */
//    // Ipopt::Index N_;
//    // /** Value of constants in constraints */
//    // Number* a_;
//    // //@}
// };

#endif
