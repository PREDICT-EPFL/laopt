#ifndef IPOPT_INTERFACE_HPP
#define IPOPT_INTERFACE_HPP

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"

// using namespace Ipopt;

template<typename Prob>
class NLP_Ipopt : public Ipopt::TNLP, Prob
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	using scalar_t = typename Prob::scalar_t;
	using param_t = typename Prob::param_t;
	using variable_t = typename Prob::variable_vec;
	using constraint_t = typename Prob::constraints_vec;
	using constraint_jacobian_t = typename Prob::constraints_jacobian_mat;
	using obj_gradient_t = typename Prob::obj_gradient_vec;
	using obj_hessian_t = typename Prob::obj_hessian_mat;
	using obj_t = typename Prob::obj_vec;

	static constexpr std::size_t num_variables = Prob::num_variables;
	static constexpr std::size_t num_constraints = Prob::constraints_t::num_constraints;

	variable_t x0;  // Initial iterate

	// Parameter struct - owned outside this class
	param_t& param;

	// Sparsity structures
	Eigen::SparseMatrix<scalar_t> jac_g;
	Eigen::SparseMatrix<scalar_t> hessian_lagrangian;

	struct sol_t
	{
		variable_t primal;
		constraint_t dual_g;
		variable_t dual_x;
		obj_t obj;
		// obj_gradient_t obj_gradient;
		// obj_hessian_t obj_hessian;
	};
	sol_t sol;


	NLP_Ipopt(param_t& param_) :
		x0(),
		param(param_),
		jac_g(num_constraints, num_variables),
		hessian_lagrangian(num_variables, num_variables)
	{
        this->constraints.initialize_sparse_jacobian(jac_g);
        this->lagrangian.initialize_sparse_hessian(hessian_lagrangian);
		x0.array() = 0.0;
	};

	bool get_nlp_info(
		Ipopt::Index&          n,
		Ipopt::Index&          m,
		Ipopt::Index&          nnz_jac_g,
		Ipopt::Index&          nnz_h_lag,
		IndexStyleEnum&        index_style
	)
	{
		n = num_variables;
		m = num_constraints;
		nnz_jac_g = this->constraints.nnz_jacobian;
		nnz_h_lag = this->lagrangian.nnz_hessian();
		index_style = TNLP::C_STYLE;


		// Compute nnz in the lower-triangular part of the hessian
		int ind = 0;
	    for (int k=0; k < hessian_lagrangian.outerSize(); ++k)
	    {
	        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(hessian_lagrangian, k); it; ++it)
	        {
	        	if(it.row() >= it.col())  // Store only the lower triangular part
	        	{
					// iRow[ind] = it.row();
					// jCol[ind] = it.col();
					ind++;
	        	}
	        }
	    }
		nnz_h_lag = ind;

		std::cout << "nnz_h_lag = " << nnz_h_lag << std::endl;
		return true;
	};

	bool get_bounds_info(
		Ipopt::Index   n,
		Ipopt::Number* x_l,
		Ipopt::Number* x_u,
		Ipopt::Index   m,
		Ipopt::Number* g_l,
		Ipopt::Number* g_u
	)
	{
		// n and m are whatever we set in get_nlp_info
		assert(n == num_variables);
		assert(m == num_constraints);

		this->variables.get_bounds(param, Eigen::Map<variable_t>(x_l), Eigen::Map<variable_t>(x_u));
		this->constraints.get_bounds(param, Eigen::Map<constraint_t>(g_l), Eigen::Map<constraint_t>(g_u));

		return true;	
	}

	bool get_starting_point(
		Ipopt::Index   n,
		bool    init_x,
		Ipopt::Number* x,
		bool    init_z,
		Ipopt::Number* z_L,
		Ipopt::Number* z_U,
		Ipopt::Index   m,
		bool    init_lambda,
		Ipopt::Number* lambda
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
		const Ipopt::Number* x,
		bool          new_x,
		Ipopt::Number&       obj_value
	)
	{
		assert(n == num_variables);		
		obj_value = this->objective(param, Eigen::Map<const variable_t>(x));
		return true;
	}

	bool eval_grad_f(
		Ipopt::Index         n,
		const Ipopt::Number* x,
		bool          new_x,
		Ipopt::Number*       grad_f
	)
	{
		assert(n == num_variables);
		this->objective(param, Eigen::Map<const variable_t>(x), Eigen::Map<obj_gradient_t>(grad_f));
		return true;
	}

	bool eval_g(
		Ipopt::Index         n,
		const Ipopt::Number* x,
		bool          new_x,
		Ipopt::Index         m,
		Ipopt::Number*       g
	)
	{
		assert(n == num_variables);
		assert(m == num_constraints);

		Eigen::Map<const variable_t > var(x);
		Eigen::Map<constraint_t > constraints(g);
		this->constraints(param, var, constraints);

		return true;   	
	}

   /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
	bool eval_jac_g(
		Ipopt::Index         n,
		const Ipopt::Number* x,
		bool          new_x,
		Ipopt::Index         m,
		Ipopt::Index         nele_jac,
		Ipopt::Index*        iRow,
		Ipopt::Index*        jCol,
		Ipopt::Number*       values
	)
	{
		assert(n == num_variables);
		assert(m == num_constraints);

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
			this->constraints(param, var, tmp, J); 
		}

		return true;
	}

   /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
	bool eval_h(
		Ipopt::Index         n,
		const Ipopt::Number* x,
		bool          new_x,
		Ipopt::Number        obj_factor,
		Ipopt::Index         m,
		const Ipopt::Number* lambda,
		bool          new_lambda,
		Ipopt::Index         nele_hess,
		Ipopt::Index*        iRow,
		Ipopt::Index*        jCol,
		Ipopt::Number*       values
	)
	{
		// std::cout << "Evaluating hessian\n";

		assert(n == num_variables);
		assert(m == num_constraints);

		if( values == NULL )
		{
			// return the structure. This is a symmetric matrix, fill the lower left
			// triangle only.

			// Copy the pre-computed hessian structure into the Ipopt variables
			int ind = 0;
		    for (int k=0; k < hessian_lagrangian.outerSize(); ++k)
		    {
		        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(hessian_lagrangian, k); it; ++it)
		        {
		        	if(it.row() <= it.col())  // Store only the lower triangular part
		        	{
						iRow[ind] = it.row();
						jCol[ind] = it.col();
						ind++;

		        	}
		        }
		    }

			assert(ind == nele_hess);
		}
		else
		{
			// // return the values. This is a symmetric matrix, fill the lower left
			// // triangle only

			// // Map the Ipopt values vectors into our pre-computed sparse matrix structure
			// Eigen::Map<Eigen::SparseMatrix<scalar_t>> H(hessian_lagrangian.rows(), hessian_lagrangian.rows(), hessian_lagrangian.nonZeros(), 
			// 										    hessian_lagrangian.outerIndexPtr(), hessian_lagrangian.innerIndexPtr(),
			// 										    values);

			Eigen::Map<const variable_t> var(x);
			Eigen::Map<const typename Prob::equalities_vec> dual_eq(lambda);
			Eigen::Map<const typename Prob::inequalities_vec> dual_ineq(lambda + Prob::num_equalities);
			// Eigen::Map<const constraint_t> lam(lambda);

			variable_t dual_var = variable_t::Ones();

			variable_t tmp_grad;
		    this->lagrangian(param, var, obj_factor,
		    							 dual_eq,
		                                 dual_ineq,
		                                 dual_var,
		                                 tmp_grad, hessian_lagrangian);

			// Copy the lower-triangular part of the hessian
			int ind = 0;
		    for (int k=0; k < hessian_lagrangian.outerSize(); ++k)
		    {
		        for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(hessian_lagrangian, k); it; ++it)
		        {
		        	if(it.row() <= it.col())  // Store only the lower triangular part
		        	{
		        		values[ind] = it.value();
						ind++;
		        	}
		        }
		    }
		}

		return true;
	}   

	// // template <class T,
	// //          typename std::enable_if_t<std::is_const<T>::value == true>* = nullptr>
	// // constexpr auto test_const(T& x) {
	// // 	std::cout << "In const" << std::endl;
 // //        return Eigen::Map<const Eigen::Matrix<scalar_t, 2, 5>>((x).data());
	// // }

	// // template <class T,
	// //          typename std::enable_if_t<std::is_const<T>::value == false>* = nullptr>
	// // constexpr auto test_const(T& x) {
	// // 	std::cout << "In non-const" << std::endl;
 // //        return Eigen::Map<Eigen::Matrix<scalar_t, 2, 5>>((x).data());
	// // }

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
	void finalize_solution(
		Ipopt::SolverReturn               status,
		Ipopt::Index                      n,
		const Ipopt::Number*              x,
		const Ipopt::Number*              z_L,
		const Ipopt::Number*              z_U,
		Ipopt::Index                      m,
		const Ipopt::Number*              g,
		const Ipopt::Number*              lambda,
		Ipopt::Number                     obj_value,
		const Ipopt::IpoptData*           ip_data,
		Ipopt::IpoptCalculatedQuantities* ip_cq
	)
	{
		sol.primal = Eigen::Map<const variable_t>(x);

		// TODO: Fill in the rest of the solution struct
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

#endif
