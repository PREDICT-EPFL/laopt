#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"


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

		std::cout << "============ TEST ===========\n";
		// auto Xl = Eigen::Map<variable_t>(x_l);
		// std::cout << "x_lb = \n" << Prob::x_get_matrix(Xl) << std::endl;
		// Eigen::Matrix<scalar_t, 2, 1> tmp1 = {1,2};
		// std::cout << "Prob::x_get_matrix(Xl).rows = " << Prob::x_get_matrix(Xl).rows() << std::endl;
		// // Prob::x_get_matrix(Xl).colwise() = tmp1;
		// std::cout << "x_lb = \n" << Prob::x_get_matrix(Xl) << std::endl;

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

	// template <class T,
	//          typename std::enable_if_t<std::is_const<T>::value == true>* = nullptr>
	// constexpr auto test_const(T& x) {
	// 	std::cout << "In const" << std::endl;
 //        return Eigen::Map<const Eigen::Matrix<scalar_t, 2, 5>>((x).data());
	// }

	// template <class T,
	//          typename std::enable_if_t<std::is_const<T>::value == false>* = nullptr>
	// constexpr auto test_const(T& x) {
	// 	std::cout << "In non-const" << std::endl;
 //        return Eigen::Map<Eigen::Matrix<scalar_t, 2, 5>>((x).data());
	// }

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


		std::cout << "========== Solution =========\n";
		auto X = Eigen::Map<const variable_t>(x);

		std::cout << "x = \n" << this->x(X).transpose() << std::endl;
		std::cout << "u = \n" << this->u(X).transpose() << std::endl;
		std::cout << "xss = " << this->xss(X).transpose() << std::endl;
		std::cout << "uss = " << this->uss(X) << std::endl;

		// typename Prob::constraint_t constraints;
		// typename Prob::constraint_jacobian_t jacobian;
		// jacobian.setZero();
		// this->constraints(X, constraints, jacobian);

		// std::cout << "Constraints at optimality:\n";
		// std::cout << constraints.transpose() << std::endl;
		// std::cout << jacobian << std::endl;

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
