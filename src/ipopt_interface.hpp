#ifndef IPOPT_INTERFACE_HPP
#define IPOPT_INTERFACE_HPP

#include "Eigen/Dense"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"

using namespace Ipopt;

namespace lampc
{

template<typename UserProblem>
struct Solver_IPOpt: public TNLP
{
	UserProblem& prob;
	using scalar_t = typename UserProblem::scalar_t;

	// IPOPT only wants the lower triangular part of the hessian
	// So we need to keep a full buffer for the computation
	Eigen::SparseMatrix<scalar_t> lag_hessian;

public:

	// User-set initial value for the primal variable
	Eigen::VectorX<scalar_t> init_primal;
	Eigen::VectorX<scalar_t> init_dual;

	// User-set initial value for the primal variable
	Eigen::VectorX<scalar_t> sol_primal;
	Eigen::VectorX<scalar_t> sol_dual;    // Dual variables
	Eigen::VectorX<scalar_t> sol_dual_x;  // Dual variables for bound constraints

	// TODO: Add initial values for the dual and variable bound duals

	/* Constructor. */
	explicit Solver_IPOpt(UserProblem& prob) :
        prob(prob),
        init_primal(prob.num_variables()),
        init_dual(prob.constraints.rows()),
        sol_primal(prob.num_variables())
	{ 
		init_primal.setZero(); // Default to zero if user doesn't set
		init_dual.setZero();

		prob.lagrangian.hessian.allocate_memory(lag_hessian);
	}

	bool get_nlp_info(
	   Index&          n,
	   Index&          m,
	   Index&          nnz_jac_g,
	   Index&          nnz_h_lag,
	   IndexStyleEnum& index_style
	) override
	{
		n = prob.num_variables();
		m = prob.constraints.rows();

		// nonzeros in the jacobian of the constraints
		nnz_jac_g = prob.constraints.jacobian.sparsity_structure.nonZeros();

		// nonzeros in the lower-triangular part of the hessian of the lagrangian
		// Iterate over the non-zeros, counting only those in the lower triangular part
		int nnz = 0;
		auto S = prob.lagrangian.hessian.sparsity_structure;
		for (int col = 0; col < S.outerSize(); ++col)
        {
            for (typename Eigen::SparseMatrix<typename decltype(S)::Scalar>::InnerIterator it(S, col); it; ++it)
            {
                if(it.row() >= it.col()) {
                    nnz++;
                }
            }
        }

		nnz_h_lag = nnz;

		index_style = C_STYLE; // 0-based indexing

		return true;
	}

	bool get_bounds_info(
	   Index   n,
	   Number* x_l,
	   Number* x_u,
	   Index   m,
	   Number* g_l,
	   Number* g_u
	) override
	{
		// Compute bounds on the variables
		prob.eval_variable_bounds(Eigen::Map<Eigen::VectorX<scalar_t>>(x_l,n), 
								  Eigen::Map<Eigen::VectorX<scalar_t>>(x_u,n));

		// Compute the constraints with a temporary variable so that we can get the bounds
		assert(m == prob.constraints.rows() && "Number of constraints does not match ipopt's number of constraints");
		Eigen::VectorX<scalar_t> con(m);
		Eigen::Map<Eigen::VectorX<scalar_t>> lb(g_l, m);
		Eigen::Map<Eigen::VectorX<scalar_t>> ub(g_u, m);

		Eigen::VectorX<scalar_t> var(prob.num_variables());
		var.array() = 0;

		prob.eval_constraints(lampc::Eval(), var, con,lb,ub);

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
	) override
	{
		// Initialize to user-specified value
		if (init_x)
        {
            Eigen::Map<Eigen::VectorX<scalar_t>>(x, n) = init_primal;
        }

		if (init_lambda)
        {
            Eigen::Map<Eigen::VectorX<scalar_t>>(lambda, m) = init_dual;
        }
		return true;
	}
 
	bool eval_f(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Number&       obj_value
	) override
	{
		Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Number*>(x), n);
		obj_value = prob.eval_objective(lampc::Eval(), var);
		return true;
	}

	bool eval_grad_f(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Number*       grad_f
	) override
	{
		Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Number*>(x), n);
		Eigen::Map<Eigen::VectorX<scalar_t>> grad(grad_f, n);
		prob.eval_objective(lampc::Gradient(), var, grad);
		return true;
	}

	bool eval_g(
	   Index         n,
	   const Number* x,
	   bool          new_x,
	   Index         m,
	   Number*       g
	) override
	{
		Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Number*>(x), n);
		Eigen::Map<Eigen::VectorX<scalar_t>> constraints(g, m);
		Eigen::VectorX<scalar_t> lb(m);
		Eigen::VectorX<scalar_t> ub(m);
		prob.eval_constraints(lampc::Eval(), var, constraints, lb, ub);

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
	) override
	{
		assert(nele_jac == prob.constraints.jacobian.sparsity_structure.nonZeros() && "Number of nonzeros is wrong for the jacobian");

		if (values == nullptr)
		{
			// return the structure of the jacobian of the constraints
			auto S = prob.constraints.jacobian.sparsity_structure;
			int i = 0;
			for (int col = 0; col < S.outerSize(); ++col)
            {
                for (typename Eigen::SparseMatrix<typename decltype(S)::Scalar>::InnerIterator it(S,col); it; ++it)
                {
                    iRow[i] = it.row();
                    jCol[i] = it.col();
                    i++;
                }
            }

			assert(i == nele_jac &&  "Number of non-zeros in the jacobian is wrong");
		}
		else
		{
			// return the values of the jacobian of the constraints in the same order as the sparsity was defined
			Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Number*>(x), n);
			Eigen::VectorX<scalar_t> constraints(m); // Value of the constraints (ignored)
			Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer(values, nele_jac); // Jacobian non-zeros
			// Eigen::VectorX<scalar_t> jacobian_buffer(nele_jac);

			Eigen::VectorX<scalar_t> lb(m); // Upper/lower bounds (ignored)
			Eigen::VectorX<scalar_t> ub(m);
			prob.eval_constraints(lampc::Jacobian(), var, constraints, lb, ub, jacobian_buffer);
	   }

	   return true;
	}

	/**
	 * Compute the hessian of the lagrangian
	 */
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
	) override
	{
		if (values == nullptr)
		{
			// Return the structure of the hessian of the lagrangian. 
			// This is a symmetric matrix, fill the lower left triangle only.

			auto S = prob.lagrangian.hessian.sparsity_structure;
			int i = 0;
			for (int col = 0; col < S.outerSize(); ++col)
            {
                for (typename Eigen::SparseMatrix<typename decltype(S)::Scalar>::InnerIterator it(S, col); it; ++it)
                {
                    if( it.row() >= it.col() )
                    {
                        iRow[i] = it.row();
                        jCol[i] = it.col();
                        i++;
                    }
                }
            }

			assert(i == nele_hess &&  "Number of non-zeros in the hessian is wrong");
		}
		else
		{
			// return the values of the hessian of the lagrangian in the same order as the sparsity was defined
			Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Number*>(x), n);

			Eigen::VectorX<scalar_t> gradient(n); // Ignored

			auto dual = Eigen::Map<Eigen::VectorX<scalar_t>>(const_cast<Number*>(lambda), prob.constraints.rows());
			prob.eval_lagrangian(lampc::Hessian(), var, obj_factor, dual, gradient, lag_hessian);

			// Copy the lower triangular part into the ipopt buffer
			// auto S = prob.lagrangian.hessian.sparsity_structure;
			auto S = lag_hessian;
			int i = 0;
			for (int col = 0; col < S.outerSize(); ++col)
            {
                for (typename Eigen::SparseMatrix<typename decltype(S)::Scalar>::InnerIterator it(S, col); it; ++it)
                {
                    if( it.row() >= it.col() )
                    {
                        values[i] = it.value();
                        i++;
                    }
                }
            }
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
	) override
	{
		sol_primal = Eigen::Map<const Eigen::VectorX<Number>>(x, n);
		prob.set_decision_variable(sol_primal);

		sol_dual = Eigen::Map<const Eigen::VectorX<Number>>(lambda, m);
	}
};

}

#endif
