#ifndef LAOPT_IPOPT_INTERFACE_HPP
#define LAOPT_IPOPT_INTERFACE_HPP

#include <Eigen/Dense>

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"

namespace laopt
{

template<typename UserProblem>
class Solver_IPOpt: public Ipopt::TNLP
{
public:
    using scalar_t = typename UserProblem::scalar_t;

private:
    UserProblem& prob;

	// IPOPT only wants the lower triangular part of the hessian
	// So we need to keep a full buffer for the computation
	Eigen::SparseMatrix<scalar_t> lag_hessian;

public:
	// internal storage for primal and dual variables
	Eigen::VectorX<scalar_t> primal;
    Eigen::VectorX<scalar_t> dual_lb;
    Eigen::VectorX<scalar_t> dual_ub;
    Eigen::VectorX<scalar_t> dual;

    /* Constructor. */
	explicit Solver_IPOpt(UserProblem& prob) :
        prob(prob),
        primal(prob.num_variables()),
        dual_lb(prob.num_variables()),
        dual_ub(prob.num_variables()),
        dual(prob.constraints.rows())
	{
        // Default to zero if user doesn't set
        primal.setZero();
        dual_lb.setZero();
        dual_ub.setZero();
        dual.setZero();

		prob.lagrangian.hessian.allocate_memory(lag_hessian);

        // set memory of decision variables
        prob.set_decision_variable(primal);
	}

	bool get_nlp_info(
	   Ipopt::Index&   n,
       Ipopt::Index&   m,
       Ipopt::Index&   nnz_jac_g,
       Ipopt::Index&   nnz_h_lag,
       IndexStyleEnum& index_style
	) override
	{
		n = prob.num_variables();
		m = prob.constraints.rows();

		// nonzeros in the jacobian of the constraints
		nnz_jac_g = prob.constraints.jacobian.get_sparsity_structure().nonZeros();

		// nonzeros in the lower-triangular part of the hessian of the lagrangian
		// Iterate over the non-zeros, counting only those in the lower triangular part
		int nnz = 0;
		auto S = prob.lagrangian.hessian.get_sparsity_structure();
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
	   Ipopt::Index   n,
	   Ipopt::Number* x_l,
	   Ipopt::Number* x_u,
	   Ipopt::Index   m,
	   Ipopt::Number* g_l,
	   Ipopt::Number* g_u
	) override
	{
		// Compute bounds on the variables
		prob.eval_variable_bounds(Eigen::Map<Eigen::VectorX<scalar_t>>(x_l, n),
								  Eigen::Map<Eigen::VectorX<scalar_t>>(x_u, n));

		// Compute the constraints with a temporary variable so that we can get the bounds
		assert(m == prob.constraints.rows() && "Number of constraints does not match ipopt's number of constraints");
		Eigen::VectorX<scalar_t> con(m);
		Eigen::Map<Eigen::VectorX<scalar_t>> lb(g_l, m);
		Eigen::Map<Eigen::VectorX<scalar_t>> ub(g_u, m);

		prob.eval_constraints(laopt::Eval(), con, lb, ub);

		return true;
	}

	bool get_starting_point(
	   Ipopt::Index   n,
	   bool           init_x,
	   Ipopt::Number* x,
	   bool           init_z,
	   Ipopt::Number* z_L,
	   Ipopt::Number* z_U,
	   Ipopt::Index   m,
	   bool           init_lambda,
	   Ipopt::Number* lambda
	) override
	{
		// Initialize to user-specified value
		if (init_x)
        {
            Eigen::Map<Eigen::VectorX<scalar_t>>(x, n) = primal;
        }

		if (init_lambda)
        {
            Eigen::Map<Eigen::VectorX<scalar_t>>(lambda, m) = dual;
        }
		return true;
	}

	bool eval_f(
       Ipopt::Index         n,
	   const Ipopt::Number* x,
	   bool                 new_x,
	   Ipopt::Number&       obj_value
	) override
	{
		Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Ipopt::Number*>(x), n);

        prob.set_decision_variable(var);
		obj_value = prob.eval_objective(laopt::Eval());
        prob.set_decision_variable(primal); // set memory back to internal primal variable since var might be deallocated

        return true;
	}

	bool eval_grad_f(
	   Ipopt::Index         n,
	   const Ipopt::Number* x,
	   bool                 new_x,
	   Ipopt::Number*       grad_f
	) override
	{
		Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Ipopt::Number*>(x), n);
		Eigen::Map<Eigen::VectorX<scalar_t>> grad(grad_f, n);

        prob.set_decision_variable(var);
		prob.eval_objective(laopt::Gradient(), grad);
        prob.set_decision_variable(primal); // set memory back to internal primal variable since var might be deallocated

        return true;
	}

	bool eval_g(
	   Ipopt::Index         n,
	   const Ipopt::Number* x,
	   bool                 new_x,
	   Ipopt::Index         m,
	   Ipopt::Number*       g
	) override
	{
		Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Ipopt::Number*>(x), n);
		Eigen::Map<Eigen::VectorX<scalar_t>> constraints(g, m);
		Eigen::VectorX<scalar_t> lb(m);
		Eigen::VectorX<scalar_t> ub(m);

        prob.set_decision_variable(var);
		prob.eval_constraints(laopt::Eval(), constraints, lb, ub);
        prob.set_decision_variable(primal); // set memory back to internal primal variable since var might be deallocated

        return true;
	}

	bool eval_jac_g(
	   Ipopt::Index         n,
	   const Ipopt::Number* x,
	   bool                 new_x,
	   Ipopt::Index         m,
	   Ipopt::Index         nele_jac,
	   Ipopt::Index*        iRow,
	   Ipopt::Index*        jCol,
	   Ipopt::Number*       values
	) override
	{
		assert(nele_jac == prob.constraints.jacobian.get_sparsity_structure().nonZeros() && "Number of nonzeros is wrong for the jacobian");

		if (values == nullptr)
		{
			// return the structure of the jacobian of the constraints
			auto S = prob.constraints.jacobian.get_sparsity_structure();
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
			Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Ipopt::Number*>(x), n);
			Eigen::VectorX<scalar_t> constraints(m); // Value of the constraints (ignored)
			Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer(values, nele_jac); // Jacobian non-zeros
			// Eigen::VectorX<scalar_t> jacobian_buffer(nele_jac);

			Eigen::VectorX<scalar_t> lb(m); // Upper/lower bounds (ignored)
			Eigen::VectorX<scalar_t> ub(m);
            prob.set_decision_variable(var);
			prob.eval_constraints(laopt::Jacobian(), constraints, lb, ub, jacobian_buffer);
            prob.set_decision_variable(primal); // set memory back to internal primal variable since var might be deallocated
	   }

	   return true;
	}

	/**
	 * Compute the hessian of the lagrangian
	 */
	bool eval_h(
	   Ipopt::Index         n,
	   const Ipopt::Number* x,
	   bool                 new_x,
	   Ipopt::Number        obj_factor,
	   Ipopt::Index         m,
	   const Ipopt::Number* lambda,
	   bool                 new_lambda,
	   Ipopt::Index         nele_hess,
	   Ipopt::Index*        iRow,
	   Ipopt::Index*        jCol,
	   Ipopt::Number*       values
	) override
	{
		if (values == nullptr)
		{
			// Return the structure of the hessian of the lagrangian.
			// This is a symmetric matrix, fill the lower left triangle only.

			auto S = prob.lagrangian.hessian.get_sparsity_structure();
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
			Eigen::Map<Eigen::VectorX<scalar_t>> var(const_cast<Ipopt::Number*>(x), n);
            Eigen::Map<Eigen::VectorX<scalar_t>> dual_var(const_cast<Ipopt::Number*>(lambda), prob.constraints.rows());
			Eigen::VectorX<scalar_t> gradient(n); // Ignored


            prob.set_decision_variable(var);
            prob.eval_lagrangian(laopt::Hessian(), obj_factor, dual_var, gradient, lag_hessian);
            prob.set_decision_variable(primal); // set memory back to internal primal variable since var might be deallocated

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
	) override
	{
		primal = Eigen::Map<const Eigen::VectorX<Ipopt::Number>>(x, n);
		dual = Eigen::Map<const Eigen::VectorX<Ipopt::Number>>(lambda, m);
	}
};

}

#endif // LAOPT_IPOPT_INTERFACE_HPP
