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


// template<int num_, typename Con_t>
// struct ConArray_t
// {
// 	static constexpr int num = num_;
// 	std::array<Con_t, num> cons;
// };



// template<typename scalar_t, int len_, int num_ = 1>
// struct Con_t
// {
// 	static constexpr int len = len_;  // Length of one constraint
// 	static constexpr int num_cons = num_;  // Number of constraints in this set

// 	const int offset;  // Offset into the global constraint
// 	const int index;   // Index into the global constraint

// 	// Offset and index of the next constraint after this one
// 	const int next_offset;
// 	const int next_index;

// 	// We want to be able to compute a dense representation of the jacobian
// 	// for each variable in this constraint.
// 	// 

// 	// J_offsets(row, col) is the offset into the 
// 	// jacobian.Values() vector where the jacobian for variable (row)
// 	// starts for constraint (col)
// 	// Eigen::Matrix<int, len, num> J_offsets;  

// 	// J_stride(row, col) is the offset into the 
// 	// jacobian.Values() vector where the jacobian for variable (row)
// 	// starts for constraint (col)
// 	// Eigen::Matrix<int, len, num> J_stride;  

//     template<typename Con>
//     EIGEN_STRONG_INLINE constexpr auto operator()(Con& con, int ind) const
//     {
//         return con.template segment<len>(offset + ind * len);
//     }

//     template<typename Con>
//     EIGEN_STRONG_INLINE constexpr auto operator()(Con& con) const
//     {
//     	using target = Eigen::Matrix<typename Con::Scalar, len, num_cons>;
//         return _map_matrix<target>(con.template segment<len * num_cons>(offset).data());
//     }

//     template<typename Prev>
// 	constexpr Con_t(const Prev& prev): 
// 		  offset(prev.next_offset), 
// 		  index(prev.next_index),
// 		  next_offset(offset + len * num_cons),
// 		  next_index(index + num_cons)
// 	{}

// 	constexpr Con_t(): 
// 		  offset(0),
// 		  index(0),
// 		  next_offset(offset + len * num_cons),
// 		  next_index(index + num_cons)
// 	{}
// };


// template<typename scalar_t, int len_, int num_ = 1>
// struct Var_t
// {
// 	static constexpr int len = len_;  // Length of one variable
// 	static constexpr int num_vars = num_;  // Number of variables in this set

// 	const int offset;  // Offset into the global variable
// 	const int index;   // Index into the global variable

// 	// Offset and index of the next variable after this one
// 	const int next_offset;
// 	const int next_index;

//     template<typename Var>
//     EIGEN_STRONG_INLINE constexpr auto operator()(Var& var, int ind) const
//     {
//         return var.template segment<len>(offset + ind * len);
//     }

//     template<typename Var>
//     EIGEN_STRONG_INLINE constexpr auto operator()(Var& var) const
//     {
//     	using target = Eigen::Matrix<typename Var::Scalar, len, num_vars>;
//         return _map_matrix<target>(var.template segment<len * num_vars>(offset).data());
//     }

// 	// offset and index of this variable in the global variable
// 	// offset and index are incremented upon return
// 	// Var_t(int& offset_, int& index_): 
// 	// 	  offset(offset_), index(index_) 
// 	// {
// 	// 	offset_ += len * num;
// 	// 	index_ += num;
// 	// }

//     template<typename Prev>
// 	constexpr Var_t(const Prev& prev): 
// 		  offset(prev.next_offset), 
// 		  index(prev.next_index),
// 		  next_offset(offset + len * num_vars),
// 		  next_index(index + num_vars)
// 	{}

// 	constexpr Var_t(): 
// 		  offset(0),
// 		  index(0),
// 		  next_offset(offset + len * num_vars),
// 		  next_index(index + num_vars)
// 	{}
// };


// template<typename... VarTypes>
// constexpr auto sum_length()
// {
//     int result = 0;
//     for(auto s : { VarTypes::total_len... }) result += s;
//     return result;
// }


// template<typename scalar_t>
// struct Blah_t
// {
// 	// template<int len_>
// 	// struct Var_t
// 	// {
// 	// 	static constexpr auto len = len_;
// 	// 	static constexpr auto total_len = len;
// 	// };

// 	template<int len_, int num_ = 1>
// 	struct Var_t
// 	{
// 		static constexpr auto len = len_;
// 		static constexpr auto num = num_;

// 		static constexpr auto total_len = len * num;

// 		static constexpr auto offset = 
// 	};


// 	template<int len, // Number of outputs of the function
// 			 typename... input_t> // Types of the inputs (Var_t)
// 	struct Con_t
// 	{
// 		static constexpr auto num_inputs = sizeof...(input_t);

// 		static constexpr std::array<int, num_inputs> input_sizes = {input_t::len...};

// 		EIGEN_STRONG_INLINE auto operator()(Eigen::Ref<constraint_t> c)
// 		{
// 			return c.template segment<len>(offset);
// 		}

// 		template<typename constraint_t, int var_num>
// 		EIGEN_STRONG_INLINE auto J(Eigen::Ref<constraint_jacobian_t> jac)
// 		{
// 			jac.template block<len, input_sizes[var_num]>(offset);
// 		}


// 	private:
// 		int offset;
// 	}


// 	using x_t   = Var_t<3, 3>;
// 	using u_t   = Var_t<x_t, 2, 2>;
// 	using xss_t = Var_t<u_t, 3>;
// 	using uss_t = Var_t<xss_t, 2>;

// 	x_t x;
// 	u_t u;
// 	xss_t xss;
// 	uss_t uss;

// 	// ConArray_t<3, 3> x;
// 	// ConArray_t<2, 2> u;
// 	// Con_t<3> xss;
// 	// Con_t<2> uss;

// 	static constexpr int var_length = sum_length<x_t, u_t, xss_t, uss_t>();
// 	static constexpr int con_length = sum_length<eq1_t, eq2_t, eq3_t>();

// 	// NLP variable types
// 	using variable_t            = Matrix<scalar_t, var_length, 1>;
// 	using constraint_t          = Matrix<scalar_t, con_length, 1>;
// 	using constraint_jacobian_t = Matrix<scalar_t, con_length,  var_length>;

// 	void test()
// 	{
// 		Eigen::Matrix<double, x_t::len, 1> y;
// 		std::cout << "y = " << y << std::endl;
// 	}

// 	// Blah_t()
// 	// {
// 	// 	set_variable_order(x, u, xss, uss);
// 	// 	set_constraint_order()
// 	// };

// 	// static constexpr auto x   = Var_t<scalar_t, 3, 3>();
// 	// static constexpr auto u   = Var_t<scalar_t, 2, 2>(x);
// 	// static constexpr auto xss = Var_t<scalar_t, 3>(u);
// 	// static constexpr auto uss = Var_t<scalar_t, 2>(xss);

// 	// static const int var_length = uss.next_offset;  // Total length of all variables
// 	// static const int num_vars   = uss.next_index;   // Number of vector variables

// 	// static constexpr auto eq1 = Con_t<scalar_t, 2>();
// 	// static constexpr auto eq2 = Con_t<scalar_t, 2, 5>(eq1);
// 	// static constexpr auto eq3 = Con_t<scalar_t, 3>(eq2);

// 	// Var_t<scalar_t, 4> t;

// 	// std::array<int, decltype(t)::len> bob;

// 	// ConArray_t<4, Con_t<scalar_t, 2>>();

// 	// eq1(con) = func1(u(v_, 0), x(v_, 1), eq1(J, 0), eq1(J, 1));
// 	// eq1(con) = func1(u(v_, 0), x(v_, 1));

// };

// template<typename T>
// EIGEN_STRONG_INLINE void func(
// 	const Eigen::Ref<const Eigen::Matrix<T, 2, 1>>& x,
// 	const Eigen::Ref<const Eigen::Matrix<T, 1, 1>>& u,
// 	Eigen::Ref<Eigen::Matrix<T, 2, 1>> out) noexcept
// {
// 	out = x;
// }


int main(void)
{
	// using scalar_t = double;

	// Blah_t<scalar_t> blah;

	// blah.test();

	// std::cout << "x   = " << blah.x.offset << ", " << blah.x.index << std::endl;
	// std::cout << "u   = " << blah.u.offset << ", " << blah.u.index << std::endl;
	// std::cout << "xss = " << blah.xss.offset << ", " << blah.xss.index << std::endl;
	// std::cout << "uss = " << blah.uss.offset << ", " << blah.uss.index << std::endl;
	// std::cout << std::endl;
	// std::cout << "var_length = " << blah.var_length << std::endl;
	// std::cout << "num_vars = " << blah.num_vars << std::endl;
	// std::cout << std::endl;

	// Eigen::Matrix<scalar_t, blah.var_length, 1> var;
	// for(int i=0; i<blah.var_length; i++) 
	// 	var[i] = i;
	// std::cout << "var = " << var.transpose() << std::endl;

	// for(int i=0; i<blah.x.num_vars; i++)
	// 	std::cout << "x(" << i << ") = " << blah.x(var, i).transpose() << std::endl;
	// std::cout << std::endl;
	// std::cout << "x() = \n" << blah.x(var) << std::endl;

	// for(int i=0; i<blah.u.num_vars; i++)
	// 	std::cout << "u(" << i << ") = " << blah.u(var, i).transpose() << std::endl;
	// std::cout << std::endl;
	// std::cout << "u() = \n" << blah.u(var) << std::endl;



	using scalar_t = double;
	using OPT = LOpt<scalar_t>;
	OPT opt;

	// Eigen::Matrix<scalar_t, OPT::NUM_VARS, 1> var;
	// for(int i=0; i<OPT::NUM_VARS; i++) 
	// 	var[i] = i;
	// std::cout << "var = " << var.transpose() << std::endl;

	// SparseMatrix<scalar_t> J(opt.NUM_CON, opt.NUM_VARS);
	// opt.constraints_sparse_initialize(J);
	// std::cout << "J = \n" << J << std::endl;


#if 0
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
   mynlp->x0 = {0,1};
   std::cout << "x0 = " << mynlp->x0 << std::endl;
   status = app->OptimizeTNLP(mynlp);

   std::cout << "A = \n" << mynlp->A << std::endl;
   std::cout << "B = \n" << mynlp->B << std::endl;



   // mynlp->q = {10,100000};

   // std::cout << "x_initial = " << mynlp->x_initial << std::endl;
   // std::cout << "q = " << mynlp->q << std::endl;   
   // status = app->OptimizeTNLP(mynlp);

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
