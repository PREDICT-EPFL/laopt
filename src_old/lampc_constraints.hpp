#ifndef __LAMPC__CONSTRAINTS_HPP
#define __LAMPC__CONSTRAINTS_HPP

#include <iostream>

namespace lampc
{


template <typename Derived> 
struct traits;


// Defines set of constraints
template<int size_>
struct Variable
{
	static constexpr int size = size_;

	// using type = Eigen::Vector<Scalar, size>;
	// using ref = Eigen::Ref<Eigen::Vector<Scalar, size>>;
};


template<typename Scalar_, typename... Variables>
struct VariableSet
{
	using Scalar = Scalar_;
	static constexpr int num_vars = meta::sum_template<Variables::size...>();
	using variable_t = Eigen::Vector<Scalar, num_vars>;

	/** Return the offset to variable var
	 */
	template<typename var>
	static constexpr inline int offset()
	{
		int acc = 0;
		bool found = false;
        auto l = { (
        	found = found || std::is_same<var, Variables>::value,
        	acc += found ? 0 : Variables::size,
        	0)... };
        return acc;
	}

	/** Return the segment for variable var from the given vector
	 */
	template<typename var>
	static inline Eigen::Ref<Eigen::Vector<Scalar, var::size>>
		get(Eigen::Ref<Eigen::Vector<Scalar, num_vars>> in)
	{
		return in.template segment<var::size>(offset<var>());
	};
};

template<typename Derived, typename... Variables>
struct Constraint
{
	static const std::size_t size = traits<Derived>::size;
	static constexpr std::size_t num_vars = meta::sum_template<Variables::size...>();

	// Upper bound on number of non-zeros
	static constexpr std::size_t nonZerosAtCompileTime = size * meta::sum_template<Variables::size...>();

	/** Append all offsets required to evaluate the jacobian to the vector
	 */
	template<typename VarSet, typename Scalar = typename VarSet::Scalar>
	static void jacobian_get_offsets(int row, Eigen::SparseMatrix<Scalar>& indexJ, std::vector<std::size_t>& offsets)
	{
		Derived::template jacobian_get_offsets<VarSet>(row, indexJ, offsets);
	}

	/** Append all offsets required to evaluate the hessian to the vector
	 */
	template<typename VarSet, typename Scalar = typename VarSet::Scalar>
	static void hessian_get_offsets(Eigen::SparseMatrix<Scalar>& indexH, std::vector<std::size_t>& offsets)
	{
		Derived::template hessian_get_offsets<VarSet>(indexH, offsets);
	}

	// static void eval(Eigen::Ref<variable_t> x)
	// {
	// 	Base::eval(x);
	// }

	// static std::size_t nonZeros()
	// {
	// 	// returns num vars * size
	// }

};

template<typename Function, typename... Variables>
struct FunctionConstraint
 : public Constraint<FunctionConstraint<Function, Variables...>, Variables...>
{
	static constexpr int size = Function::num_outputs;
	static constexpr int num_vars = meta::sum_template<Variables::size...>();

	template<typename VarSet, typename param_t>
	static void eval(const param_t& param, 
		             Eigen::Ref<typename VarSet::variable_t> x, 
		             Eigen::Ref<Eigen::Vector<typename VarSet::Scalar, size>> out)
	{
		out = Function::eval(param, VarSet::template get<Variables>(x)...);
	}

	/** Compute the jacobian
	 */
	template<typename VarSet, typename param_t>
	static inline void jacobian(param_t& param, Eigen::Ref<typename VarSet::variable_t> x, 
		Eigen::Ref<Eigen::Vector<typename VarSet::Scalar, size>> out,
		Eigen::SparseMatrix<typename VarSet::Scalar>& jac,
		std::vector<std::size_t>& jacobian_offsets,
		int start_offset)
	{
		auto ret = Function::jac(param, VarSet::template get<Variables>(x)...);
		out = ret.val;

		// Copy the jacobian into the right place
		using Scalar = typename VarSet::Scalar;
		for(int i=0; i<num_vars; i++)
		{
			Eigen::Map<Eigen::Vector<Scalar, size>>(jac.valuePtr() + jacobian_offsets[start_offset+i])
				= ret.jacobian.col(i);
		}
	}

	/** Compute the value of lambda' * f(x)
	 */
	// template<typename VarSet, typename param_t>
	// static inline void value(
	// 	)
	// {}

	/** Compute the gradient and value of lambda' * f(x)
	 */

	/** Compute the hessian, gradient and value of lambda' * f(x)
	 * 
	 * The result is added to the existing value, gradient and hessian
	 */
	template<typename VarSet, typename param_t>
	static inline void hessian(
		// Inputs
		param_t& param, 
		Eigen::Ref<typename VarSet::variable_t> x, 
		Eigen::Ref<Eigen::Vector<typename VarSet::Scalar, size>> lambda,

		// Outputs
		typename VarSet::Scalar& value,
		Eigen::Ref<typename VarSet::variable_t> gradient,
		Eigen::SparseMatrix<typename VarSet::Scalar>& hessian,

		// // Vector of the offset to each column for the hessian xxxx
		std::vector<std::size_t>& hessian_offsets,
		int start_offset
		)
	{
		using Scalar = typename VarSet::Scalar;

		auto ret = Function::hessian(param, VarSet::template get<Variables>(x)...);

        value += lambda.dot(ret.val);

        // Add to the gradient
        auto local_grad = lambda.transpose() * ret.jacobian;
        int offset = 0;
        auto l = { (
        	VarSet::template get<Variables>(gradient) += local_grad.template segment<Variables::size>(offset),
        	offset += Variables::size,
        	0)... };

        // Add the given block from the local hessian to the global hessian
        auto add_block = [&hessian_offsets, &hessian, &offset](
			Eigen::Ref<Eigen::MatrixX<Scalar>> localHessian
        	)
        {
        	// Get the block in question
        	auto blk = localHessian.block()

        	// Copy each column of the local Hessian into the global Hessian
        	for(int i=0; i<HessianBlockRow.cols(); i++)
				Eigen::Map<Eigen::VectorX<Scalar>>(hessian.valuePtr() + hessian_offsets[offset+i], HessianBlockRow.rows())
					 += HessianBlockRow.col(i);
			offset += HessianBlockRow.cols();
        	
        };

        // Add the hessian
        int row;
        offset = 0; // Offset into the hessian_offsets
        for(int i=0; i<size; i++)
        {
        	row = 0;
	        auto l = { (
	        	add_row(lambda[i] * (ret.hessian[i]).template middleRows<Variables::size>(row)),
	        	row += Variables::size,
	        	0)... };
        }
	}



	/*** Setup functions. Not part of the user interface */

	/** Add nonzero enties to the tuple list for this constraint.
	 * 
	 * row = start row of this constraint
	 */
	template<typename VarSet, typename Scalar = typename VarSet::Scalar>
	static inline void jacobian_get_tripletlist(std::vector< Eigen::Triplet<Scalar> >& tripletList, int row)
	{
		auto itr_block = [&tripletList, &row](int start_col, int num_cols)
		{
			for(int r=row; r<row+size; r++)
				for(int c=start_col; c<start_col+num_cols; c++)
				{
					tripletList.push_back(Eigen::Triplet<Scalar>(r,c,1));
				}
		};

		// Assume that the jacobian is dense in the variables
        auto l = { (
        	itr_block(VarSet::template offset<Variables>(), Variables::size),
        	0)... };
	}

	/** Add nonzero enties to the tuple list for this constraint.
	 */
	template<typename VarSet, typename Scalar = typename VarSet::Scalar>
	static inline void hessian_get_tripletlist(std::vector< Eigen::Triplet<Scalar> >& tripletList)
	{
		auto itr_variable = [&tripletList](int row_start, int row_num)
		{
			auto itr_block = [&tripletList, &row_start, &row_num](int col_start, int col_num)
			{
				for(int r=row_start; r<row_start+row_num; r++)
					for(int c=col_start; c<col_start+col_num; c++)
					{
						tripletList.push_back(Eigen::Triplet<Scalar>(r,c,1));
					}
			};

			// We've fixed the row, now iterate over the variables in the column
	        auto l = { (
	        	itr_block(VarSet::template offset<Variables>(), Variables::size),
	        	0)... };

		};

		// Assume that the hessian is dense in the variables
        auto l = { (
        	itr_variable(VarSet::template offset<Variables>(), Variables::size),
        	0)... };
	}


	/** Compute the start point of each column
	 */
	template<typename VarSet, typename Scalar = typename VarSet::Scalar>
	static void jacobian_get_offsets(int row, Eigen::SparseMatrix<Scalar>& indexJ, std::vector<std::size_t>& offsets)
	{
		auto itr_block = [&row, &offsets, &indexJ](int start_col, int num_cols)
		{
			for(int col=start_col; col<start_col+num_cols; col++)
			{
				offsets.push_back((std::size_t)(indexJ.coeff(row, col) - 1));
			}
		};

        auto l = { (
        	itr_block(VarSet::template offset<Variables>(), Variables::size),
        	0)... };
	}


	/** Compute the start-point of each column in the hessian in the order of 
	 * computation in the hessian function
	 */
	template<typename VarSet, typename Scalar = typename VarSet::Scalar>
	static void hessian_get_offsets(Eigen::SparseMatrix<Scalar>& indexH, std::vector<std::size_t>& offsets)
	{
        // Add the block-row to the hessian in the given locations
        auto add_row = [&offsets, &indexH](
			Eigen::Ref<Eigen::MatrixX<Scalar>> HessianBlockRow
        	)
        {
        	// Copy each column of the local Hessian into the global Hessian
        	for(int i=0; i<HessianBlockRow.cols(); i++)
				Eigen::Map<Eigen::VectorX<Scalar>>(hessian.valuePtr() + hessian_offsets[offset+i], HessianBlockRow.rows())
					 += HessianBlockRow.col(i);
			offset += HessianBlockRow.cols();
        	
        };

        // Add the hessian
        int row;
        offset = 0; // Offset into the hessian_offsets
        for(int i=0; i<size; i++)
        {
        	row = 0;
	        auto l = { (
	        	add_row(lambda[i] * (ret.hessian[i]).template middleRows<Variables::size>(row)),
	        	row += Variables::size,
	        	0)... };
        }
	}
};

template<typename Function, typename... Variables>
struct traits<FunctionConstraint<Function, Variables...>> 
{
	static constexpr std::size_t size = Function::num_outputs;
};


template<typename VarSet, typename ConstraintTuple>
struct ConstraintSet;

template<typename... Variables, typename... Constraints>
struct ConstraintSet<VariableSet<Variables...>, std::tuple<Constraints...>> 
{
	using VarSet = VariableSet<Variables...>;
	static constexpr int num_variables = VarSet::num_vars;
	static constexpr int num_constraints = meta::sum_template<Constraints::size...>();

	// User types
	using Scalar = typename VarSet::Scalar;
	using variable_t = typename VarSet::variable_t;
	using constraint_t = typename Eigen::Vector<Scalar, num_constraints>;
	using jacobian_t = typename Eigen::SparseMatrix<Scalar>;
	using hessian_t = typename Eigen::SparseMatrix<Scalar>;

	template<typename param_t>
	static inline void eval(param_t& param, Eigen::Ref<variable_t> x, Eigen::Ref<constraint_t> con)
	{
		// Static iterate over Constraints
		int offset = 0;
		auto l = {(
			Constraints::template eval<VarSet>(param, x, con.template segment<Constraints::size>(offset)), 
			offset += Constraints::size,
			0)...};
	};

	// Pre-store the locations to copy each column of each jacobian/hessian block to
	std::vector<std::size_t> sparse_jacobian_offsets;
	std::vector<std::size_t> sparse_hessian_offsets;

	template<typename param_t>
	inline void jacobian(param_t& param, 
						 Eigen::Ref<variable_t> x, 
						 Eigen::Ref<constraint_t> con, 
						 jacobian_t& jac) // Must be pre-initialized by sparse_init_jacobian
	{
		int offset = 0; // offset into con vector
		int jac_offset = 0; // offset into sparse_jacobian_offsets
		auto l = {(
			Constraints::template jacobian<VarSet>(param, x, 
				con.template segment<Constraints::size>(offset),
				jac,
				sparse_jacobian_offsets,
				jac_offset),
			offset += Constraints::size,
			jac_offset += Constraints::num_vars, 
			0)...};
	};

	/** Compute hessian, gradient and value of lambda' * f(x)
	 */
	template<typename param_t>
	inline void hessian(
		// Inputs
		param_t& param, 
		Eigen::Ref<variable_t> x, 
		Eigen::Ref<constraint_t> lambda, 

		// Outputs
		Scalar& value,
		Eigen::Ref<variable_t> gradient, 
		hessian_t& hessian) // Must be pre-initialized by sparse_init_hessian
	{
		int offset = 0; // offset into lambda vector
		// int jac_offset = 0; // offset into sparse_jacobian_offsets
		auto l = {(
			Constraints::template hessian<VarSet>(
				param, x, lambda.template segment<Constraints::size>(offset),
				value, gradient, hessian),
			offset += Constraints::size,
			0)...};
	};

	void sparse_init_jacobian(Eigen::SparseMatrix<Scalar>& J)
	{
		// Iterate over eack constraint in the set and copy the jacobian to the given location
		std::vector< Eigen::Triplet<Scalar> > tripletList;
		tripletList.reserve(meta::sum_template<Constraints::nonZerosAtCompileTime...>());
		int row = 0;
		auto l = {(
			Constraints::template jacobian_get_tripletlist<VarSet>(tripletList, row), 
			row += Constraints::size,
			0)...};

		J.resize(num_constraints, num_variables);
		J.setFromTriplets(tripletList.begin(), tripletList.end());

		// Index each point in the valueptr
		auto dataPtr = J.valuePtr();
		for(int i=0; i<J.nonZeros(); i++) dataPtr[i] = (Scalar)(i+1);

		// Store the order of columns required
		row = 0;
		auto m = {(
			Constraints::template jacobian_get_offsets<VarSet>(row, J, sparse_jacobian_offsets),
			row += Constraints::size,
			0)...};
	};

	void sparse_init_hessian(Eigen::SparseMatrix<Scalar>& H)
	{
		std::vector< Eigen::Triplet<Scalar> > tripletList;
		tripletList.reserve(meta::sum_template<Constraints::num_vars * Constraints::num_vars ...>());
		auto l = {(
			Constraints::template hessian_get_tripletlist<VarSet>(tripletList), 
			0)...};

		H.resize(num_variables, num_variables);
		H.setFromTriplets(tripletList.begin(), tripletList.end());

		// Index each point in the valueptr
		auto dataPtr = H.valuePtr();
		for(int i=0; i<H.nonZeros(); i++) dataPtr[i] = (Scalar)(i+1);

		// Store the order of columns required
		auto m = {(
			Constraints::template hessian_get_offsets<VarSet>(H, sparse_hessian_offsets),
			0)...};
	
    std::cout << "sparse_hessian_offsets = ";
    for(const auto x: sparse_hessian_offsets)
	    std::cout << x << ", ";
	std::cout << "\n";
	};

};

};
#endif