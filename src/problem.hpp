#ifndef __PROBLEM_HPP
#define __PROBLEM_HPP

#include <numeric>

namespace lampc
{

/**
 * Represents a variable, which is an offset into
 * a global decision variable.
 */
template<typename scalar_t>
struct VariableBase
{
	VariableBase(int n, int m, int offset) 
		: n(n), m(m), offset(offset), 
			src(NULL,n,m) // Variable is invalid until set_var is called
		{}

	inline int num_elements() { return n*m; } // Total number of elements in the variable (n*m)
	inline int size() { return n; } // Size of a single variable
	inline std::pair<int,int> shape() { return make_pair(n,m); } // Shape of the variable matrix
	inline int rows() { return n; } 
	inline int cols() { return m; } 

	inline void set_var(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		src_valid = true;
	  new (&src) Eigen::Map<Eigen::MatrixX<scalar_t>>(var.data() + offset, n, m);
	}

	// /** 
	//  * Return the matrix
	//  */
	// inline Eigen::Map<Eigen::MatrixX<scalar_t>> operator[]()
	// {
	// 	return src;
	// }

	/** 
	 * Return a reference to the index'th variable in the matrix
	 */
	inline Eigen::Ref<Eigen::VectorX<scalar_t>> operator[](int index)
	{
		assert(src_valid && "set_var must be called before using this variable");
		return src(Eigen::all,index);
	}

	// /**
	//  * Return a pair of {offset, length}
	//  */
	// inline Segment operator[](int index)
	// {
	// 	return Segment{.index = offset + index * n, .length = n};
	// }	

protected:

	// Offset into the global variable
	int offset;
	int n,m;

	// Reference to start of this variable in the global variable
	Eigen::Map<Eigen::MatrixX<scalar_t>> src;

	bool src_valid = false; // Used for debugging. Set only if src is valid.
};



template<typename scalar_t>
struct VariableTape : public VariableBase<scalar_t>
{
	VariableTape(int n, int m, int offset) : VariableBase<scalar_t>(n,m,offset)
		{}

	/** 
	 * Return a reference to the index'th variable in the matrix and the segment info
	 */
	inline std::pair<Segment, Eigen::Ref<Eigen::VectorX<scalar_t>>> operator()(int index)
	{
		assert(this->src_valid && "set_var must be called before using this variable");
		return std::make_pair(Segment{.index = this->offset + index * this->n, .length = this->n}, this->src(Eigen::all,index));
	}
};

template<typename scalar_t>
struct Variable : public VariableBase<scalar_t>
{
	Variable(int n, int m, int offset) : VariableBase<scalar_t>(n,m,offset)
		{}

	/** 
	 * Return a reference to the index'th variable in the matrix
	 */
	inline Eigen::Ref<Eigen::VectorX<scalar_t>> operator()(int index)
	{
		assert(this->src_valid && "set_var must be called before using this variable");
		return this->src(Eigen::all,index);
	}
};


/**
 * Vector function g(x) = [g1(x); ...; gN(x)]
 * 
 * Can add block-rows to the vector function and evaluate the function and its' jacobian
 */
template<typename scalar_t, typename Derived>
struct ConstraintBase
{
public:
	ConstraintBase()
	{}


	// TODO: Add overload to handle value-only calculation for constraint

};

template<typename scalar_t>
struct ConstraintTape : public ConstraintBase<scalar_t, ConstraintTape<scalar_t>>
{
	using Base = ConstraintBase<scalar_t, ConstraintTape<scalar_t>>;
	friend Base;

	BSMatrixTape<scalar_t> value;
	BSMatrixTape<scalar_t> jacobian;

public:
	ConstraintTape() : Base()
	{}

	/**
	 * Add a block-row to the constraint
	 */
	template<typename value_t, typename jacobian_t>
	ConstraintTape<scalar_t>& operator<<(
									std::pair<std::vector<Segment>, // Segment information for the columns of the jacobian
								  std::pair<value_t, jacobian_t>>
								  data)
	{
		// Push a new constraint on the bottom, and set the columns
		value(-1, 0);
		jacobian(-1, data.first);

		// Copy the value and jacobian to the right place
		std::tie(value, jacobian) = data.second;
		return *this;
	}

	/**
	 * Called when all copy operations have been completed once, which fixes the sparsity structure.
	 * 
	 * If rows and cols aren't specified, then they will be taken as large enough to contain all the
	 * blocks copied in.
	 */
	void finalize_structure(int rows=-1, int cols=-1)
	{
		value.finalize_structure(rows, 1);
		jacobian.finalize_structure(rows, cols);
	}
};

template<typename scalar_t>
struct Constraint : public ConstraintBase<scalar_t, Constraint<scalar_t>>
{
	using Base = ConstraintBase<scalar_t, Constraint<scalar_t>>;
	friend Base;

	BSMatrix<scalar_t> value;
	BSMatrix<scalar_t> jacobian;

public:
	Constraint() : Base()
	{}

	void initialize_from_tape(ConstraintTape<scalar_t>& tape)
	{
		value.initialize_from_tape(tape.value);
		jacobian.initialize_from_tape(tape.jacobian);
	}

	/**
	 * Add a block-row to the constraint
	 */
	template<typename value_t, typename jacobian_t>
	Constraint<scalar_t>& operator<<(std::pair<value_t, jacobian_t> data)
	{
		// Copy the value and jacobian to the right place
		std::tie(value, jacobian) = data;
		return *this;
	}

	/**
	 * Called when all copy operations have been completed once, which fixes the sparsity structure.
	 * 
	 * If rows and cols aren't specified, then they will be taken as large enough to contain all the
	 * blocks copied in.
	 */
	void finalize_structure(int rows=-1, int cols=-1)
	{
	}

};



template<typename scalar_t, typename _Variable, typename _Constraint>
struct ProblemBase
{
	using Variable = _Variable;
	using Constraint = _Constraint;

	Constraint constraints;

	Variable& variable(int n, int m=1)
	{
		auto v = std::make_shared<Variable>(n,m,size);
		variables.push_back(v);
		size = compute_size(); // Update the total size of the problem variable

		// Return with move symantics is required, 
		// since we need the reference stored in variables to remain valid
		return *(v.get());
	}

	int size = 0;

	void set_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		for(auto v: variables)
			v->set_var(var);
	}

	void finalize_structure(int rows=-1, int cols=-1)
	{
		constraints.finalize_structure(rows,cols);
	}

private:
	std::vector<std::shared_ptr<Variable>> variables;

	int compute_size() // Total number of optimization variables
	{
		return std::accumulate(variables.begin(), variables.end(), 0, 
							[](int total, std::shared_ptr<Variable> var) {return total + var->num_elements();});
	}
};


template<typename scalar_t>
struct ProblemTape : public ProblemBase<scalar_t, VariableTape<scalar_t>, ConstraintTape<scalar_t>>
{
	ProblemTape() {}

};

template<typename scalar_t>
struct Problem : public ProblemBase<scalar_t, Variable<scalar_t>, Constraint<scalar_t>>
{
	Problem() {}

	void initialize_from_tape(ProblemTape<scalar_t>& tape,
														Eigen::VectorX<scalar_t>& var,
														Eigen::SparseMatrix<scalar_t>& g,
														Eigen::SparseMatrix<scalar_t>& g_jacobian)
	{
		this->constraints.initialize_from_tape(tape.constraints);
    this->set_variable(var);

		this->constraints.jacobian.initialize_matrix(g_jacobian);
		this->constraints.value.initialize_matrix(g);
	}
};

};

#endif // __PROBLEM_HPP