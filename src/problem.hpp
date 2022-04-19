#ifndef __PROBLEM_HPP
#define __PROBLEM_HPP

#include <numeric>

namespace lampc
{

/**
 * We create two copies of the problem (and the contained constraints, objective, etc). 
 * 
 * The first is a Tape version, in which the variable information is also propagated with every function
 * call and passed to BSMatrixTape objects to develop the spasity structures and record the copy operations.
 * 
 * The second is the non-tape (deployment) version, which doesn't have the variable information and
 * just plays back the tape for speed. Only this version needs to be optimized.
 */


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

	BSMatrixTape<scalar_t> lb;
	BSMatrixTape<scalar_t> ub;

public:
	ConstraintTape() : Base()
	{}

	// /**
	//  * Add a block-row to the constraint
	//  */
	// template<typename value_t, typename jacobian_t>
	// ConstraintTape<scalar_t>& operator<<(
	// 								std::pair<std::vector<Segment>, // Segment information for the columns of the jacobian
	// 							  std::pair<value_t, jacobian_t>>
	// 							  data)
	// {
	// 	// Push a new constraint on the bottom, and set the columns
	// 	value(-1, 0);
	// 	jacobian(-1, data.first);

	// 	// Copy the value and jacobian to the right place
	// 	std::tie(value, jacobian) = data.second;
	// 	return *this;
	// }


	template<typename Func>
	ConstraintTape<scalar_t>& operator<<(typename lampc::CallTape<Func> call)
	{
		// Push a new constraint on the bottom, and set the columns
		value(-1, 0);

		// Copy the value to the right place
		value = call.value;
		return *this;
	}

	template<typename Func>
	ConstraintTape<scalar_t>& operator<<(typename lampc::JacobianTapeCall<Func> call)
	{
		// Push a new constraint on the bottom, and set the columns
		value(-1, 0);
		jacobian(-1, call.inputs);

		// Copy the value and jacobian to the right place
		value = call.value;
		jacobian = call.jacobian;
		return *this;
	}

	// template<int num_outputs, int... input_sizes>
	// using JacobianTapeCall = typename lampc::FunctionInfo<scalar_t, num_outputs, input_sizes...>::JacobianCall;

	// template<int num_outputs, int... input_sizes>
	// ConstraintTape<scalar_t>& operator<<(JacobianTapeCall<num_outputs, input_sizes...>& data)
	// {

	// }

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
	template<typename Func>
	Constraint<scalar_t>& operator<<(typename lampc::Call<Func> call)
	{
		// Copy the value to the right place
		value = call.value;
		return *this;
	}

	template<typename Func>
	Constraint<scalar_t>& operator<<(typename lampc::JacobianCall<Func> call)
	{
		// Copy the value and jacobian to the right place
		value = call.value;
		jacobian = call.jacobian;
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


/**
 * WeightedSum defined a function that's a weighted sum
 * 
 * f(x) = sum_i w_i * f_i(x)
 * 
 */

template<typename scalar_t>
struct WeightedSumTape
{
	WeightedSumTape() {}

	scalar_t value;
	BSMatrixTape<scalar_t> gradient;
	BSMatrixTape<scalar_t> hessian;

	int num_weights = 0;

	template<typename Func>
	void operator+=(typename lampc::HessianTapeCall<Func> call)
	{
		gradient(call.inputs, 0) += call.jacobian.colwise().sum().transpose(); // 1'*jacobian

		// Iterate over the hessian
		// The i'th hessian is wrt the i'th row of this vector function
		for(auto& h : call.hessian)
			hessian(call.inputs, call.inputs) += h;

		num_weights += call.jacobian.rows();
	}


	// template<typename value_t, typename jacobian_t, typename hessian_array_t>
	// void operator+=(std::pair<std::vector<Segment>, // Information about the variables
	// 							  std::tuple<value_t, jacobian_t, hessian_array_t>>
	// 							  data)
	// {
	// 	gradient(data.first, 0) += (std::get<1>(data.second).colwise().sum()).transpose(); // 1'*jacobian

	// 	// Iterate over the hessian
	// 	// The i'th hessian is wrt the i'th row of this vector function
	// 	for(auto& h : std::get<2>(data.second))
	// 		hessian(data.first, data.first) += h;

	// 	num_weights += std::get<0>(data.second).rows();
	// }

	void operator=(int) {}

	void finalize_structure()
	{
		gradient.finalize_structure();
		hessian.finalize_structure();
	}
};

template<typename scalar_t>
struct WeightedSum
{
	WeightedSum() : weight_src(NULL)
	{}

	scalar_t value;
	BSMatrix<scalar_t> gradient;
	BSMatrix<scalar_t> hessian;

	int num_weights = 0;
	int weight_offset = 0; // Offset into the weight vector

	// Called to restart the computation process
	inline void operator=(int)
	{
		weight_offset = 0;
		value = 0;
		gradient.set_zero();
		hessian.set_zero();
	}

	template<typename Func>
	inline void operator+=(typename lampc::HessianCall<Func> call)
	{
		assert(weight_src != NULL && "set_weight must be called before calling this function");
		int rows = call.jacobian.rows(); // Number of rows in the jacobian
		assert(num_weights >= weight_offset + rows && "weight vector is too small");

		auto w = Eigen::Map<Eigen::VectorX<scalar_t>>(weight_src+weight_offset, rows);
		weight_offset += rows;

		value += w.transpose() * call.value;
		gradient += call.jacobian.transpose() * w;

		// Iterate over the hessian
		// The i'th hessian is wrt the i'th row of this vector function
		for(int i=0; i<rows; i++)
			hessian += w(i) * call.hessian[i];
	}

	// template<typename value_t, typename jacobian_t, typename hessian_array_t>
	// inline void operator+=(std::tuple<value_t, jacobian_t, hessian_array_t> data)
	// {
	// 	assert(weight_src != NULL && "set_weight must be called before calling this function");
	// 	int rows = std::get<1>(data).rows(); // Number of rows in the jacobian
	// 	assert(num_weights >= weight_offset + rows && "weight vector is too small");

	// 	auto w = Eigen::Map<Eigen::VectorX<scalar_t>>(weight_src+weight_offset, rows);
	// 	weight_offset += rows;

	// 	value += w.transpose() * std::get<0>(data);
	// 	gradient += std::get<1>(data).transpose() * w;

	// 	// Iterate over the hessian
	// 	// The i'th hessian is wrt the i'th row of this vector function
	// 	for(int i=0; i<rows; i++)
	// 		hessian += w(i) * std::get<2>(data)[i];
	// }

	inline void finalize_structure() {}

	void initialize_from_tape(WeightedSumTape<scalar_t>& tape)
	{
		gradient.initialize_from_tape(tape.gradient);
		hessian.initialize_from_tape(tape.hessian);
		num_weights = tape.num_weights;
	}

	/**
	 * Set the memory source that the weight will be taken from
	 */
	void set_weight(Eigen::VectorX<scalar_t>& weight)
	{
		assert(num_weights == weight.rows() && "Provided weight vector is not the right length");
		weight_src = weight.data();
	}

private:
	scalar_t* weight_src;
};


template<typename scalar_t, typename _Variable, typename _Constraint, typename _Objective>
struct ProblemBase
{
	using Variable = _Variable;
	using Constraint = _Constraint;
	using Objective = _Objective;

	Constraint constraints;
	Objective objective;

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

	// void finalize_structure(int rows=-1, int cols=-1)
	// {
	// 	constraints.finalize_structure(rows,cols);
	// }

private:
	std::vector<std::shared_ptr<Variable>> variables;

	int compute_size() // Total number of optimization variables
	{
		return std::accumulate(variables.begin(), variables.end(), 0, 
							[](int total, std::shared_ptr<Variable> var) {return total + var->num_elements();});
	}
};


template<typename scalar_t>
struct ProblemTape : public ProblemBase<scalar_t, VariableTape<scalar_t>, ConstraintTape<scalar_t>, WeightedSumTape<scalar_t>>
{
	ProblemTape() {}
};

template<typename scalar_t>
struct Problem : public ProblemBase<scalar_t, Variable<scalar_t>, Constraint<scalar_t>, WeightedSum<scalar_t>>
{
	Problem() {}

	void initialize_from_tape(ProblemTape<scalar_t>& tape)
	{
		this->constraints.initialize_from_tape(tape.constraints);
		this->objective.initialize_from_tape(tape.objective);
	}

	/**
	 * Allocates memory for this problem, including the sparsity structures
	 */
	void initialize_memory(Eigen::VectorX<scalar_t>& var,
												 Eigen::VectorX<scalar_t>& g,
											   Eigen::SparseMatrix<scalar_t>& g_jacobian,
											   Eigen::VectorX<scalar_t>& obj_gradient,
											   Eigen::SparseMatrix<scalar_t>& obj_hessian,
											   Eigen::VectorX<scalar_t>& obj_weight)
	{
		var.resize(this->size, 1);
		this->constraints.jacobian.initialize_matrix(g_jacobian);
		g.resize(g_jacobian.rows(), 1);

		obj_gradient.resize(g_jacobian.cols(), 1);
		this->objective.hessian.initialize_matrix(obj_hessian);
		obj_weight.resize(this->objective.num_weights, 1);
		obj_weight.array() = 1; // Default
	}

	/**
	 * Sets the memory locations that this problem will read/write to.
	 * 
	 * Assumption: All memory has already been allocated
	 */
	void set_memory_targets(Eigen::VectorX<scalar_t>& var,
													Eigen::VectorX<scalar_t>& g,
											   	Eigen::SparseMatrix<scalar_t>& g_jacobian,
											    Eigen::VectorX<scalar_t>& obj_gradient,
 											    Eigen::SparseMatrix<scalar_t>& obj_hessian,
 											    Eigen::VectorX<scalar_t>& obj_weight)
	{
    this->set_variable(var);

    this->constraints.value.set_target(g);
    this->constraints.jacobian.set_target(g_jacobian);

    this->objective.gradient.set_target(obj_gradient);
    this->objective.hessian.set_target(obj_hessian);
    this->objective.set_weight(obj_weight);
	}
};


/**
 * Creates and runs the tape for a user-defined class, and then 
 * creates the given executable object from the tape.
 * 
 * Assumes that the user-defined class contains two functions:
 * 	eval_constraints
 * 	eval_objective
 * 
 * Calling form:
 *   auto prob = makeProblem<UserClass, double>(10);
 */
template<template<typename, typename> class P, typename scalar_t, typename... Args>
P<scalar_t, lampc::Problem<scalar_t>> make_problem(Args... args)
{
  // Create the tape to record the copy sequence
  P<scalar_t, lampc::ProblemTape<scalar_t>> tape(args...);

	// Temporary optimization variable used to call all functions 
	// when creating the tape
  Eigen::VectorX<scalar_t> var(tape.size);
  tape.set_variable(var);
  
  tape.template eval_constraints<lampc::Jacobian>(); // Create structure
  tape.constraints.finalize_structure();
  tape.template eval_constraints<lampc::Jacobian>(); // Create copy-sequence

  tape.template eval_objective<lampc::Hessian>(); // Create structure
  tape.objective.finalize_structure();
  tape.template eval_objective<lampc::Hessian>(); // Create copy-sequence

  // We now create and initialize the problem from the tape
  P<scalar_t, lampc::Problem<scalar_t>> prob(args...);
  prob.initialize_from_tape(tape);

  return prob;
}


/**
 * The Problem class doesn't own any of the memory for the problem.
 * This is done because most solvers (e.g., ipopt) own their own
 * memory.
 * 
 * This class is a helper that provides a full set of memory for a given
 * problem.
 */
template<typename scalar_t>
struct ProblemMemory
{
  Eigen::VectorX<scalar_t> var;

  Eigen::VectorX<scalar_t> g;
  Eigen::SparseMatrix<scalar_t> g_jacobian;

  Eigen::VectorX<scalar_t> obj_gradient;
  Eigen::SparseMatrix<scalar_t> obj_hessian;
  Eigen::VectorX<scalar_t> obj_weight;
};

/**
 * Allocates and associates problem memory for a given problem.
 */
template<typename scalar_t, typename P>
ProblemMemory<scalar_t> make_problem_memory(P& prob)
{
	ProblemMemory<scalar_t> mem;

	prob.initialize_memory(mem.var, mem.g, mem.g_jacobian, mem.obj_gradient, mem.obj_hessian, mem.obj_weight);
	prob.set_memory_targets(mem.var, mem.g, mem.g_jacobian, mem.obj_gradient, mem.obj_hessian, mem.obj_weight);

	return mem;
}


};

#endif // __PROBLEM_HPP