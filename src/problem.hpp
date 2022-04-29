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

protected:

	// Offset into the global variable
	int offset;
	int n,m;

	// Reference to start of this variable in the global variable
	Eigen::Map<Eigen::MatrixX<scalar_t>> src;

	bool src_valid = false; // Used for debugging. Set only if src is valid.
};


/**
 * Information about a slice of the variable. Used only in the tape.
 */
template<typename scalar_t>
struct VarSliceInfo
{
	Segment segment;
	Eigen::Ref<Eigen::VectorX<scalar_t>> vec;

	VarSliceInfo(size_t index, size_t length, Eigen::Ref<Eigen::VectorX<scalar_t>> vec)
	: segment{.index=index, .length=length}, vec(vec) {}

	// Pass through all eigen indexing calls to vec
	template<typename... Args>
	EIGEN_STRONG_INLINE auto operator()(Args... args) -> decltype(vec(args...)) {return vec(args...);}
 };

template<typename scalar_t>
struct VariableTape : public VariableBase<scalar_t>
{
	VariableTape(int n, int m, int offset) : VariableBase<scalar_t>(n,m,offset)
		{}

	/** 
	 * Return a reference to the index'th variable in the matrix and the segment info
	 */
	inline VarSliceInfo<scalar_t> operator()(int index)
	{
		assert(this->src_valid && "set_var must be called before using this variable");
		return VarSliceInfo<scalar_t>(this->offset + index * this->n, this->n, this->src(Eigen::all,index));
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
template<typename MatrixType>
struct ConstraintBase
{
	// Used for sparsity discovery
	ConstraintBase()
	{
		// Specify initial sizes
		value.resize(0,1);
		lb.resize(0,1);
		ub.resize(0,1);
		lb_x.resize(0,1);
		ub_x.resize(0,1);
	}

	// Sparsity structure is specified
	ConstraintBase(Eigen::MatrixX<bool> jacobian_sparsity_structure)
		: jacobian(jacobian_sparsity_structure)
	{
		// Set the value and the bounds to be dense vectors
		Eigen::VectorX<bool> dense_vector(jacobian_sparsity_structure.rows());
		dense_vector.array() = 1;
		value.initialize(dense_vector, 0, 1);
		lb.initialize(dense_vector, 0, 1);
		ub.initialize(dense_vector, 0, 1);

		dense_vector.resize(jacobian_sparsity_structure.cols());
		dense_vector.array() = 1;
		lb_x.initialize(dense_vector, 0, 1);
		ub_x.initialize(dense_vector, 0, 1);
	}

	// Copy sequence is specified
	ConstraintBase(BSMatrixInfo value, BSMatrixInfo jacobian, 
								 BSMatrixInfo lb, BSMatrixInfo ub,
								 BSMatrixInfo lb_x, BSMatrixInfo ub_x)
		: value(value), jacobian(jacobian), lb(lb), ub(ub), lb_x(lb_x), ub_x(ub_x)
		{}


	MatrixType value;
	MatrixType jacobian;

	// Upper/lower bounds of the constraints
	MatrixType lb;
	MatrixType ub;

	// Upper/lower bounds of the variables
	MatrixType lb_x;
	MatrixType ub_x;

	void add_variable(int var_size)
	{
		jacobian.extend(0, var_size);
		lb_x.extend(var_size, 0);
		ub_x.extend(var_size, 0);
	}

	void add_constraint(int rows)
	{
		value.extend(rows, 0);
		jacobian.extend(rows, 0);

		lb.extend(rows, 0);
		ub.extend(rows, 0);
	}
};


/**
 * Construction form of the constraint
 */
template<typename MatrixType>
struct ConstraintConstruction : public ConstraintBase<MatrixType>
{
	using Base = ConstraintBase<MatrixType>;
	using Base::Base; // Brings the base constructors into scope

	/**
	 * Add a block-row to the constraint
	 * 
	 * Vars are of type VarSliceInfo<scalar_t>'s
	 */
	template<typename F, typename... Vars>
	void add(lampc::Eval, F& f, Vars... x)
	{
		// Push a new constraint on the bottom, and set the columns
		this->add_constraint(f.num_outputs);

		// Copy the value and jacobian to the right place
		f(x.vec..., this->value(seqN(lastp1-f.num_outputs,f.num_outputs),0));
	}

	template<typename F, typename... Vars>
	void add(lampc::Jacobian, F& f, Vars... x)
	{
		// Push a new constraint on the bottom, and set the columns
		this->add_constraint(f.num_outputs);
		auto con_rows = seqN(lastp1-f.num_outputs,f.num_outputs);

		// Copy the value and jacobian to the right place
		f(x.vec..., 
			this->value(con_rows,0),
			this->jacobian(con_rows, multiSeq(x.segment.seq()...))
			);
	}

	// Default evaluation is to add the constraint with jacobian computation
	template<typename F, typename... Vars>
	void operator()(F& f, Vars... x)
	{
		add(lampc::Jacobian(), f, x...);
	}	
};

/**
 * Deployment form of the constraint
 */
template<typename scalar_t>
struct Constraint : public ConstraintBase<BSMatrix<scalar_t>>
{
	using Base = ConstraintBase<BSMatrix<scalar_t>>;
	using Base::Base;

	/**
	 * Add a block-row to the constraint
	 * 
	 * Vars are of type VarSliceInfo<scalar_t>'s
	 */
	template<typename F, typename... Vars>
	void add(lampc::Eval, F& f, Vars... x)
	{
		f(x..., this->value);
	}

	template<typename F, typename... Vars>
	void add(lampc::Jacobian, F& f, Vars... x)
	{
		f(x..., this->value, this->jacobian);
	}

	// Default evaluation is to add the constraint with jacobian computation
	template<typename F, typename... Vars>
	void operator()(F& f, Vars... x)
	{
		add(lampc::Jacobian(), f, x...);
	}	
};


// /**
//  * WeightedSum defined a function that's a weighted sum
//  * 
//  * f(x) = sum_i w_i * f_i(x)
//  * 
//  */

// template<typename scalar_t>
// struct WeightedSumTape
// {
// 	WeightedSumTape() {}

// 	scalar_t value;
// 	BSMatrixTape<scalar_t> gradient;
// 	BSMatrixTape<scalar_t> hessian;

// 	int num_weights = 0;

// 	// template<typename Func>
// 	// void operator+=(typename lampc::CallTape<Func> call)
// 	// {}

// 	// template<typename Func>
// 	// void operator+=(typename lampc::JacobianTapeCall<Func> call)
// 	// {
// 	// 	gradient(call.inputs, 0) += call.jacobian.colwise().sum().transpose(); // 1'*jacobian
// 	// 	num_weights += call.jacobian.rows();
// 	// }

// 	// template<typename Func>
// 	// void operator+=(typename lampc::HessianTapeCall<Func> call)
// 	// {
// 	// 	gradient(call.inputs, 0) += call.jacobian.colwise().sum().transpose(); // 1'*jacobian

// 	// 	// Iterate over the hessian
// 	// 	// The i'th hessian is wrt the i'th row of this vector function
// 	// 	for(auto& h : call.hessian)
// 	// 		hessian(call.inputs, call.inputs) += h;

// 	// 	num_weights += call.jacobian.rows();
// 	// }

// 	void operator=(int) {}

// 	void finalize_structure()
// 	{
// 		gradient.finalize_structure();
// 		hessian.finalize_structure();
// 	}
// };

// template<typename scalar_t>
// struct WeightedSum
// {
// 	WeightedSum() : weight_src(NULL)
// 	{}

// 	scalar_t value;
// 	BSMatrix<scalar_t> gradient;
// 	BSMatrix<scalar_t> hessian;

// 	int num_weights = 0;
// 	int weight_offset = 0; // Offset into the weight vector

// 	// Called to restart the computation process
// 	inline void operator=(int)
// 	{
// 		weight_offset = 0;
// 		value = 0;
// 		gradient.set_zero();
// 		hessian.set_zero();
// 	}

// 	// template<typename Func>
// 	// void operator+=(typename lampc::Call<Func> call)
// 	// {
// 	// 	assert(weight_src != NULL && "set_weight must be called before calling this function");
// 	// 	int rows = call.value.rows(); 
// 	// 	assert(num_weights >= weight_offset + rows && "weight vector is too small");

// 	// 	auto w = Eigen::Map<Eigen::VectorX<scalar_t>>(weight_src+weight_offset, rows);
// 	// 	weight_offset += rows;

// 	// 	value += w.transpose() * call.value;
// 	// }

// 	// template<typename Func>
// 	// void operator+=(typename lampc::JacobianCall<Func> call)
// 	// {
// 	// 	assert(weight_src != NULL && "set_weight must be called before calling this function");
// 	// 	int rows = call.jacobian.rows(); // Number of rows in the jacobian
// 	// 	assert(num_weights >= weight_offset + rows && "weight vector is too small");

// 	// 	auto w = Eigen::Map<Eigen::VectorX<scalar_t>>(weight_src+weight_offset, rows);
// 	// 	weight_offset += rows;

// 	// 	value += w.transpose() * call.value;
// 	// 	gradient += call.jacobian.transpose() * w;
// 	// }

// 	// template<typename Func>
// 	// inline void operator+=(typename lampc::HessianCall<Func> call)
// 	// {
// 	// 	assert(weight_src != NULL && "set_weight must be called before calling this function");
// 	// 	int rows = call.jacobian.rows(); // Number of rows in the jacobian
// 	// 	assert(num_weights >= weight_offset + rows && "weight vector is too small");

// 	// 	auto w = Eigen::Map<Eigen::VectorX<scalar_t>>(weight_src+weight_offset, rows);
// 	// 	weight_offset += rows;

// 	// 	value += w.transpose() * call.value;
// 	// 	gradient += call.jacobian.transpose() * w;

// 	// 	// Iterate over the hessian
// 	// 	// The i'th hessian is wrt the i'th row of this vector function
// 	// 	for(int i=0; i<rows; i++)
// 	// 		hessian += w(i) * call.hessian[i];
// 	// }

// 	// template<typename value_t, typename jacobian_t, typename hessian_array_t>
// 	// inline void operator+=(std::tuple<value_t, jacobian_t, hessian_array_t> data)
// 	// {
// 	// 	assert(weight_src != NULL && "set_weight must be called before calling this function");
// 	// 	int rows = std::get<1>(data).rows(); // Number of rows in the jacobian
// 	// 	assert(num_weights >= weight_offset + rows && "weight vector is too small");

// 	// 	auto w = Eigen::Map<Eigen::VectorX<scalar_t>>(weight_src+weight_offset, rows);
// 	// 	weight_offset += rows;

// 	// 	value += w.transpose() * std::get<0>(data);
// 	// 	gradient += std::get<1>(data).transpose() * w;

// 	// 	// Iterate over the hessian
// 	// 	// The i'th hessian is wrt the i'th row of this vector function
// 	// 	for(int i=0; i<rows; i++)
// 	// 		hessian += w(i) * std::get<2>(data)[i];
// 	// }

// 	inline void finalize_structure() {}

// 	void initialize_from_tape(WeightedSumTape<scalar_t>& tape)
// 	{
// 		gradient.initialize_from_tape(tape.gradient);
// 		hessian.initialize_from_tape(tape.hessian);
// 		num_weights = tape.num_weights;
// 	}

// 	/**
// 	 * Set the memory source that the weight will be taken from
// 	 */
// 	void set_weight(Eigen::VectorX<scalar_t>& weight)
// 	{
// 		assert(num_weights == weight.rows() && "Provided weight vector is not the right length");
// 		weight_src = weight.data();
// 	}

// private:
// 	scalar_t* weight_src;
// };

/**
 * Captures all information about the sparsity pattern of a problem.
 * Input to the ProblemTape.
 */
struct ProblemSparsityPattern
{
	// ProblemSparsityPattern(filename) // Load from file

  Eigen::MatrixX<bool> constraints_jacobian;
  Eigen::MatrixX<bool> objective_hessian;
};


/**
 * Captures all information required to generate a Problem.
 * Input to the Problem.
 */
struct ProblemInfo
{
	// ProblemCopySequence(filename) // Load from file

	BSMatrixInfo constraints_jacobian;
	BSMatrixInfo constraints_value;

	BSMatrixInfo lb;
	BSMatrixInfo ub;

	BSMatrixInfo lb_x;
	BSMatrixInfo ub_x;
};



template<typename scalar_t, typename _Variable, typename _Constraint, typename _Objective>
struct ProblemBase
{
	// Construct Sparsity discovery
	ProblemBase() 
	{}

	// Construct Tape recording
	ProblemBase(ProblemSparsityPattern pattern) : constraint(pattern.constraints_jacobian) 
	{}

	// Construct Problem
	ProblemBase(ProblemInfo info) : constraint(info.constraints_value, info.constraints_jacobian, info.lb, info.ub, info.lb_x, info.ub_x) 
	{}


	using Variable = _Variable;
	using Constraint = _Constraint;
	using Objective = _Objective;

	Constraint constraint;
	Objective objective;

	std::shared_ptr<Variable> make_variable(int n, int m=1)
	{
		auto v = std::make_shared<Variable>(n,m,size);
		variables.push_back(v);
		size = compute_size(); // Update the total size of the problem variable

	  // Grow all the problem elements
		constraint.add_variable(n * m);
		// objective.add_variable(n * m);
		
		return v;
	}

	Variable& variable(int n, int m=1)
	{
		// Return with move symantics is required, 
		// since we need the reference stored in variables to remain valid
		std::shared_ptr<Variable> v = make_variable(n,m);
		return *(v.get());
	}

	int size = 0;

	void set_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		for(auto v: variables)
			v->set_var(var);
	}

	// void initialize()
	// {
	// 	for(auto v: variables)
	// 	{
	// 		// Tell the constraints and objective about this new variable
	// 		constraint.add_variable(v->rows() * v->cols());
	// 		// objective.add_variable(n*m); // TODO!!!		
	// 	}
	// }

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

/**
 * Common class for the sparsity and tape problems
 */
template<typename scalar_t, typename Constraint>
struct ProblemConstruction : public ProblemBase<scalar_t, VariableTape<scalar_t>, Constraint, double>
{
	using Base = ProblemBase<scalar_t, VariableTape<scalar_t>, Constraint, double>;

	ProblemConstruction() {}
	ProblemConstruction(ProblemSparsityPattern pattern) : Base(pattern) {}

	Eigen::VectorX<scalar_t> var;

	/**
	 * For the construction classes, we also provide the memory used for evaluation within problem.
	 * As a result, when a new variable is created, we need to re-allocate memory for var
	 */
	using Variable = typename Base::Variable;
	Variable& variable(int n, int m=1)
	{
		std::shared_ptr<Variable> new_variable = Base::make_variable(n,m);

		// Grow the size of the variable
		var.resize(this->size);
	  this->set_variable(var);

		return *(new_variable.get());
	}
};

template<typename scalar_t>
struct ProblemTape : public ProblemConstruction<scalar_t, ConstraintConstruction<BSMatrixTape>>
{
	using Base = ProblemConstruction<scalar_t, ConstraintConstruction<BSMatrixTape>>;
	using Base::constraint;
	ProblemTape(ProblemSparsityPattern pattern) : Base(pattern)
	{}

	ProblemInfo generate()
	{
		ProblemInfo info;

		info.constraints_jacobian = constraint.jacobian.generate();
		info.constraints_value = constraint.value.generate();

		info.lb = constraint.lb.generate();
		info.ub = constraint.ub.generate();

		info.lb_x = constraint.lb_x.generate();
		info.ub_x = constraint.ub_x.generate();

		return info;
	}
};

template<typename scalar_t>
struct ProblemSparsity : public ProblemConstruction<scalar_t, ConstraintConstruction<BSMatrixSparsity>>
{
	using Base = ProblemConstruction<scalar_t, ConstraintConstruction<BSMatrixSparsity>>;
	using Base::constraint;

	/**
	 * Produce all sparsity data for this problem
	 */
	ProblemSparsityPattern generate()
	{
		ProblemSparsityPattern pattern;
		pattern.constraints_jacobian = constraint.jacobian.get_sparsity();
		// Add other sparsity patterns here
		return pattern;
	}

};

template<typename scalar_t>
struct Problem : public ProblemBase<scalar_t, Variable<scalar_t>, Constraint<scalar_t>, double>
{
	using Base = ProblemBase<scalar_t, Variable<scalar_t>, Constraint<scalar_t>, double>;
	using Base::constraint;

	const size_t num_constraints;
	const size_t num_variables;

	Problem(ProblemInfo info) : Base(info),
		num_constraints(constraint.jacobian.rows()),
		num_variables(constraint.jacobian.cols())
	{}

	/**
	 * Sets the memory locations that this problem will read/write to.
	 * 
	 * Assumption: All memory has already been allocated
	 */
	void set_memory_targets(	  
		Eigen::Ref<Eigen::VectorX<scalar_t>> var,
		Eigen::Ref<Eigen::VectorX<scalar_t>> g,
		Eigen::SparseMatrix<scalar_t>& g_jacobian,
		Eigen::Ref<Eigen::VectorX<scalar_t>> lb,
		Eigen::Ref<Eigen::VectorX<scalar_t>> ub,
		Eigen::Ref<Eigen::VectorX<scalar_t>> lb_x,
		Eigen::Ref<Eigen::VectorX<scalar_t>> ub_x)
											    // Eigen::VectorX<scalar_t>& obj_gradient,
 											   //  Eigen::SparseMatrix<scalar_t>& obj_hessian,
 											   //  Eigen::VectorX<scalar_t>& obj_weight)
	{
		std::cout << "set_memory_targets\n";
    this->set_variable(var);

    this->constraint.value.set_target(g);
    this->constraint.jacobian.set_target(g_jacobian);
		this->constraint.lb.set_target(lb);
		this->constraint.ub.set_target(ub);
		this->constraint.lb_x.set_target(lb_x);
		this->constraint.ub_x.set_target(ub_x);

    // this->objective.gradient.set_target(obj_gradient);
    // this->objective.hessian.set_target(obj_hessian);
    // this->objective.set_weight(obj_weight);
	}

	/**
	 * The Problem class doesn't own any of the memory for the problem.
	 * This is done because most solvers (e.g., ipopt) own their own
	 * memory.
	 * 
	 * This class is a helper that provides a full set of memory for a given
	 * problem.
	 */
	struct ProblemMemory
	{
	  Eigen::VectorX<scalar_t> var;

		Eigen::VectorX<scalar_t> g;
		Eigen::SparseMatrix<scalar_t> g_jacobian;
		Eigen::VectorX<scalar_t> lb;
		Eigen::VectorX<scalar_t> ub;
		Eigen::VectorX<scalar_t> lb_x;
		Eigen::VectorX<scalar_t> ub_x;

		ProblemMemory(size_t num_constraints, size_t num_variables) :
			var(num_variables),
			g(num_constraints),
			lb(num_constraints), ub(num_constraints),
			lb_x(num_variables), ub_x(num_variables)
			{}

	  // Eigen::VectorX<scalar_t> obj_gradient;
	  // Eigen::SparseMatrix<scalar_t> obj_hessian;
	  // Eigen::VectorX<scalar_t> obj_weight;
	};

	/**
	 * Allocates and associates problem memory for a given problem.
	 */
	ProblemMemory make_problem_memory()
	{
		ProblemMemory mem(num_constraints, num_variables);
		constraint.jacobian.allocate_memory(mem.g_jacobian);

		set_memory_targets(mem.var, mem.g, mem.g_jacobian, mem.lb, mem.ub, mem.lb_x, mem.ub_x);

		mem.var.array() = 0;
		mem.g.array() = 0;
		for(int i=0; i<mem.g_jacobian.nonZeros(); i++) mem.g_jacobian.valuePtr()[i] = 0;
		mem.lb.array() = 0;
		mem.ub.array() = 0;
		mem.lb_x.array() = 0;
		mem.ub_x.array() = 0;

		return mem;
	}

};


// /**
//  * Creates and runs the tape for a user-defined class, and then 
//  * creates the given executable object from the tape.
//  * 
//  * Assumes that the user-defined class contains two functions:
//  * 	eval_constraints
//  * 	eval_objective
//  * 
//  * Calling form:
//  *   auto prob = makeProblem<UserClass, double>(10);
//  */
// template<template<typename, typename> class P, typename scalar_t, typename... Args>
// P<scalar_t, lampc::Problem<scalar_t>> make_problem(Args... args)
// {
//   // Create the tape to record the copy sequence
//   P<scalar_t, lampc::ProblemTape<scalar_t>> tape(args...);

// 	// Temporary optimization variable used to call all functions 
// 	// when creating the tape
//   Eigen::VectorX<scalar_t> var(tape.size);
//   tape.set_variable(var);
  
//   tape.template eval_constraints<lampc::Jacobian>(); // Create structure
//   tape.constraints.finalize_structure();
//   tape.template eval_constraints<lampc::Jacobian>(); // Create copy-sequence

//   // tape.template eval_objective<lampc::Hessian>(); // Create structure
//   // tape.objective.finalize_structure();
//   // tape.template eval_objective<lampc::Hessian>(); // Create copy-sequence

//   // We now create and initialize the problem from the tape
//   P<scalar_t, lampc::Problem<scalar_t>> prob(args...);
//   prob.initialize_from_tape(tape);

//   return prob;
// }




};

#endif // __PROBLEM_HPP