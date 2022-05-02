#ifndef __PROBLEM_HPP
#define __PROBLEM_HPP

#include <numeric>

namespace lampc
{

/**
 * We create three copies of the problem (and the contained constraints, objective, etc). 
 * 
 * The first is a Sparsity version, in which the variable information is also propagated with every function
 * call and passed to BSMatrixTape objects to develop the spasity structures.
 *
 * The second is a Tape version, where the same process is followed to record the copy operations.
 * 
 * The third is the deployment version, which doesn't have the variable information and
 * just plays back the tape for speed. Only this version needs to be optimized.
 */

/**
 * A single variable from a set of variables.
 */
template<typename scalar_t>
struct VarSlice
{
	Segment segment; // Offset into the global variable
	Eigen::Ref<Eigen::VectorX<scalar_t>> vec;

	VarSlice(size_t index, size_t length, Eigen::Ref<Eigen::VectorX<scalar_t>> vec)
	: segment{.index=index, .length=length}, vec(vec) {}

	// Pass through all eigen indexing calls to vec
	template<typename... Args>
	EIGEN_STRONG_INLINE auto operator()(Args... args) -> decltype(vec(args...)) {return vec(args...);}
 };


/**
 * Represents a set of variables, which is an offset into
 * a global decision variable.
 */
template<typename scalar_t_>
struct VariableBase
{
	using scalar_t = scalar_t_;

	VariableBase(int n, int m, int offset) 
		: n(n), m(m), offset(offset), 
			src(NULL,n,m) // Variable<scalar_t> is invalid until set_var is called
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

	/** 
	 * Return a reference to the index'th variable in the matrix and the segment info
	 */
	inline VarSlice<scalar_t> operator()(int index)
	{
		assert(this->src_valid && "set_var must be called before using this variable");
		return VarSlice<scalar_t>(this->offset + index * this->n, this->n, this->src(Eigen::all,index));
	}

protected:

	// Offset into the global variable
	int offset;
	int n,m;

	// Reference to start of this variable in the global variable
	Eigen::Map<Eigen::MatrixX<scalar_t>> src;

	bool src_valid = false; // Used for debugging. Set only if src is valid.
};


template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}

/**
 * Vector function g(x) = [g1(x); ...; gN(x)]
 * 
 * Can add block-rows to the vector function and evaluate the function and its' jacobian
 */
template<typename SparseMatrixType, typename DenseMatrixType>
struct ConstraintBase
{
	// Used for sparsity discovery
	ConstraintBase()
	{}

	void initialize()
	{
		// Reset all sizes so that the "last" keywords work
		value.resize(0,1);
		lb.resize(0,1);
		ub.resize(0,1);
		lb_x.resize(0,1);
		ub_x.resize(0,1);

		jacobian.reset_copy_index();
	}

	// Sparsity structure is specified
	ConstraintBase(Eigen::MatrixX<bool> jacobian_sparsity_structure)
		: jacobian(jacobian_sparsity_structure)
	{
		initialize();
	}

	// Copy sequence is specified
	ConstraintBase(BSMatrixInfo jacobian_info)
		: jacobian(jacobian_info)
	{
		initialize();
	}

	DenseMatrixType value;
	SparseMatrixType jacobian;

	// Upper/lower bounds of the constraints
	DenseMatrixType lb;
	DenseMatrixType ub;

	// Upper/lower bounds of the variables
	DenseMatrixType lb_x;
	DenseMatrixType ub_x;

	/**
	 * Add a block-row to the constraint
	 * 
	 * Vars are of type VarSlice<scalar_t>'s
	 */
	template<typename F, typename... Vars>
	void add(lampc::Eval, F& f, Vars... x)
	{
		// Push a new constraint on the bottom, and set the columns
		auto out_indices = seqN(value.rows(), fix<F::num_outputs>);
		this->add_constraint(f.num_outputs);

		// Copy the value and jacobian to the right place
		value(out_indices,0) = f(lampc::Eval(), x.vec...);
	}

	template<typename F, typename... Vars>
	void add(lampc::Jacobian, F& f, Vars... x)
	{
		auto out_indices = seqN(value.rows(), fix<F::num_outputs>);
		this->add_constraint(f.num_outputs);

		// // Verion 1: Fixed-sized call. Does not exploit any sparsity in the function, but better for small functions.
		// auto out = f(lampc::Jacobian(), x.vec...);
		// value(out_indices,0) = std::get<0>(out);
		// this->jacobian(out_indices, lampc::multiSeq_to_index<F::num_inputs>({x.segment...})) = std::get<1>(out);

		// Verion 2: Pass the output matrices to the function, which can then exploit sparsity.
		//   From testing, this will only be faster if the function is *very* sparse
		f(x.vec..., 
			value(out_indices,0),
			this->jacobian(out_indices, lampc::multiSeq_to_index<F::num_inputs>({x.segment...}))
			);
	}

	// Default evaluation is to add the constraint with jacobian computation
	template<typename F, typename... Vars>
	void operator()(F& f, Vars... x)
	{
		add(lampc::Jacobian(), f, x...);
	}

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
 * WeightedSum defines a function of the form
 * 
 * f(x) = sum_i w_i * f_i(x)
 * 
 */
template<typename SparseMatrixType, typename DenseMatrixType>
struct WeightedSumBase
{
	void initialize()
	{
		// Reset all sizes so that the "last" keywords work
		weight.resize(0,1);
		gradient.resize(0,1);
		hessian.reset_copy_index();
	}

	// Used for sparsity discovery
	WeightedSumBase()
	{
		initialize();
	}

	// Sparsity structure is specified
	WeightedSumBase(Eigen::MatrixX<bool> hessian_sparsity_structure)
		: hessian(hessian_sparsity_structure)
	{
		initialize();
	}

	// Copy sequence is specified
	WeightedSumBase(BSMatrixInfo hessian_info)
		: hessian(hessian_info)
	{
		initialize();
	}

	using scalar_t = typename DenseMatrixType::scalar_t;
	scalar_t value;
	DenseMatrixType gradient;
	SparseMatrixType hessian;

	DenseMatrixType weight;

	void add_constraint(int rows)
	{
		weight.extend(rows, 0);
	}

	void add_variable(int var_size)
	{
		gradient.extend(var_size, 0);
		hessian.extend(var_size, var_size);
	}

	/**
	 * Add a summand to the sum
	 * 
	 * Vars are of type VarSlice<scalar_t>'s
	 */
	template<typename F, typename... Vars>
	void add(lampc::Eval, F& f, Vars... x)
	{
		this->add_constraint(f.num_outputs);
		auto con_rows = seqN(lastp1-f.num_outputs,f.num_outputs);

		// Copy the value and jacobian to the right place
		value += f.weightedsum(x.vec..., weight(con_rows, 0));
	}

	template<typename F, typename... Vars>
	void add(lampc::Gradient, F& f, Vars... x)
	{
		this->add_constraint(f.num_outputs);
		auto con_rows = seqN(lastp1-f.num_outputs,f.num_outputs);

		// Copy the value and jacobian to the right place
		value += f.weightedsum(x.vec..., gradient(multiSeq(x.segment.seq()...),0),
													 weight(con_rows,0));
	}

	template<typename F, typename... Vars>
	void add(lampc::Hessian, F& f, Vars... x)
	{
		this->add_constraint(f.num_outputs);
		auto con_rows = seqN(lastp1-f.num_outputs,f.num_outputs);

		// Copy the value and jacobian to the right place
		value += f.weightedsum(x.vec..., 
									gradient(multiSeq(x.segment.seq()...),0),
									hessian(multiSeq(x.segment.seq()...), multiSeq(x.segment.seq()...)),
									weight(con_rows,0));
	}

	// Default evaluation is to add the constraint with jacobian computation
	template<typename F, typename... Vars>
	void operator()(F& f, Vars... x)
	{
		add(lampc::Gradient(), f, x...);
	}

};

/**
 * Captures all information about the sparsity pattern of a problem.
 * Input to the ProblemTape.
 */
struct ProblemSparsityPattern
{
	// ProblemSparsityPattern(filename) // Load from file

  Eigen::MatrixX<bool> constraints_jacobian;
  Eigen::MatrixX<bool> obj_hessian;
	size_t obj_num_weights; // Number of weights in the objective
};


/**
 * Captures all information required to generate a Problem.
 * Input to the Problem.
 */
struct ProblemInfo
{
	// ProblemInfo(filename) // Load from file

	BSMatrixInfo constraints_jacobian;
	BSMatrixInfo obj_hessian;
	size_t obj_num_weights; // Number of weights in the objective
};



template<typename _Variable, typename _Constraint, typename _Objective>
struct ProblemBase
{
	// Construct Sparsity discovery
	ProblemBase() 
	{}

	// Construct Tape recording
	ProblemBase(ProblemSparsityPattern pattern) : 
		constraint(pattern.constraints_jacobian),
		objective(pattern.obj_hessian) 
	{}

	// Construct Problem
	ProblemBase(ProblemInfo info) : 
		constraint(info.constraints_jacobian), 
		objective(info.obj_hessian) 
	{}


	using Variable = _Variable;
	using Constraint = _Constraint;
	using Objective = _Objective;
	using scalar_t = typename Variable::scalar_t;

	Constraint constraint;
	Objective objective;

	std::shared_ptr<Variable> make_variable(int n, int m=1)
	{
		auto v = std::make_shared<Variable>(n,m,size);
		variables.push_back(v);
		size = compute_size(); // Update the total size of the problem variable

	  // Grow all the problem elements
		constraint.add_variable(n * m);
		objective.add_variable(n * m);
		
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
template<typename Variable, typename Constraint, typename Objective>
struct ProblemConstruction : public ProblemBase<Variable, Constraint, Objective>
{
	using Base = ProblemBase<Variable, Constraint, Objective>;

	ProblemConstruction() {}
	ProblemConstruction(ProblemSparsityPattern pattern) : Base(pattern) {}

	Eigen::VectorX<typename Variable::scalar_t> var;

	/**
	 * For the construction classes, we also provide the memory used for evaluation within problem.
	 * As a result, when a new variable is created, we need to re-allocate memory for var
	 */
	Variable& variable(int n, int m=1)
	{
		std::shared_ptr<Variable> new_variable = Base::make_variable(n,m);

		// Grow the size of the variable
		var.resize(this->size);
	  this->set_variable(var);

		return *(new_variable.get());
	}
};

template<typename scalar_t, 
				 typename Variable = VariableBase<scalar_t>,
				 typename Constraint = ConstraintBase<BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>,
				 typename Objective = WeightedSumBase<BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>>
struct ProblemTape : public ProblemConstruction<Variable, Constraint, Objective>
{
	using Base = ProblemConstruction<Variable, Constraint, Objective>;
	using Base::constraint;
	using Base::objective;
	ProblemTape(ProblemSparsityPattern pattern) : Base(pattern)
	{}

	ProblemInfo generate()
	{
		ProblemInfo info;

		info.constraints_jacobian = constraint.jacobian.generate();
		info.obj_hessian = objective.hessian.generate();
		info.obj_num_weights = objective.weight.cols();

		return info;
	}
};

template<typename scalar_t, 
				 typename Variable = VariableBase<scalar_t>,
				 typename Constraint = ConstraintBase<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>,
				 typename Objective = WeightedSumBase<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>>
struct ProblemSparsity : public ProblemConstruction<Variable, Constraint, Objective>
{
	/**
	 * Produce sparsity data for this problem
	 */
	ProblemSparsityPattern generate()
	{
		ProblemSparsityPattern pattern;
		pattern.constraints_jacobian = this->constraint.jacobian.get_sparsity();
		pattern.obj_hessian = this->objective.hessian.get_sparsity();
		pattern.obj_num_weights = this->objective.weight.cols();
		return pattern;
	}

};

template<typename scalar_t, 
				 typename Variable = VariableBase<scalar_t>,
				 typename Constraint = ConstraintBase<BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>,
				 typename Objective = WeightedSumBase<BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>>
struct Problem : public ProblemBase<Variable, Constraint, Objective>
{
	using Base = ProblemBase<Variable, Constraint, Objective>;
	using Base::constraint;
	using Base::objective;

	const size_t num_constraints;
	const size_t num_variables;
	const size_t obj_num_weights; // Number of weights in the objective

	Problem(ProblemInfo info) : Base(info),
		num_constraints(constraint.jacobian.rows()),
		num_variables(constraint.jacobian.cols()),
		obj_num_weights(info.obj_num_weights)
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
		Eigen::Ref<Eigen::VectorX<scalar_t>> ub_x,
		Eigen::Ref<Eigen::VectorX<scalar_t>> obj_gradient,
		Eigen::SparseMatrix<scalar_t>& obj_hessian,
		Eigen::Ref<Eigen::VectorX<scalar_t>> obj_weight)
	{
		std::cout << "set_memory_targets\n";
    this->set_variable(var);

    this->constraint.value.set_buffer(g);
    this->constraint.jacobian.set_target(g_jacobian);
		this->constraint.lb.set_buffer(lb);
		this->constraint.ub.set_buffer(ub);
		this->constraint.lb_x.set_buffer(lb_x);
		this->constraint.ub_x.set_buffer(ub_x);
    this->objective.gradient.set_buffer(obj_gradient);
    this->objective.hessian.set_target(obj_hessian);
    this->objective.weight.set_buffer(obj_weight);
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

		Eigen::VectorX<scalar_t> obj_gradient;
		Eigen::SparseMatrix<scalar_t> obj_hessian;
		Eigen::VectorX<scalar_t> obj_weight;

		ProblemMemory(size_t num_constraints, size_t num_variables, size_t obj_num_weights) :
			var(num_variables),
			g(num_constraints),
			lb(num_constraints), ub(num_constraints),
			lb_x(num_variables), ub_x(num_variables),
			obj_gradient(num_variables),
			obj_weight(obj_num_weights)
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
		ProblemMemory mem(num_constraints, num_variables, obj_num_weights);
		constraint.jacobian.allocate_memory(mem.g_jacobian);
		objective.hessian.allocate_memory(mem.obj_hessian);

		set_memory_targets(mem.var, mem.g, mem.g_jacobian, mem.lb, mem.ub, mem.lb_x, mem.ub_x,
											 mem.obj_gradient, mem.obj_hessian, mem.obj_weight);

		mem.var.array() = 0;
		mem.g.array() = 0;
		for(int i=0; i<mem.g_jacobian.nonZeros(); i++) mem.g_jacobian.valuePtr()[i] = 0;
		mem.lb.array() = 0;
		mem.ub.array() = 0;
		mem.lb_x.array() = 0;
		mem.ub_x.array() = 0;

		mem.obj_gradient.array() = 0;
		for(int i=0; i<mem.obj_hessian.nonZeros(); i++) mem.obj_hessian.valuePtr()[i] = 0;
		mem.obj_weight.array() = 0;

		return mem;
	}

};

};

#endif // __PROBLEM_HPP