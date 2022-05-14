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
 * A Variable is just an Eigen Map that also maintains an index vector into the global
 * decision variable
 */
template<typename _scalar_t, int _size>
class Variable : public IndexedVector<Eigen::Map<Eigen::Vector<_scalar_t, _size>>>
{
	using scalar_t = _scalar_t;
	static constexpr int m_size = _size;
	using Base = IndexedVector<Eigen::Map<Eigen::Vector<_scalar_t, _size>>>;

	using Base::Base;

public:
	Variable() : Base(NULL) // The map initially points to NULL
		{}

	constexpr int size() { return m_size; }
	bool is_valid() { return this->data() != NULL; }

	/**
	 * offset = offset of this variable into the global decision variable.
	 * 
	 * Returns a callback function used to set the optimization variable
	 */
	auto register_variable(int offset)
	{
		this->set_offset(offset);

		auto p_this = this;
    return [p_this](scalar_t* master_variable) 
    { 
    	new (p_this) Eigen::Map<Eigen::Vector<scalar_t,m_size>>(master_variable + p_this->offset());
    };
	}
};




/**
 * Takes a paramater pack of Eigen::Vector's and concantenantes
 * them into a single Eigen::Vector.
 * Everything must be fixed-size.
 */
template<int... n>
Eigen::Vector<int, meta::sum_template<n...>()>
concantenate_indices(const Eigen::Vector<int, n>&... args)
{
	Eigen::Vector<int, meta::sum_template<n...>()> out;
	int offset = 0;
	auto l = {
		(
			out(seqN(offset,n)) = args,
			offset += n,
			0
		)...
	};
	return out;
}


/**
 * Information about the function.
 * 
 * Either sparsity or tape informaation, depending on Matrix and Vector
 */
template<typename Matrix, typename Vector>
struct FunctionInfo
{
	int rows, variables;

	typename Matrix::Info jacobian;
	typename Vector::Info value;
	typename Vector::Info lb;
	typename Vector::Info ub;
};

/**
 * A vector-valued function with a sparse jacobian.
 */
template<typename Matrix, typename Vector>
class VectorFunction
{
public:
	Vector value;
	Matrix jacobian;
	Vector lb, ub;

	void initialize()
	{
		value.resize(0,1);
		jacobian.resize(0,0);
		lb.resize(0,1);
		ub.resize(0,1);

		value.set_zero();
		jacobian.set_zero();
		lb.set_zero();
		ub.set_zero();
	}

	VectorFunction() { initialize(); }

	/**
	 * Constructor for tape recording or deloyment
	 */
	template<typename _Matrix, typename _Vector>
	VectorFunction(const FunctionInfo<_Matrix, _Vector>& info) :
		value(info.value), jacobian(info.jacobian), lb(info.lb), ub(info.ub)
	{
		initialize();
	}

	/**
	 * Increase the number of rows and/or the number of inputs
	 */
	void extend(int rows, int variables=0)
	{
		value.extend(rows,0);
		lb.extend(rows,0);
		ub.extend(rows,0);		
		jacobian.extend(rows, variables);
	}

	/**
	 * Add constraints to the problem
	 */

	template<typename F, typename... Vars>
	VectorFunction<Matrix,Vector>& operator()(lampc::Eval, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(value.rows(), fix<num_outputs>);
		this->extend(num_outputs, 0);

    f(lampc::Eval(), value(out_indices), vars...);
    return *this;
	}

	template<typename F, typename... Vars>
	VectorFunction<Matrix,Vector>& operator()(lampc::Jacobian, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(value.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend(num_outputs, 0);

    f(lampc::Jacobian(), value(out_indices), jacobian(out_indices, in_indices), vars...);
    return *this;
	}

	// Defaults to jacobian computation
	template<typename F, typename... Vars>
	VectorFunction<Matrix,Vector>& operator()(F f, Vars... vars)
	{
		operator()(lampc::Jacobian(), f, vars...);
    return *this;
	}	

	/**
	 * Set the upper and lower bounds
	 * 
	 * Assumes that the length of ub is the correct size
	 * for the last constraint added. So we set the last
	 * _ub.size() values of ub.
	 */
	template<typename Derived>
	void set_ub(const Eigen::MatrixBase<Derived>& _ub)
	{
		ub(seqN(ub.rows() - _ub.rows(), _ub.rows())) = _ub;
	}
	template<typename Derived>
	void set_lb(const Eigen::MatrixBase<Derived>& _lb)
	{
		lb(seqN(lb.rows() - _lb.rows(), _lb.rows())) = _lb;
	}

	FunctionInfo<Matrix,Vector> generate()
	{
		FunctionInfo<Matrix,Vector> info;

		info.rows = jacobian.rows();
		info.variables = jacobian.cols();

		info.jacobian = jacobian.generate();
		info.value = value.generate();
		info.lb = lb.generate();
		info.ub = ub.generate();

		return info;
	}
};

/**
 * Constraint functions
 */
template<typename Matrix, typename Vector, typename Derived>
VectorFunction<Matrix,Vector>& operator<=(VectorFunction<Matrix,Vector>& f, const Eigen::MatrixBase<Derived>& ub)
{
	f.set_ub(ub);
	return f;
}
template<typename Matrix, typename Vector, typename Derived>
VectorFunction<Matrix,Vector>& operator<=(const Eigen::MatrixBase<Derived>& lb, VectorFunction<Matrix,Vector>& f)
{
	f.set_lb(lb);
	return f;
}
template<typename Matrix, typename Vector, typename Derived>
VectorFunction<Matrix,Vector>& operator>=(const Eigen::MatrixBase<Derived>& ub, VectorFunction<Matrix,Vector>& f)
{
	f.set_ub(ub);
	return f;
}
template<typename Matrix, typename Vector, typename Derived>
VectorFunction<Matrix,Vector>& operator>=(VectorFunction<Matrix,Vector>& f, const Eigen::MatrixBase<Derived>& lb)
{
	f.set_lb(lb);
	return f;
}
template<typename Matrix, typename Vector, typename Derived>
VectorFunction<Matrix,Vector>& operator==(VectorFunction<Matrix,Vector>& f, const Eigen::MatrixBase<Derived>& eq)
{
	f.set_ub(eq);
	f.set_lb(eq);
	return f;
}
template<typename Matrix, typename Vector, typename Derived>
VectorFunction<Matrix,Vector>& operator==(const Eigen::MatrixBase<Derived>& eq, VectorFunction<Matrix,Vector>& f)
{
	f.set_ub(eq);
	f.set_lb(eq);
	return f;
}



/**
 * Information about a weighted sum function.
 * 
 * Either sparsity or tape informaation, depending on Matrix and Vector
 */
template<typename Matrix, typename Vector>
struct WeightedSumInfo
{
	int rows, cols;
	typename Matrix::Info hessian;
	typename Vector::Info gradient;
	typename Vector::Info weights;
};

/**
 * A scalar function of the form f(x) = sum wi fi(x), with a sparse hessian.
 */
template<typename Matrix, typename Vector>
class WeightedSum
{
public:
	using scalar_t = typename Vector::scalar_t;
	Matrix hessian;
	Vector gradient;
	scalar_t value;
	Vector weights;

public:

	/**
	 * Increase the number of rows and/or the number of inputs
	 */
	void extend(int rows, int variables=0)
	{
		hessian.extend(variables, variables);
		gradient.extend(variables, 0);
		weights.extend(rows, 0);
	}

	void initialize()
	{
		hessian.resize(0,0);
		gradient.resize(0,1);
		weights.resize(0,1);

		value = 0;
		gradient.set_zero();
		hessian.set_zero();
	}

	/**
	 * Sparsity discovery constructor
	 */
	WeightedSum() { initialize(); }

	/**
	 * Constructor for tape recording or deloyment
	 */
	template<typename _Matrix, typename _Vector>
	WeightedSum(const WeightedSumInfo<_Matrix, _Vector>& info) :
		hessian(info.hessian), gradient(info.gradient), weights(info.weights)
	{
		initialize();
	}

	template<typename F, typename... Vars>
	void add(lampc::Eval, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		this->extend(num_outputs, 0);

    value += f.weightedsum(lampc::Eval(), weights(out_indices), value(out_indices), vars...);
	}

	template<typename F, typename... Vars>
	void add(lampc::Gradient, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend(num_outputs, 0);

    value += f.weightedsum(lampc::Gradient(), gradient(in_indices), weights(out_indices), vars...);
	}

	template<typename F, typename... Vars>
	void add(lampc::Hessian, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend(num_outputs, 0);

    value += f.weightedsum(lampc::Hessian(), 
    	       							 gradient(in_indices), hessian(in_indices, in_indices), 
    											 weights(out_indices), vars...);
	}

	// Defaults to hessian computation
	template<typename F, typename... Vars>
	void add(F f, Vars... vars)
	{
		add(lampc::Hessian(), f, vars...);
	}	

	WeightedSumInfo<Matrix,Vector> generate()
	{
		WeightedSumInfo<Matrix,Vector> info;

		info.rows = weights.rows();
		info.cols = hessian.cols();

		info.hessian = hessian.generate();
		info.gradient = gradient.generate();
		info.weights = weights.generate();

		return info;
	}
};




template<typename Matrix, typename Vector>
struct ProblemInfo
{
	FunctionInfo<Matrix,Vector> constraints;
	WeightedSumInfo<Matrix,Vector> objective;
	typename Vector::Info lb_x;
	typename Vector::Info ub_x;

	int num_variables;
	int num_constraints;
};

template<typename _scalar_t, typename Matrix, typename Vector>
class ProblemBase
{
public:
	using scalar_t = _scalar_t;

protected:
	// Callbacks used to register the global decision variable with each variable
	std::vector<std::function<void(scalar_t*)>> variable_callbacks;

	int m_num_variables = 0;

	// Upper/lower bounds
	Vector lb_x, ub_x;

public:

	/**
	 * Default constructor for sparsity discovery
	 */
	ProblemBase()
	{
		lb_x.extend(0,1);
		ub_x.extend(0,1);
	}

	/**
	 * Constructor for tape recording and deployment
	 */
	template<typename _Matrix, typename _Vector>
	ProblemBase(const ProblemInfo<_Matrix,_Vector>& info) :
		constraints(info.constraints), objective(info.objective), lb_x(info.lb_x), ub_x(info.ub_x)
	{
		lb_x.extend(0,1);
		ub_x.extend(0,1);
	}

	VectorFunction<Matrix, Vector> constraints;
	WeightedSum<Matrix, Vector> objective;

	/**
	 * Add the variable to the optimization problem.
	 * 
	 * Order in which they're added determines their order in the global
	 * decision variable
	 */
	template<int n>
	void add_variable(Variable<scalar_t, n>& var)
	{
		variable_callbacks.push_back(var.register_variable(m_num_variables));
		m_num_variables += n;

		constraints.extend(0, n);
		objective.extend(0, n);
		lb_x.extend(n, 0);
		ub_x.extend(n, 0);
	}

	int num_variables() { return m_num_variables; }

	/**
	 * Use the memory in var as the global decision variable
	 */	
	void set_decision_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		assert(var.rows() == num_variables() && "Decision variable is the wrong size");		
		for(auto& call : variable_callbacks) call(var.data());
	}

	/**
	 * Generate data for this problem.
	 * 
	 * Calls generate on every matrix / vector of the problem.
	 */
	auto generate()
	{
		ProblemInfo<Matrix,Vector> info;

		info.constraints = constraints.generate();
		info.objective = objective.generate();

		info.lb_x = lb_x.generate();
		info.ub_x = ub_x.generate();

		info.num_variables = num_variables();

		return info;
	}
};

template<typename scalar_t>
using ProblemSparsity = ProblemBase<scalar_t, BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>;

template<typename scalar_t>
using ProblemTape = ProblemBase<scalar_t, BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>;

template<typename scalar_t>
using Problem = ProblemBase<scalar_t, BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>;

template<typename Problem, typename OCP>
auto generate_problem(Problem&& prob, OCP&& ocp)
{
  ocp.define_variables(prob);

  Eigen::VectorX<typename Problem::scalar_t> var(prob.num_variables());
  var.array() = 0;
  prob.set_decision_variable(var);
  ocp.eval_constraints(prob.constraints);
  ocp.eval_objective(prob.objective);

  return prob.generate();
}


/**
 * A helper class that can be used to create all the required memory for 
 * a VectorFunction if the memory isn't held externally.
 */
template<typename scalar_t>
struct FunctionMemory
{
	Eigen::SparseMatrix<scalar_t> jacobian;
	Eigen::VectorX<scalar_t> value;
	Eigen::VectorX<scalar_t> lb;
	Eigen::VectorX<scalar_t> ub;

	template<typename Matrix, typename Vector, typename _Matrix, typename _Vector>
	FunctionMemory(VectorFunction<Matrix,Vector>& f, FunctionInfo<_Matrix,_Vector>& info)
	{
		f.jacobian.allocate_memory(jacobian);
		value.resize(info.rows, 1); f.value.set_buffer(value);
		lb.resize(info.rows, 1); f.lb.set_buffer(lb);
		ub.resize(info.rows, 1); f.ub.set_buffer(ub);

		// Set this memory as the buffer for the function
		f.jacobian.set_target(jacobian);
		f.value.set_buffer(value);
		f.lb.set_buffer(lb);
		f.ub.set_buffer(ub);
	}
};

/**
 * A helper class that can be used to create all the required memory for 
 * a WeightedSum if the memory isn't held externally.
 */
template<typename scalar_t>
struct WeightedSumMemory
{
	Eigen::SparseMatrix<scalar_t> hessian;
	Eigen::VectorX<scalar_t> gradient;
	Eigen::VectorX<scalar_t> weights;

	template<typename WSum, typename Info>
	WeightedSumMemory(WSum& w, Info& info)
	{
		w.hessian.allocate_memory(hessian);
		gradient.resize(info.cols, 1); w.gradient.set_buffer(gradient);
		weights.resize(info.weights.rows, 1); w.weights.set_buffer(weights);

		// Set this memory as the buffer for the function
		w.hessian.set_target(hessian);
		w.gradient.set_buffer(gradient);
		w.weights.set_buffer(weights);
	}
};

/**
 * A helper class that can be used to create all the required memory for 
 * a Problem if the memory isn't held externally.
 */
template<typename scalar_t>
struct ProblemMemory
{
	FunctionMemory<scalar_t> constraints;
	WeightedSumMemory<scalar_t> objective;

  Eigen::VectorX<scalar_t> var;

	template<typename Problem, typename Info>
	ProblemMemory(Problem& prob, Info& info) :
		constraints(prob.constraints, info.constraints), objective(prob.objective, info.objective),
		var(prob.num_variables())
	{
		// Zero everything
		prob.constraints.initialize();
		prob.objective.initialize();
	  var.array() = 0;
	  prob.set_decision_variable(var);

	  for(int i=0; i<objective.hessian.nonZeros(); i++) objective.hessian.valuePtr()[i] = i;
	  for(int i=0; i<constraints.jacobian.nonZeros(); i++) constraints.jacobian.valuePtr()[i] = i;
	}
};



template <typename scalar_t>
std::ostream& operator<<(std::ostream& o, const ProblemInfo<BSMatrixSparsity,BSMatrixDenseConstruction<scalar_t>>& info)
{
	o << "==== Problem Sparsity Information ====\n";
	o << "Variables    : " << info.num_variables << std::endl;

	o << "Constraints  : " << info.constraints.rows << std::endl;
  Eigen::SparseMatrix<bool> sparsity_structure = (info.constraints.jacobian.array() > 0).matrix().sparseView();  
  o << "  Non-zeros  : " << sparsity_structure.nonZeros() << std::endl;

	o << "Objective    : " << info.objective.hessian.rows() << std::endl;
  Eigen::SparseMatrix<bool> hessian_sparsity_structure = (info.objective.hessian.array() > 0).matrix().sparseView();  
  o << "  Non-zeros  : " << hessian_sparsity_structure.nonZeros() << std::endl;

  return o;
}

template <typename scalar_t>
std::ostream& operator<<(std::ostream& o, const ProblemInfo<BSMatrixTape,BSMatrixDenseConstruction<scalar_t>>& info)
{
	o << "==== Problem Tape Information ====\n";
	o << "Variables    : " << info.num_variables << std::endl;
	o << "Constraints  : " << info.constraints.rows << std::endl;
  o << "  Non-zeros  : " << info.constraints.jacobian.sparsity_structure.nonZeros() << std::endl;
  o << "  Tape length: " << info.constraints.jacobian.copy_segments.size() << std::endl;
	o << "Objective    : " << info.objective.rows << std::endl;
  o << "  Non-zeros  : " << info.objective.hessian.sparsity_structure.nonZeros() << std::endl;
  o << "  Tape length: " << info.objective.hessian.copy_segments.size() << std::endl;

  return o;
}



template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}










// /**
//  * Vector function g(x) = [g1(x); ...; gN(x)]
//  * 
//  * Can add block-rows to the vector function and evaluate the function and its' jacobian
//  */
// template<typename SparseMatrixType, typename DenseMatrixType>
// struct ConstraintBase
// {
// 	// Used for sparsity discovery
// 	ConstraintBase()
// 	{}

// 	void initialize()
// 	{
// 		// Reset all sizes so that the "last" keywords work
// 		value.resize(0,1);
// 		lb.resize(0,1);
// 		ub.resize(0,1);
// 		lb_x.resize(0,1);
// 		ub_x.resize(0,1);

// 		jacobian.reset_copy_index();
// 	}

// 	// Sparsity structure is specified
// 	ConstraintBase(Eigen::MatrixX<bool> jacobian_sparsity_structure)
// 		: jacobian(jacobian_sparsity_structure)
// 	{
// 		initialize();
// 	}

// 	// Copy sequence is specified
// 	ConstraintBase(BSMatrixInfo jacobian_info)
// 		: jacobian(jacobian_info)
// 	{
// 		initialize();
// 	}

// 	DenseMatrixType value;
// 	SparseMatrixType jacobian;

// 	// Upper/lower bounds of the constraints
// 	DenseMatrixType lb;
// 	DenseMatrixType ub;

// 	// Upper/lower bounds of the variables
// 	DenseMatrixType lb_x;
// 	DenseMatrixType ub_x;

// 	/**
// 	 * Add a block-row to the constraint
// 	 * 
// 	 * Vars are of type VarSlice<scalar_t>'s
// 	 */
// 	template<typename F, typename... Vars>
// 	void add(lampc::Eval, F& f, Vars... x)
// 	{
// 		auto out_indices = seqN(value.rows(), fix<F::num_outputs>);
// 		this->add_constraint(f.num_outputs);

// 		// Copy the value and jacobian to the right place
// 		f(x.vec..., value(out_indices, 0));
// 	}

	// template<typename F, typename... Vars>
	// void add(lampc::Jacobian, F& f, IndexedVector<Vars>&&... x)
	// {
	// 	auto out_indices = seqN(value.rows(), fix<F::num_outputs>);
	// 	this->add_constraint(f.num_outputs);

	// 	// // Verion 1: Fixed-sized call. Does not exploit any sparsity in the function, but a little better for small functions.
	// 	// auto out = f(lampc::Jacobian(), x.vec...);
	// 	// value(out_indices,0) = std::get<0>(out);
	// 	// this->jacobian(out_indices, lampc::multiSeq_to_index<F::num_inputs>({x.segment...})) = std::get<1>(out);

	// 	// Verion 2: Pass the output matrices to the function, which can then exploit sparsity.
	// 	//   From testing, this will only be faster if the function is *very* sparse
	// 	f(x.vec..., 
	// 		value(out_indices,0),
	// 		this->jacobian(out_indices, lampc::multiSeq_to_index<F::num_inputs>({x.segment...}))
	// 		);
	// }

// 	// Default evaluation is to add the constraint with jacobian computation
// 	template<typename F, typename... Vars>
// 	void operator()(F& f, Vars... x)
// 	{
// 		add(lampc::Jacobian(), f, x...);
// 	}

// 	void add_variable(int var_size)
// 	{
// 		jacobian.extend(0, var_size);
// 		lb_x.extend(var_size, 0);
// 		ub_x.extend(var_size, 0);
// 	}

// 	void add_constraint(int rows)
// 	{
// 		value.extend(rows, 0);
// 		jacobian.extend(rows, 0);

// 		lb.extend(rows, 0);
// 		ub.extend(rows, 0);
// 	}
// };


// /**
//  * WeightedSum defines a function of the form
//  * 
//  * f(x) = sum_i w_i * f_i(x)
//  * 
//  */
// template<typename SparseMatrixType, typename DenseMatrixType>
// struct WeightedSumBase
// {
// 	void initialize()
// 	{
// 		// Reset all sizes so that the "last" keywords work
// 		weight.resize(0,1);
// 		gradient.zero_buffer();
// 		gradient.resize(0,1);
// 		hessian.set_zero();
// 		hessian.reset_copy_index();
// 		value = 0;
// 	}

// 	// Used for sparsity discovery
// 	WeightedSumBase()
// 	{
// 		initialize();
// 	}

// 	// Sparsity structure is specified
// 	WeightedSumBase(Eigen::MatrixX<bool> hessian_sparsity_structure)
// 		: hessian(hessian_sparsity_structure)
// 	{
// 		initialize();
// 	}

// 	// Copy sequence is specified
// 	WeightedSumBase(BSMatrixInfo hessian_info)
// 		: hessian(hessian_info)
// 	{
// 		initialize();
// 	}

// 	using scalar_t = typename DenseMatrixType::scalar_t;
// 	scalar_t value;
// 	DenseMatrixType gradient;
// 	SparseMatrixType hessian;

// 	DenseMatrixType weight;

// 	void add_constraint(int rows)
// 	{
// 		weight.extend(rows, 0);
// 	}

// 	void add_variable(int var_size)
// 	{
// 		gradient.extend(var_size, 0);
// 		hessian.extend(var_size, var_size);
// 	}

// 	/**
// 	 * Add a summand to the sum
// 	 * 
// 	 * Vars are of type VarSlice<scalar_t>'s
// 	 */
// 	template<typename F, typename... Vars>
// 	void add(lampc::Eval, F& f, Vars... x)
// 	{
// 		auto con_rows = seqN(weight.rows(), fix<F::num_outputs>);
// 		this->add_constraint(F::num_outputs);

// 		// Copy the value to the right place
// 		value += f.weightedsum(x.vec..., weight(con_rows, 0));
// 	}

// 	template<typename F, typename... Vars>
// 	void add(lampc::Gradient, F& f, Vars... x)
// 	{
// 		auto con_rows = seqN(weight.rows(), fix<F::num_outputs>);
// 		this->add_constraint(F::num_outputs);

// 		// Copy the value and gradient to the right place
// 		value += f.weightedsum(x.vec..., weight(con_rows,0),
// 													 gradient(lampc::multiSeq_to_index<F::num_inputs>({x.segment...}),0));
// 	}

// 	template<typename F, typename... Vars>
// 	void add(lampc::Hessian, F& f, Vars... x)
// 	{
// 		auto con_rows = seqN(weight.rows(), fix<F::num_outputs>);
// 		this->add_constraint(F::num_outputs);

// 		// Copy the value and jacobian to the right place
// 		auto inputs = lampc::multiSeq_to_index<F::num_inputs>({x.segment...});
// 		value += f.weightedsum(x.vec..., weight(con_rows,0),
// 													 gradient(inputs,0), hessian(inputs, inputs));
// 	}

// 	// Default evaluation is to add the constraint with jacobian computation
// 	template<typename F, typename... Vars>
// 	void operator()(F& f, Vars... x)
// 	{
// 		add(lampc::Hessian(), f, x...);
// 	}
// };

// /**
//  * Captures all information about the sparsity pattern of a problem.
//  * Input to the ProblemTape.
//  */
// struct ProblemSparsityPattern
// {
// 	// ProblemSparsityPattern(filename) // Load from file

//   Eigen::MatrixX<bool> constraints_jacobian;
//   Eigen::MatrixX<bool> obj_hessian;
// 	size_t obj_num_weights; // Number of weights in the objective
// };


// /**
//  * Captures all information required to generate a Problem.
//  * Input to the Problem.
//  */
// struct ProblemInfo
// {
// 	// ProblemInfo(filename) // Load from file

// 	BSMatrixInfo constraints_jacobian;
// 	BSMatrixInfo obj_hessian;
// 	size_t obj_num_weights; // Number of weights in the objective
// };



// template<typename _Variable, typename _Constraint, typename _Objective>
// struct ProblemBase
// {
// 	// Construct Sparsity discovery
// 	ProblemBase() 
// 	{}

// 	// Construct Tape recording
// 	ProblemBase(ProblemSparsityPattern pattern) : 
// 		constraint(pattern.constraints_jacobian),
// 		objective(pattern.obj_hessian) 
// 	{}

// 	// Construct Problem
// 	ProblemBase(ProblemInfo info) : 
// 		constraint(info.constraints_jacobian), 
// 		objective(info.obj_hessian) 
// 	{}


// 	using Variable = _Variable;
// 	using Constraint = _Constraint;
// 	using Objective = _Objective;
// 	using scalar_t = typename Variable::scalar_t;

// 	Constraint constraint;
// 	Objective objective;

// 	std::shared_ptr<Variable> make_variable(int n, int m=1)
// 	{
// 		auto v = std::make_shared<Variable>(n,m,size);
// 		variables.push_back(v);
// 		size = compute_size(); // Update the total size of the problem variable

// 	  // Grow all the problem elements
// 		constraint.add_variable(n * m);
// 		objective.add_variable(n * m);
		
// 		return v;
// 	}

// 	Variable& variable(int n, int m=1)
// 	{
// 		// Return with move symantics is required, 
// 		// since we need the reference stored in variables to remain valid
// 		std::shared_ptr<Variable> v = make_variable(n,m);
// 		return *(v.get());
// 	}

// 	int size = 0;

// 	void set_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
// 	{
// 		for(auto v: variables)
// 			v->set_var(var);
// 	}

// private:
// 	std::vector<std::shared_ptr<Variable>> variables;

// 	int compute_size() // Total number of optimization variables
// 	{
// 		return std::accumulate(variables.begin(), variables.end(), 0, 
// 							[](int total, std::shared_ptr<Variable> var) {return total + var->num_elements();});
// 	}
// };

// /**
//  * Common class for the sparsity and tape problems
//  */
// template<typename Variable, typename Constraint, typename Objective>
// struct ProblemConstruction : public ProblemBase<Variable, Constraint, Objective>
// {
// 	using Base = ProblemBase<Variable, Constraint, Objective>;

// 	ProblemConstruction() {}
// 	ProblemConstruction(ProblemSparsityPattern pattern) : Base(pattern) {}

// 	Eigen::VectorX<typename Variable::scalar_t> var;

// 	/**
// 	 * For the construction classes, we also provide the memory used for evaluation within problem.
// 	 * As a result, when a new variable is created, we need to re-allocate memory for var
// 	 */
// 	Variable& variable(int n, int m=1)
// 	{
// 		std::shared_ptr<Variable> new_variable = Base::make_variable(n,m);

// 		// Grow the size of the variable
// 		var.resize(this->size);
// 	  this->set_variable(var);

// 		return *(new_variable.get());
// 	}
// };

// template<typename scalar_t, 
// 				 typename Variable = VariableBase<scalar_t>,
// 				 typename Constraint = ConstraintBase<BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>,
// 				 typename Objective = WeightedSumBase<BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>>
// struct ProblemTape : public ProblemConstruction<Variable, Constraint, Objective>
// {
// 	using Base = ProblemConstruction<Variable, Constraint, Objective>;
// 	using Base::constraint;
// 	using Base::objective;
// 	ProblemTape(ProblemSparsityPattern pattern) : Base(pattern)
// 	{}

// 	ProblemInfo generate()
// 	{
// 		ProblemInfo info;

// 		info.constraints_jacobian = constraint.jacobian.generate();
// 		info.obj_hessian = objective.hessian.generate();
// 		info.obj_num_weights = objective.weight.size();

// 		return info;
// 	}
// };

// template<typename scalar_t, 
// 				 typename Variable = VariableBase<scalar_t>,
// 				 typename Constraint = ConstraintBase<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>,
// 				 typename Objective = WeightedSumBase<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>>
// struct ProblemSparsity : public ProblemConstruction<Variable, Constraint, Objective>
// {
// 	/**
// 	 * Produce sparsity data for this problem
// 	 */
// 	ProblemSparsityPattern generate()
// 	{
// 		ProblemSparsityPattern pattern;
// 		pattern.constraints_jacobian = this->constraint.jacobian.get_sparsity();
// 		pattern.obj_hessian = this->objective.hessian.get_sparsity();
// 		pattern.obj_num_weights = this->objective.weight.size();
// 		return pattern;
// 	}

// };

// template<typename scalar_t, 
// 				 typename Variable = VariableBase<scalar_t>,
// 				 typename Constraint = ConstraintBase<BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>,
// 				 typename Objective = WeightedSumBase<BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>>
// struct Problem : public ProblemBase<Variable, Constraint, Objective>
// {
// 	using Base = ProblemBase<Variable, Constraint, Objective>;
// 	using Base::constraint;
// 	using Base::objective;

// 	const size_t num_constraints;
// 	const size_t num_variables;
// 	const size_t obj_num_weights; // Number of weights in the objective

// 	Problem(ProblemInfo info) : Base(info),
// 		num_constraints(constraint.jacobian.rows()),
// 		num_variables(constraint.jacobian.cols()),
// 		obj_num_weights(info.obj_num_weights)
// 	{}

// 	/**
// 	 * Sets the memory locations that this problem will read/write to.
// 	 * 
// 	 * Assumption: All memory has already been allocated
// 	 */
// 	void set_memory_targets(	  
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> var,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> g,
// 		Eigen::SparseMatrix<scalar_t>& g_jacobian,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> lb,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> ub,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> lb_x,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> ub_x,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> obj_gradient,
// 		Eigen::SparseMatrix<scalar_t>& obj_hessian,
// 		Eigen::Ref<Eigen::VectorX<scalar_t>> obj_weight)
// 	{
// 		std::cout << "set_memory_targets\n";
//     this->set_variable(var);

//     this->constraint.value.set_buffer(g);
//     this->constraint.jacobian.set_target(g_jacobian);
// 		this->constraint.lb.set_buffer(lb);
// 		this->constraint.ub.set_buffer(ub);
// 		this->constraint.lb_x.set_buffer(lb_x);
// 		this->constraint.ub_x.set_buffer(ub_x);
//     this->objective.gradient.set_buffer(obj_gradient);
//     this->objective.hessian.set_target(obj_hessian);
//     this->objective.weight.set_buffer(obj_weight);
// 	}

// 	/**
// 	 * The Problem class doesn't own any of the memory for the problem.
// 	 * This is done because most solvers (e.g., ipopt) own their own
// 	 * memory.
// 	 * 
// 	 * This class is a helper that provides a full set of memory for a given
// 	 * problem.
// 	 */
// 	struct ProblemMemory
// 	{
// 	  Eigen::VectorX<scalar_t> var;

// 		Eigen::VectorX<scalar_t> g;
// 		Eigen::SparseMatrix<scalar_t> g_jacobian;
// 		Eigen::VectorX<scalar_t> lb;
// 		Eigen::VectorX<scalar_t> ub;
// 		Eigen::VectorX<scalar_t> lb_x;
// 		Eigen::VectorX<scalar_t> ub_x;

// 		Eigen::VectorX<scalar_t> obj_gradient;
// 		Eigen::SparseMatrix<scalar_t> obj_hessian;
// 		Eigen::VectorX<scalar_t> obj_weight;

// 		ProblemMemory(size_t num_constraints, size_t num_variables, size_t obj_num_weights) :
// 			var(num_variables),
// 			g(num_constraints),
// 			lb(num_constraints), ub(num_constraints),
// 			lb_x(num_variables), ub_x(num_variables),
// 			obj_gradient(num_variables),
// 			obj_weight(obj_num_weights)
// 			{}

// 	  // Eigen::VectorX<scalar_t> obj_gradient;
// 	  // Eigen::SparseMatrix<scalar_t> obj_hessian;
// 	  // Eigen::VectorX<scalar_t> obj_weight;
// 	};

// 	/**
// 	 * Allocates and associates problem memory for a given problem.
// 	 */
// 	ProblemMemory make_problem_memory()
// 	{
// 		ProblemMemory mem(num_constraints, num_variables, obj_num_weights);
// 		constraint.jacobian.allocate_memory(mem.g_jacobian);
// 		objective.hessian.allocate_memory(mem.obj_hessian);

// 		set_memory_targets(mem.var, mem.g, mem.g_jacobian, mem.lb, mem.ub, mem.lb_x, mem.ub_x,
// 											 mem.obj_gradient, mem.obj_hessian, mem.obj_weight);

// 		mem.var.array() = 0;
// 		mem.g.array() = 0;
// 		for(int i=0; i<mem.g_jacobian.nonZeros(); i++) mem.g_jacobian.valuePtr()[i] = 0;
// 		mem.lb.array() = 0;
// 		mem.ub.array() = 0;
// 		mem.lb_x.array() = 0;
// 		mem.ub_x.array() = 0;

// 		mem.obj_gradient.array() = 0;
// 		for(int i=0; i<mem.obj_hessian.nonZeros(); i++) mem.obj_hessian.valuePtr()[i] = 0;
// 		mem.obj_weight.array() = 0;

// 		return mem;
// 	}

// };

};

#endif // __PROBLEM_HPP