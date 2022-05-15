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
 * The return type of a constraint "add" function. 
 * 
 * Contains references to the upper/lower bounds for the object, 
 * the value and jacobian and the indices.
 * 
 * Can be saved by the user in order to refer to the constraint
 * later on, or used to set the constraints.
 */
template<typename scalar_t>
struct BoundRef
{
	Eigen::Ref<Eigen::VectorX<scalar_t>> lb;
	Eigen::Ref<Eigen::VectorX<scalar_t>> ub;

	BoundRef(
		Eigen::Ref<Eigen::VectorX<scalar_t>> _lb,
		Eigen::Ref<Eigen::VectorX<scalar_t>> _ub)
			:	lb(_lb), ub(_ub)
	{}
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
	template<typename F, typename... Vars, typename scalar_t = meta::get_scalar_t<Vars...>>
	BoundRef<scalar_t> add(lampc::Eval, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(value.rows(), fix<num_outputs>);
		this->extend(num_outputs, 0);

    f(lampc::Eval(), value(out_indices), vars...);
    return BoundRef<scalar_t>(lb(out_indices), ub(out_indices));
	}

	template<typename F, typename... Vars, typename scalar_t = meta::get_scalar_t<Vars...>>
	BoundRef<scalar_t> add(lampc::Jacobian, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(value.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend(num_outputs, 0);

    f(lampc::Jacobian(), value(out_indices), jacobian(out_indices, in_indices), vars...);
    return BoundRef<scalar_t>(lb(out_indices), ub(out_indices));
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
template<typename scalar_t, typename Derived>
BoundRef<scalar_t> operator<=(BoundRef<scalar_t> f, const Eigen::MatrixBase<Derived>& ub) 
{ f.ub = ub; return f; }

template<typename scalar_t, typename Derived>
BoundRef<scalar_t> operator<=(const Eigen::MatrixBase<Derived>& lb, BoundRef<scalar_t> f)
{ f.lb = lb; return f; }

template<typename scalar_t, typename Derived>
BoundRef<scalar_t> operator>=(const Eigen::MatrixBase<Derived>& ub, BoundRef<scalar_t> f)
{ f.ub = ub; return f; }

template<typename scalar_t, typename Derived>
BoundRef<scalar_t> operator>=(BoundRef<scalar_t> f, const Eigen::MatrixBase<Derived>& lb)
{ f.lb = lb; return f; }

template<typename scalar_t, typename Derived>
BoundRef<scalar_t> operator==(BoundRef<scalar_t> f, const Eigen::MatrixBase<Derived>& eq)
{
	f.ub = eq;
	f.lb = eq;
	return f;
}
template<typename scalar_t, typename Derived>
BoundRef<scalar_t> operator==(const Eigen::MatrixBase<Derived>& eq, BoundRef<scalar_t> f)
{
	f.ub = eq;
	f.lb = eq;
	return f;
}

// Broadcast versions
template<typename scalar_t, typename Scalar>
BoundRef<scalar_t> operator<=(BoundRef<scalar_t> f, const Scalar ub) 
{ f.ub.array() = ub; return f; }

template<typename scalar_t, typename Scalar>
BoundRef<scalar_t> operator<=(const Scalar lb, BoundRef<scalar_t> f)
{ f.lb.array() = lb; return f; }

template<typename scalar_t, typename Scalar>
BoundRef<scalar_t> operator>=(const Scalar ub, BoundRef<scalar_t> f)
{ f.ub.array() = ub; return f; }

template<typename scalar_t, typename Scalar>
BoundRef<scalar_t> operator>=(BoundRef<scalar_t> f, const Scalar lb)
{ f.lb.array() = lb; return f; }

template<typename scalar_t, typename Scalar>
BoundRef<scalar_t> operator==(BoundRef<scalar_t> f, const Scalar eq)
{
	f.ub.array() = eq;
	f.lb.array() = eq;
	return f;
}
template<typename scalar_t, typename Scalar>
BoundRef<scalar_t> operator==(const Scalar eq, BoundRef<scalar_t> f)
{
	f.ub.array() = eq;
	f.lb.array() = eq;
	return f;
}


/**
 * A class used by the WeightedSum class to enumlate
 * the behaviour of BoundRef when a constraint is being 
 * called as a weighted sum (i.e, for the lagrangian)
 * 
 * In this case, we want any calls on the bounds to just
 * be ignored.
 */
struct BoundRefFake
{
	struct ignoreme
	{
		BoundRefFake& operator=(BoundRefFake& other)  { return other; }
		BoundRefFake& operator=(BoundRefFake&& other) { return other; }
	};

	ignoreme lb;
	ignoreme ub;

	BoundRefFake() {}
};

/**
 * Constraint functions
 */
template<typename Derived>
BoundRefFake operator<=(BoundRefFake f, const Eigen::MatrixBase<Derived>& ub) { return f; }

template<typename Derived>
BoundRefFake operator<=(const Eigen::MatrixBase<Derived>& lb, BoundRefFake f) { return f; }

template<typename Derived>
BoundRefFake operator>=(const Eigen::MatrixBase<Derived>& ub, BoundRefFake f) { return f; }

template<typename Derived>
BoundRefFake operator>=(BoundRefFake f, const Eigen::MatrixBase<Derived>& lb) { return f; }

template<typename Derived>
BoundRefFake operator==(BoundRefFake f, const Eigen::MatrixBase<Derived>& eq) { return f; }

template<typename Derived>
BoundRefFake operator==(const Eigen::MatrixBase<Derived>& eq, BoundRefFake f) { return f; }

template<typename scalar_t>
BoundRefFake operator<=(BoundRefFake f, const scalar_t ub) { return f; }
template<typename scalar_t>
BoundRefFake operator<=(const scalar_t lb, BoundRefFake f) { return f; }
template<typename scalar_t>
BoundRefFake operator>=(const scalar_t ub, BoundRefFake f) { return f; }
template<typename scalar_t>
BoundRefFake operator>=(BoundRefFake f, const scalar_t lb) { return f; }
template<typename scalar_t>
BoundRefFake operator==(BoundRefFake f, const scalar_t eq) { return f; }
template<typename scalar_t>
BoundRefFake operator==(const scalar_t eq, BoundRefFake f) { return f; }




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
	BoundRefFake add(lampc::Eval, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		this->extend(num_outputs, 0);

    value += f.weightedsum(lampc::Eval(), weights(out_indices), value(out_indices), vars...);
    return BoundRefFake();
	}

	template<typename F, typename... Vars>
	BoundRefFake add(lampc::Gradient, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend(num_outputs, 0);

    value += f.weightedsum(lampc::Gradient(), gradient(in_indices), weights(out_indices), vars...);
    return BoundRefFake();
	}

	template<typename F, typename... Vars>
	BoundRefFake add(lampc::Hessian, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend(num_outputs, 0);

    value += f.weightedsum(lampc::Hessian(), 
    	       							 gradient(in_indices), hessian(in_indices, in_indices), 
    											 weights(out_indices), vars...);
    return BoundRefFake();
	}

	// Defaults to hessian computation
	template<typename F, typename... Vars>
	BoundRefFake add(F f, Vars... vars)
	{
		return add(lampc::Hessian(), f, vars...);
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
	WeightedSumInfo<Matrix,Vector> lagrangian;
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
		lb_x.resize(0,1);
		ub_x.resize(0,1);
	}

	/**
	 * Constructor for tape recording and deployment
	 */
	template<typename _Matrix, typename _Vector>
	ProblemBase(const ProblemInfo<_Matrix,_Vector>& info) :
		constraints(info.constraints), objective(info.objective), lagrangian(info.lagrangian), lb_x(info.lb_x), ub_x(info.ub_x)
	{
		lb_x.resize(0,1);
		ub_x.resize(0,1);
	}

	VectorFunction<Matrix, Vector> constraints;
	WeightedSum<Matrix, Vector> objective;
	WeightedSum<Matrix, Vector> lagrangian;

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
		lagrangian.extend(0, n);
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
		info.lagrangian = lagrangian.generate();

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

  ocp.eval_constraints(Jacobian(), prob.constraints);
  ocp.eval_objective(Hessian(), prob.objective);

  ocp.eval_objective(Hessian(), prob.lagrangian);
  ocp.eval_constraints(Hessian(), prob.lagrangian);

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
	FunctionMemory(VectorFunction<Matrix,Vector>& f, FunctionInfo<_Matrix,_Vector>& info) :
		value(info.rows,1), lb(info.rows,1), ub(info.rows,1)
	{
		f.jacobian.allocate_memory(jacobian);

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
	WeightedSumMemory(WSum& w, Info& info) :
        gradient(info.gradient.rows),
        weights(info.weights.rows)
	{
		w.hessian.allocate_memory(hessian);

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
	WeightedSumMemory<scalar_t> lagrangian;

  Eigen::VectorX<scalar_t> var;

  Eigen::VectorX<scalar_t> lb_x;
  Eigen::VectorX<scalar_t> ub_x;

	template<typename Problem, typename Info>
	ProblemMemory(Problem& prob, Info& info) :
			constraints(prob.constraints, info.constraints),
      objective(prob.objective, info.objective),
			lagrangian(prob.lagrangian, info.lagrangian),
			lb_x(info.num_variables), ub_x(info.num_variables),
			var(prob.num_variables())
	{
		// Zero everything
		prob.constraints.initialize();
		prob.objective.initialize();
		prob.lagrangian.initialize();
	  var.array() = 0;
	  prob.set_decision_variable(var);

		prob.lb_x.set_buffer(lb_x);
		prob.ub_x.set_buffer(ub_x);

	  for(int i=0; i<objective.hessian.nonZeros(); i++) objective.hessian.valuePtr()[i] = i;
	  for(int i=0; i<constraints.jacobian.nonZeros(); i++) constraints.jacobian.valuePtr()[i] = i;
	  for(int i=0; i<lagrangian.hessian.nonZeros(); i++) lagrangian.hessian.valuePtr()[i] = i;
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
  Eigen::SparseMatrix<bool> objective_sparsity_structure = (info.objective.hessian.array() > 0).matrix().sparseView();  
  o << "  Non-zeros  : " << objective_sparsity_structure.nonZeros() << std::endl;

	o << "Lagrangian    : " << info.lagrangian.hessian.rows() << std::endl;
  Eigen::SparseMatrix<bool> lagrangian_sparsity_structure = (info.lagrangian.hessian.array() > 0).matrix().sparseView();  
  o << "  Non-zeros  : " << lagrangian_sparsity_structure.nonZeros() << std::endl;
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
	o << "Lagrangian   : " << info.lagrangian.rows << std::endl;
  o << "  Non-zeros  : " << info.lagrangian.hessian.sparsity_structure.nonZeros() << std::endl;
  o << "  Tape length: " << info.lagrangian.hessian.copy_segments.size() << std::endl;

  return o;
}



template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}



};

#endif // __PROBLEM_HPP