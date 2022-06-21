#ifndef __PROBLEM_HPP
#define __PROBLEM_HPP

#include <numeric>
#include <iterator>

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
	static constexpr int m_size = _size;
	using Base = IndexedVector<Eigen::Map<Eigen::Vector<_scalar_t, _size>>>;

	using Base::Base;

public:
	using scalar_t = _scalar_t;

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

template<typename Index, typename Function>
struct BoundRef;

template<typename Index, typename Function>
auto make_boundref(Function& f, Index index)
{
	return BoundRef<Index,Function>(f,index);
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

template<typename scalar_t>
struct FunctionMemory;

/**
 * A vector-valued function with a sparse jacobian.
 */
template<typename Matrix, typename Vector>
class VectorFunction
{
	using scalar_t = typename Vector::scalar_t;

	// Only set in deployment
	int m_rows = -1;

public:
	Vector value;
	Matrix jacobian;
	Vector lb, ub;
	int num_variables = 0;
	inline int rows() {return m_rows;}

	/**
	 * Must be called before each evaluation
	 * 
	 * Resets the size of the function to zero, but keeps the number of variables.
	 */
	void initialize()
	{
		value.resize(0,1);
		jacobian.resize(0,num_variables);
		lb.resize(0,1);
		ub.resize(0,1);
	}

	/**
	 * Set all values to zero
	 */
	void set_zero()
	{
		value.set_zero();
		jacobian.set_zero();
		lb.set_zero();
		ub.set_zero();
	}

	/**
	 * Sets the memory buffers for all the elements of the VectorFunction
	 * 
	 * Mem must contain appropriate memory elements jacobian, value, lb and ub
	 */
	void set_memory(Eval, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _value, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _lb,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _ub)
	{
		value.set_buffer(_value);
		lb.set_buffer(_lb);
		ub.set_buffer(_ub);
	}

	void set_memory(Jacobian, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _value, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _lb,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _ub,
									Eigen::SparseMatrix<scalar_t>& _jacobian)
	{
		set_memory(Jacobian(), _value,_lb,_ub,Eigen::Map<Eigen::VectorX<scalar_t>>(_jacobian.valuePtr(),_jacobian.nonZeros()));
	}

	void set_memory(Jacobian, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _value, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _lb,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _ub,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _jacobian_buffer)	
	{
		jacobian.set_target(_jacobian_buffer);
		value.set_buffer(_value);
		lb.set_buffer(_lb);
		ub.set_buffer(_ub);
	}

	// Set the memory just for the bounds. (Used for variable_bounds)
	void set_memory(Eigen::Ref<Eigen::VectorX<scalar_t>> _lb, Eigen::Ref<Eigen::VectorX<scalar_t>> _ub)
	{
		lb.set_buffer(_lb);
		ub.set_buffer(_ub);
	}

	// Put computed values into the structure mem
	void set_memory(Eval, FunctionMemory<scalar_t>& mem)
	{
		set_memory(lampc::Eval(), mem.value, mem.lb, mem.ub);
	}
	void set_memory(Jacobian, FunctionMemory<scalar_t>& mem)
	{
		set_memory(lampc::Jacobian(), mem.value, mem.lb, mem.ub, mem.jacobian_buffer);
	}
	void set_memory(FunctionMemory<scalar_t>& mem)
	{
		set_memory(mem.lb, mem.ub);
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
		set_zero();

		m_rows = info.rows;
	}

	/**
	 * Increase the number of rows
	 */
	void extend_rows(int rows)
	{
		value.extend(rows,0);
    jacobian.extend(rows, 0);
		lb.extend(rows,0);
		ub.extend(rows,0);		
	}

	/**
	 * Increase the number of variables
	 */
	void extend_variables(int variables)
	{
		jacobian.extend(0, variables);
		num_variables += variables;
	}


	/**
	 * Add constraints to the problem
	 */
	template<typename F, typename... Vars, typename scalar_t = meta::get_scalar_t<Vars...>>
	auto add(lampc::Eval, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(value.rows(), fix<num_outputs>);
		this->extend_rows(num_outputs);

    f(lampc::Eval(), value(out_indices), vars...);
    return make_boundref(*this, out_indices);
	}

	template<typename F, typename... Vars, typename scalar_t = meta::get_scalar_t<Vars...>>
	auto add(lampc::Jacobian, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(value.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend_rows(num_outputs);

    f(lampc::Jacobian(), value(out_indices), jacobian(out_indices, in_indices), vars...);
    return make_boundref(*this, out_indices);
	}

	/**
	 * Used to set variable bounds
	 */
	template<typename Var, typename scalar_t = typename Var::Scalar>
	auto add(Var var)
	{
    return make_boundref(*this, var.indices());
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

		// std::cout << std::endl << std::endl;
		// std::cout << "info.rows = " << info.rows << std::endl;
		// std::cout << "info.variables = " << info.variables << std::endl;

		// std::cout << "type(info.jacobian) = " << type_name<decltype(info.jacobian)>() << std::endl;
		// // std::cout << "type(info.value) = " << type_name<decltype(info.value)>() << std::endl;
		// // std::cout << "type(info.lb) = " << type_name<decltype(info.lb)>() << std::endl;
		// // std::cout << "type(info.ub) = " << type_name<decltype(info.ub)>() << std::endl;

		// // std::cout << "jacobian.shape = " << info.jacobian.rows() << " x " << info.jacobian.cols() << std::endl;
		// std::cout << "value.shape = " << info.value.rows << " x " << info.value.cols << std::endl;
		// std::cout << "lb.shape = " << info.lb.rows << " x " << info.lb.cols << std::endl;
		// std::cout << "ub.shape = " << info.ub.rows << " x " << info.ub.cols << std::endl;

		return info;
	}
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
template<typename Index, typename Function>
struct BoundRef
{	
	Function& f;
	Index index;

	BoundRef(Function& _f, Index& _index) : f(_f), index(_index)
	{}

	template<typename Derived>
	inline void set_lb(const Eigen::MatrixBase<const Derived>& lb)
	{
		f.lb(index) = lb;
	}

	template<typename Scalar>
	inline void set_lb(const Scalar lb)
	{
		f.lb(index).array() = lb;
	}

	template<typename Derived>
	inline void set_ub(const Eigen::MatrixBase<const Derived>& ub)
	{
		f.ub(index) = ub;
	}

	template<typename Scalar>
	inline void set_ub(const Scalar ub)
	{
		f.ub(index).array() = ub;
	}
};


/**
 * Constraint functions
 */
template<typename Index, typename Function, typename Derived>
BoundRef<Index,Function> operator<=(BoundRef<Index,Function> f, const Derived& ub) 
{ 
	f.set_ub(ub); 
	return f; 
}

template<typename Index, typename Function, typename Derived>
BoundRef<Index,Function> operator<=(const Derived& lb, BoundRef<Index,Function> f)
{ 
	f.set_lb(lb); 
	return f; 
}

template<typename Index, typename Function, typename Derived>
BoundRef<Index,Function> operator>=(BoundRef<Index,Function> f, const Derived& lb)
{ 
	f.set_lb(lb); 
	return f; 
}

template<typename Index, typename Function, typename Derived>
BoundRef<Index,Function> operator>=(const Derived& ub, BoundRef<Index,Function> f)
{ 
	f.set_ub(ub); 
	return f; 
}

template<typename Index, typename Function, typename Derived>
BoundRef<Index,Function> operator==(BoundRef<Index,Function> f, const Derived& eq)
{
	f.set_ub(eq);
	f.set_lb(eq);
	return f;
}
template<typename Index, typename Function, typename Derived>
BoundRef<Index,Function> operator==(const Derived& eq, BoundRef<Index,Function> f)
{
	f.set_ub(eq);
	f.set_lb(eq);
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

template<typename scalar_t>
struct WeightedSumMemory;

/**
 * A scalar function of the form f(x) = sum wi fi(x), with a sparse hessian.
 */
template<typename Matrix, typename Vector>
class WeightedSum
{
	// Only set in deployment
	int m_rows = -1;

public:
	using scalar_t = typename Vector::scalar_t;
	Matrix hessian;
	Vector gradient;
	scalar_t value;
	Vector weights;

	int num_variables = 0;
	inline int rows() {return m_rows;}

public:

	/**
	 * Increase the number of rows in the function
	 */
	void extend_rows(int rows)
	{
		weights.extend(rows, 0);
	}

	/**
	 * Increase the number of rows in the function
	 */
	void extend_variables(int variables)
	{
		hessian.extend(variables, variables);
		gradient.extend(variables, 0);

		num_variables += variables;
	}

	/**
	 * Must be called before each evaluation
	 */
	void initialize()
	{
		hessian.resize(num_variables, num_variables);
		gradient.resize(num_variables, 1);
		weights.resize(0, 1);
	}

	/**
	 * Sets all values to zero
	 */
    void set_zero(Eval)
    {
        value = 0;
    }

    void set_zero(Gradient)
    {
        value = 0;
        gradient.set_zero();
    }

	void set_zero(Hessian)
	{
		value = 0;
		gradient.set_zero();
		hessian.set_zero();
	}

	/**
	 * Sets the memory buffers for all the elements of the WeightedSum
	 */
	void set_memory(Eval, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _weights)
	{
		weights.set_buffer(_weights);
	}

	void set_memory(Gradient, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _weights,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _gradient)
	{
		weights.set_buffer(_weights);
		gradient.set_buffer(_gradient);
	}

	void set_memory(Hessian, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _weights,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _gradient,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _hessian_buffer)
	{
		weights.set_buffer(_weights);
		gradient.set_buffer(_gradient);
		hessian.set_target(_hessian_buffer);
	}

	void set_memory(Hessian, 
									Eigen::Ref<Eigen::VectorX<scalar_t>> _weights,
									Eigen::Ref<Eigen::VectorX<scalar_t>> _gradient,
									Eigen::SparseMatrix<scalar_t>& _hessian)
	{
		set_memory(Hessian(), _weights, _gradient, Eigen::Map<Eigen::VectorX<scalar_t>>(_hessian.valuePtr(), _hessian.nonZeros()));
	}

	void set_memory(Eval, Eigen::Ref<Eigen::VectorX<scalar_t>> weights, WeightedSumMemory<scalar_t>& mem) 
	{
		set_memory(lampc::Eval(), weights);
	}
	
	void set_memory(Gradient, Eigen::Ref<Eigen::VectorX<scalar_t>> weights, WeightedSumMemory<scalar_t>& mem) 
	{
		set_memory(lampc::Eval(), weights, mem.gradient);
	}
	void set_memory(Hessian, Eigen::Ref<Eigen::VectorX<scalar_t>> weights, WeightedSumMemory<scalar_t>& mem) 
	{
		set_memory(lampc::Hessian(), weights, mem.gradient, mem.hessian_buffer);
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
		set_zero(Hessian{});

		m_rows = info.rows;
	}

	template<typename F, typename... Vars>
	BoundRefFake add(lampc::Eval, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		this->extend_rows(num_outputs);

    value += f.weightedsum(lampc::Eval(), weights(out_indices), vars...);
    return BoundRefFake();
	}

	template<typename F, typename... Vars>
	BoundRefFake add(lampc::Gradient, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend_rows(num_outputs);

    value += f.weightedsum(lampc::Gradient(), gradient(in_indices), weights(out_indices), vars...);
    return BoundRefFake();
	}

	template<typename F, typename... Vars>
	BoundRefFake add(lampc::Hessian, F f, Vars... vars)
	{
		static constexpr int num_outputs = FuncInfo<F,Vars...>::num_outputs;

		auto out_indices = seqN(weights.rows(), fix<num_outputs>);
		auto in_indices = concantenate_indices(vars.indices()...);
		this->extend_rows(num_outputs);

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

template<typename Vector>
struct VariableBoundsInfo
{
	int num_variables;
	typename Vector::Info lb;
	typename Vector::Info ub;
};

// /**
//  * An object that holds the bounds for the variables
//  */
// template<typename Vector>
// class VariableBounds
// {
// 	Vector lb, ub;

// public:

// 	VariableBounds()
// 	{
// 		initialize();
// 	}

// 	template<typename Info>
// 	VariableBounds(Info& info) :
// 			lb(info.lb), ub(info.ub)
// 	{
// 		initialize();
// 	}

// 	initialize()
// 	{
// 		lb.resize(0,1);
// 		ub.resize(0,1);
// 	}

// 	void extend_variables(int n)
// 	{
// 		lb.extend(n,0);
// 		ub.extend(n,0);
// 	}

// 	using Info = VariableBoundsInfo<Vector>;
// 	auto generate()
// 	{
// 		Info info;

// 		info.lb = lb.generate();
// 		info.ub = ub.generate();

// 		info.num_variables = num_variables();

// 		return info;
// 	}

// 	void operator()xxxx
// };


template<typename Matrix, typename Vector>
struct ProblemInfo
{
	FunctionInfo<Matrix,Vector> constraints;
	FunctionInfo<Matrix,Vector> variable_bounds;
	WeightedSumInfo<Matrix,Vector> objective;
	WeightedSumInfo<Matrix,Vector> lagrangian;

	int num_variables;
	int num_constraints;
};

template<typename UserCode, typename _scalar_t, typename Matrix, typename Vector>
class ProblemBase
{
public:
	using scalar_t = _scalar_t;

	// Callbacks used to register the global decision variable with each variable
	std::vector<std::function<void(scalar_t*)>> variable_callbacks;

	int m_num_variables = 0;

	/**
	 * UserCode object must define three functions:
	 * 
	 *   template<typename OptProblem>
	 *   void define_variables(OptProblem& problem)
	 * 
	 *   template<typename Constraints, typename Dtype>
	 *   void eval_constraints(Dtype dtype, Constraints& con)
	 * 
	 *   template<typename Objective, typename Dtype>
	 *   void eval_objective(Dtype dtype, Objective& obj)
	 */
	UserCode& usercode;

public:

	/**
	 * Default constructor for sparsity discovery
	 */
	ProblemBase(UserCode& _usercode) : usercode(_usercode)
	{
		usercode.define_variables(*this);
	}

	/**
	 * Constructor for tape recording and deployment
	 */
	template<typename _Matrix, typename _Vector>
	ProblemBase(UserCode& _usercode, const ProblemInfo<_Matrix,_Vector>& info) : 
		usercode(_usercode),
		constraints(info.constraints), objective(info.objective), lagrangian(info.lagrangian), variable_bounds(info.variable_bounds)
	{
		usercode.define_variables(*this);
	}

	using constraint_t = VectorFunction<Matrix, Vector>;
	using objective_t = WeightedSum<Matrix, Vector>;
	using lagrangian_t = WeightedSum<Matrix, Vector>;
	using variablebounds_t = VectorFunction<Matrix, Vector>;

	constraint_t constraints;
	objective_t objective;
	lagrangian_t lagrangian;
	variablebounds_t variable_bounds;

	/**
	 * Compute the various elements of the problem by calling the 
	 * user-code. Stores the result in constraints/objective and lagrangian respecitvely
	 * 
	 * These calls are only made internally when the Matrix and Vector types manage
	 * their own memory (i.e., during construction). They should not be called
	 * by the user or during deployment.
	 */
	template<typename DType> void __eval_constraints_construction(DType dtype) 
	{
		constraints.initialize();
		constraints.set_zero();
		usercode.eval_constraints(dtype, constraints);
	}

	void __eval_variable_bounds_construction() 
	{
		// variable_bounds.initialize();
		variable_bounds.set_zero();
		usercode.eval_variable_bounds(variable_bounds);
	}

	template<typename DType> void __eval_objective_construction(DType dtype) 
	{
		objective.initialize();
		objective.set_zero(dtype);
		usercode.eval_objective(dtype, objective);
	}

	template<typename DType> void __eval_lagrangian_construction(DType dtype) 
	{
		lagrangian.initialize();
		lagrangian.set_zero(dtype);
		usercode.eval_objective(dtype, lagrangian);
		usercode.eval_constraints(dtype, lagrangian);
	}

	/**
	 * Computes the number of equalities based on how many of the upper and lower bounds are equal
	 */
	int num_equalities()
	{
	  Eigen::VectorX<scalar_t> _value(constraints.rows());
	  Eigen::VectorX<scalar_t> _lb(constraints.rows());
	  Eigen::VectorX<scalar_t> _ub(constraints.rows());
	  Eigen::VectorX<scalar_t> _var(num_variables());
	  eval_constraints(Eval(), _var, _value, _lb, _ub);

	  int n=0;
	  for(int i=0; i<_lb.rows(); i++)
	  	if(_lb[i] == _ub[i]) n++;
	  return n;
	}


	/**
	 * Compute the various elements of the problem by calling the 
	 * user-code. Stores the result in the given memory buffer.
	 * 
	 * Mem must be compatible with set_memory
	 */
  template<typename DType, typename... Args> 
  void eval_constraints(DType dtype, 
  										  Eigen::Ref<Eigen::VectorX<scalar_t>> var, 
  										  Args&... args)
  {
	  set_decision_variable(var);
  	constraints.set_memory(dtype, args...);
  	constraints.initialize();
  	usercode.eval_constraints(dtype, constraints);
  }

  template<typename... Args> 
  void eval_variable_bounds(Args&&... args)
  {
  	variable_bounds.set_memory(args...);
  	variable_bounds.set_zero();
  	// variable_bounds.initialize();
  	usercode.eval_variable_bounds(variable_bounds);
  }

  template<typename DType, typename... Args> 
  scalar_t eval_objective(DType dtype, 
  										Eigen::Ref<Eigen::VectorX<scalar_t>> var, 
  									  Args&... args)
  {
	  set_decision_variable(var);

	  // Weights for the objective are always 1
		Eigen::VectorX<scalar_t> weights(objective.rows());
		weights.array() = 1;

  	objective.set_memory(dtype, weights, args...);
  	objective.initialize();
    objective.set_zero(dtype);
  	usercode.eval_objective(dtype, objective);
  	return objective.value;
  }

  /**
   * Compute the lagrangian function.
   * 
   * Note that we actually compute the lagrangian of
   *   L(prim,dual) = obj(prim) + dual'*g(prima)
   * i.e., we ignore the variable bounds.
   * This is done at the moment because IPOpt only uses the hessian of the lagrangian, so it
   * doesn't matter, but this needs to be fixed.
   */
  template<typename DType, typename... Args> 
  scalar_t eval_lagrangian(DType dtype, 
  										     Eigen::Ref<Eigen::VectorX<scalar_t>> var, 
  										     scalar_t obj_factor, // Weight to multiply the objective by [normally 1]
  										     Eigen::Ref<Eigen::VectorX<scalar_t>> dual, // Dual variable
  									       Args&... args)
  {
  	assert(dual.rows() == constraints.rows() && "Dual vector is the wrong length");

		Eigen::VectorX<scalar_t> weights(lagrangian.rows());
		weights(seqN(0,objective.rows())).array() = obj_factor;
		weights.tail(constraints.rows()) = dual;

	  set_decision_variable(var);
  	lagrangian.set_memory(dtype, weights, args...);
  	lagrangian.initialize();
    lagrangian.set_zero(dtype);

  	usercode.eval_objective(dtype, lagrangian);
  	usercode.eval_constraints(dtype, lagrangian);

// TODO : Evaluate variable bounds in lagrangian function

  	return lagrangian.value;
  }


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

		constraints.extend_variables(n);
		objective.extend_variables(n);
		lagrangian.extend_variables(n);
		variable_bounds.extend_variables(n);
		variable_bounds.extend_rows(n); // The rows of the variable_bounds are the variables, so this one is backwards
	}

	int num_variables() { return m_num_variables; }

	/**
	 * Use the memory in var as the global decision variable
	 */	
	void set_decision_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		assert(var.rows() >= num_variables() && "Decision variable is the wrong size");		
		for(auto& call : variable_callbacks) call(var.data());
	}

	/**
	 * Generate data for this problem.
	 * 
	 * Calls generate on every matrix / vector of the problem.
	 */
	using Info = ProblemInfo<Matrix,Vector>;
	auto generate()
	{
		Info info;

		info.constraints = constraints.generate();
		info.objective = objective.generate();
		info.lagrangian = lagrangian.generate();
		info.variable_bounds = variable_bounds.generate();

		info.num_variables = num_variables();

		return info;
	}
};

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using Sparsity = ProblemBase<UserCode, scalar_t, BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using SparsityInfo = typename Sparsity<UserCode,scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using Tape = ProblemBase<UserCode, scalar_t, BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using TapeInfo = typename Tape<UserCode,scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using Problem = ProblemBase<UserCode, scalar_t, BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>;

/**
 * Generates sparsity information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
SparsityInfo<UserCode> generate_sparsity(UserCode& usercode)
{
	Sparsity<UserCode,scalar_t> prob(usercode);

  Eigen::VectorX<scalar_t> var(prob.num_variables());
  var.array() = 0;
  prob.set_decision_variable(var);

  prob.__eval_constraints_construction(Jacobian());
  prob.__eval_variable_bounds_construction();
  prob.__eval_objective_construction(Hessian());
  prob.__eval_lagrangian_construction(Hessian());

  return prob.generate();
}

/**
 * Generates tape information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
TapeInfo<UserCode> generate_tape(UserCode& usercode, SparsityInfo<UserCode> sparsity)
{
	Tape<UserCode,scalar_t> prob(usercode, sparsity);

  Eigen::VectorX<scalar_t> var(prob.num_variables());
  var.array() = 0;
  prob.set_decision_variable(var);

  prob.__eval_constraints_construction(Jacobian());
  prob.__eval_variable_bounds_construction();
  prob.__eval_objective_construction(Hessian());
  prob.__eval_lagrangian_construction(Hessian());

  return prob.generate();
}

/**
 * Generates tape and sparsity information for the given user code, and then creates a problem.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
Problem<UserCode> generate(UserCode& usercode)
{
	auto tape = generate_tape(usercode, generate_sparsity(usercode));
	return Problem<UserCode>(usercode, tape);
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

	// Pointer to the valuePtr of the jacobian 
	Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer; 

	template<typename Matrix, typename Vector>
	FunctionMemory(VectorFunction<Matrix,Vector>& f) :
		value(f.rows(),1), lb(f.rows(),1), ub(f.rows(),1), jacobian_buffer(NULL,0)
	{
		f.jacobian.allocate_memory(jacobian);
		new (&jacobian_buffer) Eigen::Map<Eigen::VectorX<scalar_t>>(jacobian.valuePtr(),jacobian.nonZeros());

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

	// Pointer to the valuePtr of the hessian 
	Eigen::Map<Eigen::VectorX<scalar_t>> hessian_buffer; 

	template<typename WSum>
	WeightedSumMemory(WSum& w) :
        gradient(w.num_variables),
        weights(w.rows()),
        hessian_buffer(NULL,0)
	{
		w.hessian.allocate_memory(hessian);
		new (&hessian_buffer) Eigen::Map<Eigen::VectorX<scalar_t>>(hessian.valuePtr(),hessian.nonZeros());

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
	FunctionMemory<scalar_t> variable_bounds;

  Eigen::VectorX<scalar_t> var;

	template<typename Problem>
	ProblemMemory(Problem& prob) :
			constraints(prob.constraints),
			variable_bounds(prob.variable_bounds),
      objective(prob.objective),
			lagrangian(prob.lagrangian),
			var(prob.num_variables())
	{
		// Zero everything
		prob.constraints.initialize();
		// prob.variable_bounds.initialize();
		prob.variable_bounds.set_zero();
		prob.objective.initialize();
		prob.lagrangian.initialize();
	  var.array() = 0;
	  prob.set_decision_variable(var);

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

	o << "Variable bnds: " << info.variable_bounds.rows << std::endl;

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
	o << "Variable bnds: " << info.variable_bounds.rows << std::endl;
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