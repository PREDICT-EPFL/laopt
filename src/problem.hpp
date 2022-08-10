#ifndef __PROBLEM_HPP
#define __PROBLEM_HPP

#include <numeric>
#include <iterator>
#include "variable.hpp"
#include "expr_base.hpp"
#include "lampc_function_tag.hpp"

namespace lampc
{

/**
 * Tags to dispatch the relevant problem function evaluations
 */
struct Eval {};
struct Jacobian {};
struct Gradient {};
struct Hessian {};

/**
 * We create three copies of the problem (and the contained constraints, objective, etc). 
 * 
 * The first is a Sparsity version, in which the variable information is also propagated with every function
 * call and passed to BSMatrixTape objects to develop the sparsity structures.
 *
 * The second is a Tape version, where the same process is followed to record the copy operations.
 * 
 * The third is the deployment version, which doesn't have the variable information and
 * just plays back the tape for speed. Only this version needs to be optimized.
 */

struct VariableBound {};

template<typename Derived, typename DerivedLb>
struct VariableLowerBound : VariableBound
{
    const IndexedVector<Derived> &variable;
    const DerivedLb& lb;
    explicit VariableLowerBound(const IndexedVector<Derived>& variable, const DerivedLb& lb) : variable(variable), lb(lb) {}
};

template<typename Derived, typename DerivedUb>
struct VariableUpperBound : VariableBound
{
    const IndexedVector<Derived>& variable;
    const DerivedUb& ub;
    explicit VariableUpperBound(const IndexedVector<Derived>& variable, const DerivedUb& ub) : variable(variable), ub(ub) {}
};

template<typename Derived, typename DerivedLb, typename DerivedUb>
struct VariableLowerUpperBound : VariableBound
{
    const IndexedVector<Derived>& variable;
    const DerivedLb& lb;
    const DerivedUb& ub;
    explicit VariableLowerUpperBound(const IndexedVector<Derived>& variable, const DerivedLb& lb, const DerivedUb& ub) : variable(variable), lb(lb), ub(ub) {}
};

template<typename Derived, typename DerivedLb>
VariableLowerBound<Derived, DerivedLb> operator<=(const DerivedLb& lb, const IndexedVector<Derived>& variable)
{
    return VariableLowerBound<Derived, DerivedLb>(variable, lb);
}

template<typename Derived, typename DerivedUb>
VariableUpperBound<Derived, DerivedUb> operator<=(const IndexedVector<Derived>& variable, const DerivedUb& ub)
{
    return VariableUpperBound<Derived, DerivedUb>(variable, ub);
}

template<typename Derived, typename DerivedUb>
VariableUpperBound<Derived, DerivedUb> operator>=(const DerivedUb& ub, const IndexedVector<Derived>& variable)
{
    return VariableUpperBound<Derived, DerivedUb>(variable, ub);
}

template<typename Derived, typename DerivedLb>
VariableLowerBound<Derived, DerivedLb> operator>=(const IndexedVector<Derived>& variable, const DerivedLb& lb)
{
    return VariableLowerBound<Derived, DerivedLb>(variable, lb);
}

template<typename Derived, typename DerivedEq>
VariableLowerUpperBound<Derived, DerivedEq, DerivedEq> operator==(const IndexedVector<Derived>& variable, const DerivedEq& eq)
{
    return VariableLowerUpperBound<Derived, DerivedEq, DerivedEq>(variable, eq, eq);
}

template<typename Derived, typename DerivedEq>
VariableLowerUpperBound<Derived, DerivedEq, DerivedEq> operator==(const DerivedEq& eq, const IndexedVector<Derived>& variable)
{
    return VariableLowerUpperBound<Derived, DerivedEq, DerivedEq>(variable, eq, eq);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
VariableLowerUpperBound<Derived, DerivedLb, DerivedUb> operator<=(const VariableLowerBound<Derived, DerivedLb>& vlb, const DerivedUb& ub)
{
    return VariableLowerUpperBound<Derived, DerivedLb, DerivedUb>(vlb.variable, vlb.lb, ub);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
VariableLowerUpperBound<Derived, DerivedLb, DerivedUb> operator<=(const DerivedLb& lb, const VariableUpperBound<Derived, DerivedUb>& vub)
{
    return VariableLowerUpperBound<Derived, DerivedLb, DerivedUb>(vub.variable, lb, vub.ub);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
VariableLowerUpperBound<Derived, DerivedLb, DerivedUb> operator>=(const DerivedUb& ub, const VariableLowerBound<Derived, DerivedLb>& vlb)
{
    return VariableLowerUpperBound<Derived, DerivedLb, DerivedUb>(vlb.variable, vlb.lb, ub);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
VariableLowerUpperBound<Derived, DerivedLb, DerivedUb> operator>=(const VariableUpperBound<Derived, DerivedUb>& vub, const DerivedLb& lb)
{
    return VariableLowerUpperBound<Derived, DerivedLb, DerivedUb>(vub.variable, lb, vub.ub);
}


struct ExprBound {};

template<typename Derived, typename DerivedLb>
struct ExprLowerBound : ExprBound
{
    const Derived& expr;
    const DerivedLb& lb;
    explicit ExprLowerBound(const ExprBase<Derived>& expr, const DerivedLb& lb) : expr(expr.derived()), lb(lb) {}
};

template<typename Derived, typename DerivedUb>
struct ExprUpperBound : ExprBound
{
    const Derived& expr;
    const DerivedUb& ub;
    explicit ExprUpperBound(const ExprBase<Derived>& expr, const DerivedUb& ub) : expr(expr.derived()), ub(ub) {}
};

template<typename Derived, typename DerivedLb, typename DerivedUb>
struct ExprLowerUpperBound : ExprBound
{
    const Derived& expr;
    const DerivedLb& lb;
    const DerivedUb& ub;
    explicit ExprLowerUpperBound(const ExprBase<Derived>& expr, const DerivedLb& lb, const DerivedUb& ub) : expr(expr.derived()), lb(lb), ub(ub) {}
};

template<typename Derived, typename DerivedUb>
ExprUpperBound<Derived, DerivedUb> operator<=(const ExprBase<Derived>& expr, const DerivedUb& ub)
{
    return ExprUpperBound<Derived, DerivedUb>(expr, ub);
}

template<typename Derived, typename DerivedUb>
ExprUpperBound<Derived, DerivedUb> operator>=(const DerivedUb& ub, const ExprBase<Derived>& expr)
{
    return ExprUpperBound<Derived, DerivedUb>(expr, ub);
}

template<typename Derived, typename DerivedLb>
ExprLowerBound<Derived, DerivedLb> operator<=(const DerivedLb& lb, const ExprBase<Derived>& expr)
{
    return ExprLowerBound<Derived, DerivedLb>(expr, lb);
}

template<typename Derived, typename DerivedLb>
ExprLowerBound<Derived, DerivedLb> operator>=(const ExprBase<Derived>& expr, const DerivedLb& lb)
{
    return ExprLowerBound<Derived, DerivedLb>(expr, lb);
}

template<typename Derived, typename DerivedEq>
ExprLowerUpperBound<Derived, DerivedEq, DerivedEq> operator==(const ExprBase<Derived>& expr, const DerivedEq& eq)
{
    return ExprLowerUpperBound<Derived, DerivedEq, DerivedEq>(expr, eq, eq);
}

template<typename Derived, typename DerivedEq>
ExprLowerUpperBound<Derived, DerivedEq, DerivedEq> operator==(const DerivedEq& eq, const ExprBase<Derived>& expr)
{
    return ExprLowerUpperBound<Derived, DerivedEq, DerivedEq>(expr, eq, eq);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
ExprLowerUpperBound<Derived, DerivedLb, DerivedUb> operator<=(const ExprLowerBound<Derived, DerivedLb>& flb, const DerivedUb& ub)
{
    return ExprLowerUpperBound<Derived, DerivedLb, DerivedUb>(flb.expr, flb.lb, ub);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
ExprLowerUpperBound<Derived, DerivedLb, DerivedUb> operator<=(const DerivedLb& lb, const ExprUpperBound<Derived, DerivedUb>& fub)
{
    return ExprLowerUpperBound<Derived, DerivedLb, DerivedUb>(fub.expr, lb, fub.ub);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
ExprLowerUpperBound<Derived, DerivedLb, DerivedUb> operator>=(const DerivedUb& ub, const ExprLowerBound<Derived, DerivedLb>& flb)
{
    return ExprLowerUpperBound<Derived, DerivedLb, DerivedUb>(flb.expr, flb.lb, ub);
}

template<typename Derived, typename DerivedLb, typename DerivedUb>
ExprLowerUpperBound<Derived, DerivedLb, DerivedUb> operator>=(const ExprUpperBound<Derived, DerivedUb>& fub, const DerivedLb& lb)
{
    return ExprLowerUpperBound<Derived, DerivedLb, DerivedUb>(fub.expr, lb, fub.ub);
}

template<typename Derived>
class NegExpr : public ExprBase<NegExpr<Derived>>
{
public:
    const Derived& expr;

    static constexpr int n_inputs = Derived::n_inputs;
    static constexpr int n_outputs = Derived::n_outputs;

    explicit NegExpr(const Derived& expr) : expr(expr) {}
};

template<typename DerivedLhs, typename DerivedRhs>
class AddExpr : public ExprBase<AddExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs& lhs;
    const DerivedRhs& rhs;

    static_assert(DerivedLhs::n_outputs == DerivedRhs::n_outputs, "Output dimension of expressions must be the same");
    static constexpr int n_inputs = DerivedLhs::n_inputs + DerivedRhs::n_inputs;
    static constexpr int n_outputs = DerivedLhs::n_outputs;

    explicit AddExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}
};

template<typename DerivedLhs, typename DerivedRhs>
AddExpr<DerivedLhs, DerivedRhs> operator+(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return AddExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

// we need this special case to be not ambiguous with Eigen
template<typename DerivedLhs, typename DerivedRhs>
AddExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>> operator+(const IndexedVector<DerivedLhs>& lhs, const IndexedVector<DerivedRhs>& rhs)
{
    return AddExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>>(lhs, rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
class SubExpr : public ExprBase<SubExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs& lhs;
    const DerivedRhs& rhs;

    static_assert(DerivedLhs::n_outputs == DerivedRhs::n_outputs, "Output dimension of expressions must be the same");
    static constexpr int n_inputs = DerivedLhs::n_inputs + DerivedRhs::n_inputs;
    static constexpr int n_outputs = DerivedLhs::n_outputs;

    explicit SubExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}
};

template<typename DerivedLhs, typename DerivedRhs>
SubExpr<DerivedLhs, DerivedRhs> operator-(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return SubExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

// we need this special case to be not ambiguous with Eigen
template<typename DerivedLhs, typename DerivedRhs>
SubExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>> operator-(const IndexedVector<DerivedLhs>& lhs, const IndexedVector<DerivedRhs>& rhs)
{
    return SubExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>>(lhs, rhs);
}

/**
 * Information about the function.
 * 
 * Either sparsity or tape information, depending on Matrix and Vector
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

    VectorFunction()
    {
        initialize();
    }

    inline int rows()
    {
        return m_rows;
    }

    /**
	 * Must be called before each evaluation
	 * 
	 * Resets the size of the function to zero, but keeps the number of variables.
	 */
	void initialize()
	{
		value.resize(0,1);
		jacobian.resize(0, num_variables);
		lb.resize(0,1);
		ub.resize(0,1);
	}

	/**
	 * Set all values to zero
	 */
	void set_zero(Eval)
	{
		value.set_zero();
		lb.set_zero();
		ub.set_zero();
	}

    void set_zero(Jacobian)
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
		set_memory(Jacobian{},
                   _value,
                   _lb,
                   _ub,
                   Eigen::Map<Eigen::VectorX<scalar_t>>(_jacobian.valuePtr(), _jacobian.nonZeros()));
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
		set_memory(lampc::Eval{}, mem.value, mem.lb, mem.ub);
	}
	void set_memory(Jacobian, FunctionMemory<scalar_t>& mem)
	{
		set_memory(lampc::Jacobian{}, mem.value, mem.lb, mem.ub, mem.jacobian_buffer);
	}
	void set_memory(FunctionMemory<scalar_t>& mem)
	{
		set_memory(mem.lb, mem.ub);
	}

	/**
	 * Constructor for tape recording or deployment
	 */
	template<typename TMatrix, typename TVector>
	explicit VectorFunction(const FunctionInfo<TMatrix, TVector>& info) :
		value(info.value), jacobian(info.jacobian), lb(info.lb), ub(info.ub)
	{
		initialize();
		set_zero(Jacobian{});

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

    Vector weights;
    scalar_t value;
	Vector gradient;
    Matrix hessian;

	int num_variables = 0;

    /**
     * Sparsity discovery constructor
     */
    WeightedSum()
    {
        initialize();
    }

    /**
     * Constructor for tape recording or deployment
     */
    template<typename TMatrix, typename TVector>
    explicit WeightedSum(const WeightedSumInfo<TMatrix, TVector>& info) :
            weights(info.weights), gradient(info.gradient), hessian(info.hessian)
    {
        initialize();
        set_zero(Hessian{});

        m_rows = info.rows;
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

    inline int rows()
    {
        return m_rows;
    }

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
	void set_memory(Eval, Eigen::Ref<Eigen::VectorX<scalar_t>> _weights)
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
		set_memory(Hessian{}, _weights, _gradient, Eigen::Map<Eigen::VectorX<scalar_t>>(_hessian.valuePtr(), _hessian.nonZeros()));
	}

	void set_memory(Eval, Eigen::Ref<Eigen::VectorX<scalar_t>> _weights, WeightedSumMemory<scalar_t>& mem)
	{
		set_memory(lampc::Eval{}, _weights);
	}
	
	void set_memory(Gradient, Eigen::Ref<Eigen::VectorX<scalar_t>> _weights, WeightedSumMemory<scalar_t>& mem)
	{
		set_memory(lampc::Eval{}, _weights, mem.gradient);
	}
	void set_memory(Hessian, Eigen::Ref<Eigen::VectorX<scalar_t>> _weights, WeightedSumMemory<scalar_t>& mem)
	{
		set_memory(lampc::Hessian{}, _weights, mem.gradient, mem.hessian_buffer);
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

template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ProblemVariableRegistrar;

template<typename DType, typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ObjectiveEvaluator;

template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class VariableBoundsEvaluator;

template<typename DType, typename UserCode, typename Scalar, typename Matrix, typename Vector>
class VectorConstraintsEvaluator;

template<typename DType, typename UserCode, typename Scalar, typename Matrix, typename Vector>
class WeightedSumConstraintsEvaluator;

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

template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ProblemBase
{
public:
	using scalar_t = Scalar;
    using objective_t = WeightedSum<Matrix, Vector>;
    using variable_bounds_t = VectorFunction<Matrix, Vector>;
    using constraint_t = VectorFunction<Matrix, Vector>;
    using lagrangian_t = WeightedSum<Matrix, Vector>;

    UserCode& user_code;

    objective_t objective;
    variable_bounds_t variable_bounds;
    constraint_t constraints;
    lagrangian_t lagrangian;

    // Callbacks used to register the global decision variable with each variable
    std::vector<std::function<void(scalar_t*)>> variable_callbacks;

    int m_num_variables = 0;

public:

	/**
	 * Default constructor for sparsity discovery
	 */
	explicit ProblemBase(UserCode& user_code) : user_code(user_code)
	{
        ProblemVariableRegistrar<UserCode, Scalar, Matrix, Vector> problem_variable_registrar(*this);
		user_code.define_problem(problem_variable_registrar);
	}

	/**
	 * Constructor for tape recording and deployment
	 */
	template<typename TMatrix, typename TVector>
	explicit ProblemBase(UserCode& user_code, const ProblemInfo<TMatrix, TVector>& info) :
		user_code(user_code),
        objective(info.objective), variable_bounds(info.variable_bounds), constraints(info.constraints), lagrangian(info.lagrangian)
	{
        ProblemVariableRegistrar<UserCode, Scalar, Matrix, Vector> problem_variable_registrar(*this);
        user_code.define_problem(problem_variable_registrar);
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
        eval_constraints(Eval{}, _var, _value, _lb, _ub);

        int n = 0;
        for (int i = 0; i < _lb.rows(); i++)
        {
            if (_lb[i] == _ub[i]) n++;
        }
        return n;
	}

    /**
	 * Compute the various elements of the problem by calling the 
	 * user-code. Stores the result in constraints/objective and lagrangian respectively
	 * 
	 * These calls are only made internally when the Matrix and Vector types manage
	 * their own memory (i.e., during construction). They should not be called
	 * by the user or during deployment.
	 */
    template<typename DType>
    void eval_objective_construction(DType dtype)
    {
        objective.initialize();
        objective.set_zero(dtype);

        ObjectiveEvaluator<DType, UserCode, Scalar, Matrix, Vector> objective_evaluator(*this, objective);
        user_code.define_problem(objective_evaluator);
    }

    void eval_variable_bounds_construction()
    {
        // We don't initialize the variable bounds because they are already set by the ProblemVariableRegistrar
        variable_bounds.lb(Eigen::seqN(0, num_variables())).array() = -std::numeric_limits<Scalar>::infinity();
        variable_bounds.ub(Eigen::seqN(0, num_variables())).array() = std::numeric_limits<Scalar>::infinity();

        VariableBoundsEvaluator<UserCode, Scalar, Matrix, Vector> variable_bounds_evaluator(*this, variable_bounds);
        user_code.define_problem(variable_bounds_evaluator);
    }
    
    template<typename DType>
    void eval_constraints_construction(DType dtype)
    {
        constraints.initialize();
        constraints.set_zero(dtype);

        VectorConstraintsEvaluator<DType, UserCode, Scalar, Matrix, Vector> constraints_evaluator(*this, constraints);
        user_code.define_problem(constraints_evaluator);
    }

    template<typename DType>
    void eval_lagrangian_construction(DType dtype)
    {
        lagrangian.initialize();
        lagrangian.set_zero(dtype);

        ObjectiveEvaluator<DType, UserCode, Scalar, Matrix, Vector> lagrangian_objective_evaluator(*this, lagrangian);
        user_code.define_problem(lagrangian_objective_evaluator);
        WeightedSumConstraintsEvaluator<DType, UserCode, Scalar, Matrix, Vector> lagrangian_constraints_evaluator(*this, lagrangian);
        user_code.define_problem(lagrangian_constraints_evaluator);
    }


	/**
	 * Compute the various elements of the problem by calling the 
	 * user-code. Stores the result in the given memory buffer.
	 * 
	 * Mem must be compatible with set_memory
	 */
    template<typename DType, typename... Args>
    scalar_t eval_objective(DType dtype,
                            Eigen::Ref<Eigen::VectorX<scalar_t>> var,
                            Args &... args)
    {
        // Weights for the objective are always 1
        // TODO: Don't allocated a new dynamic vector at every evaluation since it introduces a malloc and free each time. Maybe move to solver interface?
        Eigen::VectorX<scalar_t> weights(objective.rows());
        weights.array() = 1;

        set_decision_variable(var);
        objective.set_memory(dtype, weights, args...);
        objective.initialize();
        objective.set_zero(dtype);

        ObjectiveEvaluator<DType, UserCode, Scalar, Matrix, Vector> objective_evaluator(*this, objective);
        user_code.define_problem(objective_evaluator);

        return objective.value;
    }

    template<typename... Args>
    void eval_variable_bounds(Args &&... args)
    {
        variable_bounds.set_memory(args...);
        // We don't initialize the variable bounds because they are already set by the ProblemVariableRegistrar
        variable_bounds.lb(Eigen::seqN(0, num_variables())).array() = -std::numeric_limits<Scalar>::infinity();
        variable_bounds.ub(Eigen::seqN(0, num_variables())).array() = std::numeric_limits<Scalar>::infinity();

        VariableBoundsEvaluator<UserCode, Scalar, Matrix, Vector> variable_bounds_evaluator(*this, variable_bounds);
        user_code.define_problem(variable_bounds_evaluator);
    }

    template<typename DType, typename... Args>
    void eval_constraints(DType dtype,
                          Eigen::Ref<Eigen::VectorX<scalar_t>> var,
                          Args &... args)
    {
        set_decision_variable(var);
        constraints.set_memory(dtype, args...);
        constraints.initialize();
        constraints.set_zero(dtype);

        VectorConstraintsEvaluator<DType, UserCode, Scalar, Matrix, Vector> constraints_evaluator(*this, constraints);
        user_code.define_problem(constraints_evaluator);
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
                             Args &... args)
    {
        assert(dual.rows() == constraints.rows() && "Dual vector is the wrong length");

        // TODO: Don't allocated a new dynamic vector at every evaluation since it introduces a malloc and free each time. Maybe move to solver interface?
        Eigen::VectorX<scalar_t> weights(lagrangian.rows());
        weights(Eigen::seqN(0, objective.rows())).array() = obj_factor;
        weights.tail(constraints.rows()) = dual;

        set_decision_variable(var);
        lagrangian.set_memory(dtype, weights, args...);
        lagrangian.initialize();
        lagrangian.set_zero(dtype);

        // TODO : Evaluate variable bounds in lagrangian function
        ObjectiveEvaluator<DType, UserCode, Scalar, Matrix, Vector> lagrangian_objective_evaluator(*this, lagrangian);
        user_code.define_problem(lagrangian_objective_evaluator);
        WeightedSumConstraintsEvaluator<DType, UserCode, Scalar, Matrix, Vector> lagrangian_constraints_evaluator(*this, lagrangian);
        user_code.define_problem(lagrangian_constraints_evaluator);

        return lagrangian.value;
    }

	int num_variables()
    {
        return m_num_variables;
    }

	/**
	 * Use the memory in var as the global decision variable
	 */	
	void set_decision_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		assert(var.rows() == num_variables() && "Decision variable is the wrong size");
		for(auto& call : variable_callbacks)
        {
            call(var.data());
        }
	}

	/**
	 * Generate data for this problem.
	 * 
	 * Calls generate on every matrix / vector of the problem.
	 */
	using Info = ProblemInfo<Matrix,Vector>;
    Info generate()
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

template<typename Derived>
struct ExprEvaluator;

template<typename Derived>
struct ExprEvaluator<IndexedVector<Derived>>
{
    static EIGEN_STRONG_INLINE auto
    function(const IndexedVector<Derived>& indexed_vector)
    {
        lib::IDENTITY id;
        return id.function(indexed_vector.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE void
    jacobian(const IndexedVector<Derived>& indexed_vector, const OutIndices& out_indices, VectorFunction<Matrix, Vector>& out)
    {
        lib::IDENTITY id;
        id.jacobian(out.value(out_indices),
                    out.jacobian(out_indices, indexed_vector.indices()),
                    indexed_vector.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    wsum(const IndexedVector<Derived>& indexed_vector, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::IDENTITY id;
        return id.wsum(out.weights(out_indices),
                       indexed_vector.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    gradient(const IndexedVector<Derived>& indexed_vector, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::IDENTITY id;
        return id.gradient(out.gradient(indexed_vector.indices()),
                           out.weights(out_indices),
                           indexed_vector.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    hessian(const IndexedVector<Derived>& indexed_vector, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::IDENTITY id;
        return id.hessian(out.gradient(indexed_vector.indices()),
                          out.hessian(indexed_vector.indices(), indexed_vector.indices()),
                          out.weights(out_indices),
                          indexed_vector.cast_base());
    }
};

template<typename Function, typename Tag, typename Info, typename Capture>
struct ExprEvaluator<FunctionCapture<Function, Tag, Info, Capture>>
{
    static EIGEN_STRONG_INLINE auto
    function(const FunctionCapture<Function, Tag, Info, Capture>& function_capture)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.function(Tag{},
                                                  vars.cast_base()...);
        });
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE void
    jacobian(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, const OutIndices& out_indices, VectorFunction<Matrix, Vector>& out)
    {
        function_capture.capture([&](auto&&... vars) {
            auto in_indices = concatenate_indices(vars.indices()...);
            function_capture.func.jacobian(Tag{},
                                           out.value(out_indices),
                                           out.jacobian(out_indices, in_indices),
                                           vars.cast_base()...);
        });
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    wsum(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.wsum(Tag{},
                                              out.weights(out_indices),
                                              vars.cast_base()...);
        });
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    gradient(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        return function_capture.capture([&](auto&&... vars) {
            auto in_indices = concatenate_indices(vars.indices()...);
            return function_capture.func.gradient(Tag{},
                                                  out.gradient(in_indices),
                                                  out.weights(out_indices),
                                                  vars.cast_base()...);
        });
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    hessian(const FunctionCapture<Function, Tag, Info, Capture> function_capture, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        return function_capture.capture([&](auto&&... vars) {
            auto in_indices = concatenate_indices(vars.indices()...);
            return function_capture.func.hessian(Tag{},
                                                 out.gradient(in_indices),
                                                 out.hessian(in_indices, in_indices),
                                                 out.weights(out_indices),
                                                 vars.cast_base()...);
        });
    }
};

template<typename Derived>
struct ExprEvaluator<NegExpr<IndexedVector<Derived>>>
{
    static EIGEN_STRONG_INLINE auto
    function(const NegExpr<IndexedVector<Derived>>& expr)
    {
        lib::IDENTITY id(-1);
        return id.function(expr.expr.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE void
    jacobian(const NegExpr<IndexedVector<Derived>>& expr, const OutIndices& out_indices, VectorFunction<Matrix, Vector>& out)
    {
        lib::IDENTITY id(-1);
        id.jacobian(out.value(out_indices),
                    out.jacobian(out_indices, expr.expr.indices()),
                    expr.expr.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    wsum(const NegExpr<IndexedVector<Derived>>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::IDENTITY id(-1);
        return id.wsum(out.weights(out_indices),
                       expr.expr.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    gradient(const NegExpr<IndexedVector<Derived>>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::IDENTITY id(-1);
        return id.gradient(out.gradient(expr.expr.indices()),
                           out.weights(out_indices),
                           expr.expr.cast_base());
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    hessian(const NegExpr<IndexedVector<Derived>>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::IDENTITY id(-1);
        return id.hessian(out.gradient(expr.expr.indices()),
                          out.hessian(expr.expr.indices(), expr.expr.indices()),
                          out.weights(out_indices),
                          expr.expr.cast_base());
    }
};

template<typename Function, typename Tag, typename Info, typename Capture>
struct ExprEvaluator<NegExpr<FunctionCapture<Function, Tag, Info, Capture>>>
{
    static EIGEN_STRONG_INLINE auto
    function(const NegExpr<FunctionCapture<Function, Tag, Info, Capture>>& expr)
    {
        lib::NEG<Function, Tag> neg(expr.expr.func);
        auto function_capture = expr.expr.capture([&](auto&&... vars) { return neg(vars...); });
        return ExprEvaluator<decltype(function_capture)>::function(function_capture);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE void
    jacobian(const NegExpr<FunctionCapture<Function, Tag, Info, Capture>>& expr, const OutIndices& out_indices, VectorFunction<Matrix, Vector>& out)
    {
        lib::NEG<Function, Tag> neg(expr.expr.func);
        auto function_capture = expr.expr.capture([&](auto&&... vars) { return neg(vars...); });
        ExprEvaluator<decltype(function_capture)>::jacobian(function_capture, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    wsum(const NegExpr<FunctionCapture<Function, Tag, Info, Capture>>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::NEG<Function, Tag> neg(expr.expr.func);
        auto function_capture = expr.expr.capture([&](auto&&... vars) { return neg(vars...); });
        return ExprEvaluator<decltype(function_capture)>::wsum(function_capture, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    gradient(const NegExpr<FunctionCapture<Function, Tag, Info, Capture>>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::NEG<Function, Tag> neg(expr.expr.func);
        auto function_capture = expr.expr.capture([&](auto&&... vars) { return neg(vars...); });
        return ExprEvaluator<decltype(function_capture)>::gradient(function_capture, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    hessian(const NegExpr<FunctionCapture<Function, Tag, Info, Capture>>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        lib::NEG<Function, Tag> neg(expr.expr.func);
        auto function_capture = expr.expr.capture([&](auto&&... vars) { return neg(vars...); });
        return ExprEvaluator<decltype(function_capture)>::hessian(function_capture, out_indices, out);
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<AddExpr<DerivedLhs, DerivedRhs>>
{
    static EIGEN_STRONG_INLINE auto
    function(const AddExpr<DerivedLhs, DerivedRhs>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) + ExprEvaluator<DerivedRhs>::function(expr.rhs);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE void
    jacobian(const AddExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, VectorFunction<Matrix, Vector>& out)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_indices, out);
        ExprEvaluator<DerivedRhs>::jacobian(expr.rhs, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    wsum(const AddExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, out_indices, out) + ExprEvaluator<DerivedRhs>::wsum(expr.rhs, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    gradient(const AddExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        return ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_indices, out) + ExprEvaluator<DerivedRhs>::gradient(expr.rhs, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    hessian(const AddExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        return ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_indices, out) + ExprEvaluator<DerivedRhs>::hessian(expr.rhs, out_indices, out);
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>
{
    static EIGEN_STRONG_INLINE auto
    function(const SubExpr<DerivedLhs, DerivedRhs>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        NegExpr<DerivedRhs> neg_expr(expr.rhs);
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) + ExprEvaluator<NegExpr<DerivedRhs>>::function(neg_expr);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, VectorFunction<Matrix, Vector>& out)
    {
        NegExpr<DerivedRhs> neg_expr(expr.rhs);
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_indices, out);
        ExprEvaluator<NegExpr<DerivedRhs>>::jacobian(neg_expr, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    wsum(const SubExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        NegExpr<DerivedRhs> neg_expr(expr.rhs);
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, out_indices, out) + ExprEvaluator<NegExpr<DerivedRhs>>::wsum(neg_expr, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    gradient(const SubExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        NegExpr<DerivedRhs> neg_expr(expr.rhs);
        return ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_indices, out) + ExprEvaluator<NegExpr<DerivedRhs>>::gradient(neg_expr, out_indices, out);
    }

    template<typename OutIndices, typename Matrix, typename Vector>
    static EIGEN_STRONG_INLINE auto
    hessian(const SubExpr<DerivedLhs, DerivedRhs>& expr, const OutIndices& out_indices, WeightedSum<Matrix, Vector>& out)
    {
        NegExpr<DerivedRhs> neg_expr(expr.rhs);
        return ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_indices, out) + ExprEvaluator<NegExpr<DerivedRhs>>::hessian(neg_expr, out_indices, out);
    }
};

template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ProblemVariableRegistrar
{
    ProblemBase<UserCode, Scalar, Matrix, Vector>& problem;

public:
    explicit ProblemVariableRegistrar(ProblemBase<UserCode, Scalar, Matrix, Vector>& problem) : problem(problem) {}

    template<int n>
    EIGEN_STRONG_INLINE void add_variable(Variable<Scalar, n>& var)
    {
        problem.variable_callbacks.push_back(var.register_variable(problem.m_num_variables));
        problem.m_num_variables += n;

        problem.variable_bounds.extend_variables(n);
        problem.variable_bounds.extend_rows(n); // already extend variable_bounds for bounds

        problem.objective.extend_variables(n);
        problem.constraints.extend_variables(n);
        problem.lagrangian.extend_variables(n);
    }

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr(Args...) {}
};

template<typename DType, typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ObjectiveEvaluator
{
    ProblemBase<UserCode, Scalar, Matrix, Vector>& problem;
    WeightedSum<Matrix, Vector>& objective;

public:
    explicit ObjectiveEvaluator(ProblemBase<UserCode, Scalar, Matrix, Vector>& problem, WeightedSum<Matrix, Vector>& objective) :
        problem(problem), objective(objective) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_variable(Args...) {}

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value>::type
    add_obj(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(objective.weights.rows(), Eigen::fix<n_outputs>);
        objective.extend_rows(n_outputs);

        objective.value += ExprEvaluator<Derived>::wsum(expr.derived(), out_indices, objective);
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Gradient>::value>::type
    add_obj(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(objective.weights.rows(), Eigen::fix<n_outputs>);
        objective.extend_rows(n_outputs);

        objective.value += ExprEvaluator<Derived>::gradient(expr.derived(), out_indices, objective);
    }

    template<typename Derived, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Hessian>::value>::type
    add_obj(const ExprBase<Derived>& expr)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(objective.weights.rows(), Eigen::fix<n_outputs>);
        objective.extend_rows(n_outputs);

        objective.value += ExprEvaluator<Derived>::hessian(expr.derived(), out_indices, objective);
    }

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr(Args...) {}
};

template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class VariableBoundsEvaluator
{
    ProblemBase<UserCode, Scalar, Matrix, Vector>& problem;
    VectorFunction<Matrix, Vector>& variable_bounds;

public:
    explicit VariableBoundsEvaluator(ProblemBase<UserCode, Scalar, Matrix, Vector>& problem, VectorFunction<Matrix, Vector>& variable_bounds) :
        problem(problem), variable_bounds(variable_bounds) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_variable(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj(Args...) {}

    template<typename Derived, typename DerivedBound>
    EIGEN_STRONG_INLINE void add_constr(const VariableLowerBound<Derived, Eigen::MatrixBase<DerivedBound>>& bound)
    {
        variable_bounds.lb(bound.variable.indices()) = bound.lb;
    }

    template<typename Derived, typename DerivedScalar>
    EIGEN_STRONG_INLINE void add_constr(const VariableLowerBound<Derived, DerivedScalar>& bound)
    {
        variable_bounds.lb(bound.variable.indices()).array() = bound.lb;
    }

    template<typename Derived, typename DerivedBound>
    EIGEN_STRONG_INLINE void add_constr(const VariableUpperBound<Derived, Eigen::MatrixBase<DerivedBound>>& bound)
    {
        variable_bounds.ub(bound.variable.indices()) = bound.ub;
    }

    template<typename Derived, typename DerivedScalar>
    EIGEN_STRONG_INLINE void add_constr(const VariableUpperBound<Derived, DerivedScalar>& bound)
    {
        variable_bounds.ub(bound.variable.indices()).array() = bound.ub;
    }

    template<typename Derived, typename DerivedLb, typename DerivedUb>
    EIGEN_STRONG_INLINE void add_constr(const VariableLowerUpperBound<Derived, DerivedLb, DerivedUb>& bound)
    {
        add_constr(VariableLowerBound<Derived, DerivedLb>(bound.variable, bound.lb));
        add_constr(VariableUpperBound<Derived, DerivedUb>(bound.variable, bound.ub));
    }

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr(Args...) {}
};

template<typename DType, typename UserCode, typename Scalar, typename Matrix, typename Vector>
class VectorConstraintsEvaluator
{
    ProblemBase<UserCode, Scalar, Matrix, Vector>& problem;
    VectorFunction<Matrix, Vector>& constraints;

public:
    explicit VectorConstraintsEvaluator(ProblemBase<UserCode, Scalar, Matrix, Vector>& problem, VectorFunction<Matrix, Vector>& constraints) :
        problem(problem), constraints(constraints) {}

private:
    template<typename Derived, typename DerivedLb, typename OutIndices>
    EIGEN_STRONG_INLINE void
    add_bounds(const ExprLowerBound<Derived, Eigen::MatrixBase<DerivedLb>>& bound, const OutIndices& out_indices)
    {
        constraints.lb(out_indices) = bound.lb;
    }

    template<typename Derived, typename DerivedScalar, typename OutIndices>
    EIGEN_STRONG_INLINE void
    add_bounds(const ExprLowerBound<Derived, DerivedScalar>& bound, const OutIndices& out_indices)
    {
        constraints.lb(out_indices).array() = bound.lb;
    }

    template<typename Derived, typename DerivedLb, typename OutIndices>
    EIGEN_STRONG_INLINE void
    add_bounds(const ExprUpperBound<Derived, Eigen::MatrixBase<DerivedLb>>& bound, const OutIndices& out_indices)
    {
        constraints.ub(out_indices) = bound.ub;
    }

    template<typename Derived, typename DerivedScalar, typename OutIndices>
    EIGEN_STRONG_INLINE void
    add_bounds(const ExprUpperBound<Derived, DerivedScalar>& bound, const OutIndices& out_indices)
    {
        constraints.ub(out_indices).array() = bound.ub;
    }

    template<typename Derived, typename DerivedLb, typename DerivedUb, typename OutIndices>
    EIGEN_STRONG_INLINE void
    add_bounds(const ExprLowerUpperBound<Derived, DerivedLb, DerivedUb>& bound, const OutIndices& out_indices)
    {
        add_bounds(ExprLowerBound<Derived, DerivedLb>(bound.expr, bound.lb), out_indices);
        add_bounds(ExprUpperBound<Derived, DerivedLb>(bound.expr, bound.ub), out_indices);
    }

public:
    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_variable(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj(Args...) {}

    template<template<typename...> class Bound, typename Derived, typename ...BoundArgs, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value && std::is_base_of<ExprBound, Bound<Derived, BoundArgs...>>::value>::type
    add_constr(const Bound<Derived, BoundArgs...>& bound)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(constraints.value.rows(), Eigen::fix<n_outputs>);
        constraints.extend_rows(n_outputs);

        constraints.value(out_indices) = ExprEvaluator<Derived>::function(bound.expr.derived());
        add_bounds(bound, out_indices);
    }

    template<template<typename...> class Bound, typename Derived, typename ...BoundArgs, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Jacobian>::value && std::is_base_of<ExprBound, Bound<Derived, BoundArgs...>>::value>::type
    add_constr(const Bound<Derived, BoundArgs...>& bound)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(constraints.value.rows(), Eigen::fix<n_outputs>);
        constraints.extend_rows(n_outputs);

        ExprEvaluator<Derived>::jacobian(bound.expr.derived(), out_indices, constraints);
        add_bounds(bound, out_indices);
    }

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr(Args...) {}
};

template<typename DType, typename UserCode, typename Scalar, typename Matrix, typename Vector>
class WeightedSumConstraintsEvaluator
{
    ProblemBase<UserCode, Scalar, Matrix, Vector>& problem;
    WeightedSum<Matrix, Vector>& constraints;

public:
    explicit WeightedSumConstraintsEvaluator(ProblemBase<UserCode, Scalar, Matrix, Vector>& problem, WeightedSum<Matrix, Vector>& constraints) :
        problem(problem), constraints(constraints) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_variable(Args...) {}

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_obj(Args...) {}

    template<template<typename...> class Bound, typename Derived, typename ...BoundArgs, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Eval>::value && std::is_base_of<ExprBound, Bound<Derived, BoundArgs...>>::value>::type
    add_constr(const Bound<Derived, BoundArgs...>& bound)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(constraints.weights.rows(), Eigen::fix<n_outputs>);
        constraints.extend_rows(n_outputs);

        constraints.value += ExprEvaluator<Derived>::wsum(bound.expr.derived(), out_indices, constraints);
    }

    template<template<typename...> class Bound, typename Derived, typename ...BoundArgs, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Gradient>::value && std::is_base_of<ExprBound, Bound<Derived, BoundArgs...>>::value>::type
    add_constr(const Bound<Derived, BoundArgs...>& bound)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(constraints.weights.rows(), Eigen::fix<n_outputs>);
        constraints.extend_rows(n_outputs);

        constraints.value += ExprEvaluator<Derived>::gradient(bound.expr.derived(), out_indices, constraints);
    }

    template<template<typename...> class Bound, typename Derived, typename ...BoundArgs, typename LocalDType = DType>
    EIGEN_STRONG_INLINE typename std::enable_if<std::is_same<LocalDType, Hessian>::value && std::is_base_of<ExprBound, Bound<Derived, BoundArgs...>>::value>::type
    add_constr(const Bound<Derived, BoundArgs...>& bound)
    {
        static constexpr int n_outputs = Derived::n_outputs;

        auto out_indices = Eigen::seqN(constraints.weights.rows(), Eigen::fix<n_outputs>);
        constraints.extend_rows(n_outputs);

        constraints.value += ExprEvaluator<Derived>::hessian(bound.expr.derived(), out_indices, constraints);
    }

    template<typename ...Args>
    EIGEN_STRONG_INLINE void add_constr(Args...) {}
};

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using Sparsity = ProblemBase<UserCode, scalar_t, BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using SparsityInfo = typename Sparsity<UserCode, scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using Tape = ProblemBase<UserCode, scalar_t, BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using TapeInfo = typename Tape<UserCode, scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using Problem = ProblemBase<UserCode, scalar_t, BSMatrix<scalar_t>, BSMatrixDenseDeployment<scalar_t>>;

/**
 * Generates sparsity information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
SparsityInfo<UserCode> generate_sparsity(UserCode& user_code)
{
	Sparsity<UserCode, scalar_t> prob(user_code);

    Eigen::VectorX<scalar_t> var(prob.num_variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_construction(Jacobian{});
    prob.eval_variable_bounds_construction();
    prob.eval_objective_construction(Hessian{});
    prob.eval_lagrangian_construction(Hessian{});

    return prob.generate();
}

/**
 * Generates tape information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
TapeInfo<UserCode> generate_tape(UserCode& user_code, SparsityInfo<UserCode> sparsity)
{
	Tape<UserCode, scalar_t> prob(user_code, sparsity);

    Eigen::VectorX<scalar_t> var(prob.num_variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_construction(Jacobian{});
    prob.eval_variable_bounds_construction();
    prob.eval_objective_construction(Hessian{});
    prob.eval_lagrangian_construction(Hessian{});

    return prob.generate();
}

/**
 * Generates tape and sparsity information for the given user code, and then creates a problem.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
Problem<UserCode> generate(UserCode& user_code)
{
    TapeInfo<UserCode> tape = generate_tape(user_code, generate_sparsity(user_code));
    return Problem<UserCode>(user_code, tape);
}

/**
 * A helper class that can be used to create all the required memory for 
 * a VectorFunction if the memory isn't held externally.
 */
template<typename scalar_t>
struct FunctionMemory {
    Eigen::SparseMatrix<scalar_t> jacobian;
    Eigen::VectorX<scalar_t> value;
    Eigen::VectorX<scalar_t> lb;
    Eigen::VectorX<scalar_t> ub;

    // Pointer to the valuePtr of the jacobian
    Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer;

    template<typename Matrix, typename Vector>
    explicit FunctionMemory(VectorFunction<Matrix, Vector> &f) :
        value(f.rows(), 1),
        lb(f.rows(), 1),
        ub(f.rows(), 1),
        jacobian_buffer(NULL, 0)
    {
        f.jacobian.allocate_memory(jacobian);
        new (&jacobian_buffer) Eigen::Map<Eigen::VectorX<scalar_t>>(jacobian.valuePtr(), jacobian.nonZeros());

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
	explicit WeightedSumMemory(WSum& w) :
        gradient(w.num_variables),
        weights(w.rows()),
        hessian_buffer(NULL, 0)
	{
		w.hessian.allocate_memory(hessian);
		new (&hessian_buffer) Eigen::Map<Eigen::VectorX<scalar_t>>(hessian.valuePtr(), hessian.nonZeros());

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
struct ProblemMemory {
    FunctionMemory<scalar_t> constraints;
    FunctionMemory<scalar_t> variable_bounds;
    WeightedSumMemory<scalar_t> objective;
    WeightedSumMemory<scalar_t> lagrangian;

    Eigen::VectorX<scalar_t> var;

    template<typename Problem>
    explicit ProblemMemory(Problem &prob) :
        constraints(prob.constraints),
        variable_bounds(prob.variable_bounds),
        objective(prob.objective),
        lagrangian(prob.lagrangian),
        var(prob.num_variables())
    {
        // Zero everything
        prob.constraints.initialize();
        // prob.variable_bounds.initialize();
        // prob.variable_bounds.set_zero();
        prob.objective.initialize();
        prob.lagrangian.initialize();
        var.array() = 0;
        prob.set_decision_variable(var);

        for (int i = 0; i < objective.hessian.nonZeros(); i++) objective.hessian.valuePtr()[i] = i;
        for (int i = 0; i < constraints.jacobian.nonZeros(); i++) constraints.jacobian.valuePtr()[i] = i;
        for (int i = 0; i < lagrangian.hessian.nonZeros(); i++) lagrangian.hessian.valuePtr()[i] = i;
    }
};

template <typename scalar_t>
std::ostream& operator<<(std::ostream &o, const ProblemInfo<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>> &info)
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

}

#endif // __PROBLEM_HPP
