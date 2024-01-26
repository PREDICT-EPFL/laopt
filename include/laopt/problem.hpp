#ifndef LAOPT_PROBLEM_HPP
#define LAOPT_PROBLEM_HPP

#include <numeric>
#include <iterator>
#include "problem_dispatch_types.hpp"
#include "problem_vector_function.hpp"
#include "problem_weighted_sum_function.hpp"
#include "opt_problem.hpp"
#include "problem_evaluators/problem_size_evaluator.hpp"
#include "problem_evaluators/objective_evaluator.hpp"
#include "problem_evaluators/decision_variable_setter.hpp"
#include "problem_evaluators/variable_bounds_evaluator.hpp"
#include "problem_evaluators/vector_constraints_evaluator.hpp"
#include "problem_evaluators/weighted_sum_constraints_evaluator.hpp"
#include "indexed_vector.hpp"
#include "expressions.hpp"

namespace laopt
{

template<typename Matrix, typename Vector>
struct ProblemInfo
{
    WeightedSumFunctionInfo<Matrix,Vector> objective;
    VectorFunctionInfo<Matrix,Vector>      variable_bounds;
    VectorFunctionInfo<Matrix,Vector>      constraints;
	WeightedSumFunctionInfo<Matrix,Vector> lagrangian;
};

template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ProblemBase;

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


template<typename UserCode, typename Scalar, typename Matrix, typename Vector>
class ProblemBase
{
private:
    UserCode& user_code;

public:
    using scalar_t = Scalar;

    WeightedSumFunction<Matrix, Vector> objective;
    VectorFunction<Matrix, Vector>      variable_bounds;
    VectorFunction<Matrix, Vector>      constraints;
    WeightedSumFunction<Matrix, Vector> lagrangian;

	/**
	 * Default constructor for sparsity discovery
	 */
	explicit ProblemBase(UserCode& user_code) : user_code(user_code)
	{
        ProblemSizeEvaluator<Matrix, Vector> problem_size_evaluator(variable_bounds, objective, constraints, lagrangian);
		user_code.define_problem(problem_size_evaluator);
	}

	/**
	 * Constructor for tape recording and deployment
	 */
	template<typename TMatrix, typename TVector>
	explicit ProblemBase(UserCode& user_code, const ProblemInfo<TMatrix, TVector>& info) :
		user_code(user_code), objective(info.objective), variable_bounds(info.variable_bounds),
        constraints(info.constraints), lagrangian(info.lagrangian) {}

    int variables()
    {
        return objective.variables();
    }

    /**
     * Use the memory in var as the global decision variable
     */
    void set_decision_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
    {
        assert(var.rows() == variables() && "Decision variable is the wrong size");

        DecisionVariableSetter<Scalar> decision_variable_setter(var);
        user_code.define_problem(decision_variable_setter);
    }

    scalar_t eval_objective()
    {
        objective.value = 0;

        eval_objective_no_memory(Eval{});
        return objective.value;
    }

    void eval_objective_gradient(Eigen::Ref<Eigen::VectorX<scalar_t>> gradient)
    {
        objective.gradient.set_buffer(gradient);

        objective.gradient.set_zero();

        eval_objective_no_memory(Gradient{});
    }

    void eval_objective_hessian(Eigen::Ref<Eigen::VectorX<scalar_t>> hessian_buffer)
    {
        objective.hessian.set_target(hessian_buffer);

        objective.hessian.set_zero();

        eval_objective_no_memory(Hessian{});
    }

    void eval_variable_bounds(Eigen::Ref<Eigen::VectorX<scalar_t>> lb,
                              Eigen::Ref<Eigen::VectorX<scalar_t>> ub)
    {
        variable_bounds.lb.set_buffer(lb);
        variable_bounds.ub.set_buffer(ub);

        variable_bounds.lb(Eigen::seqN(0, variables())).array() = -std::numeric_limits<Scalar>::infinity();
        variable_bounds.ub(Eigen::seqN(0, variables())).array() = std::numeric_limits<Scalar>::infinity();

        eval_variable_bounds_no_memory();
    }

    void eval_constraints(Eigen::Ref<Eigen::VectorX<scalar_t>> cons,
                          Eigen::Ref<Eigen::VectorX<scalar_t>> lb,
                          Eigen::Ref<Eigen::VectorX<scalar_t>> ub)
    {
        constraints.value.set_buffer(cons);
        constraints.lb.set_buffer(lb);
        constraints.ub.set_buffer(ub);

        constraints.value.set_zero();
        constraints.lb.set_zero();
        constraints.ub.set_zero();

        eval_constraints_no_memory(Eval{});
    }

    void eval_constraints_jacobian(Eigen::Ref<Eigen::VectorX<scalar_t>> jacobian)
    {
        constraints.jacobian.set_target(jacobian);

        constraints.jacobian.set_zero();

        eval_constraints_no_memory(Jacobian{});
    }

    /**
     * Compute the hessian of the lagrangian:
     * ∇^2 L(prim, dual) = obj_factor * ∇^2 obj(prim) + sum_i dual_i * ∇^2 g_i(prim)
     */
    void eval_lagrangian_hessian(const scalar_t obj_factor,
                                 Eigen::Ref<Eigen::VectorX<scalar_t>> dual,
                                 Eigen::Ref<Eigen::VectorX<scalar_t>> hessian_buffer)
    {
        assert(dual.rows() == constraints.rows() &&
               dual.rows() == lagrangian.rows() && "Dual vector is the wrong length!");

        lagrangian.weights.set_buffer(dual);
        lagrangian.hessian.set_target(hessian_buffer);

        lagrangian.hessian.set_zero();

        eval_lagrangian_no_memory(Hessian{}, obj_factor);
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

        return info;
    }

    template<typename UserCode_, typename scalar_t_>
    friend SparsityInfo<UserCode_> generate_sparsity(UserCode_& user_code);

    template<typename UserCode_, typename scalar_t_>
    friend TapeInfo<UserCode_> generate_tape(UserCode_& user_code, SparsityInfo<UserCode_> sparsity);

private:
    /**
	 * Compute the various elements of the problem by calling the
	 * user-code. Stores the result in constraints/objective and lagrangian respectively
	 *
	 * These calls are only made internally when the Matrix and Vector types manage
	 * their own memory (i.e., during construction). They should not be called
	 * by the user or during deployment.
	 */
    template<typename DType>
    void eval_objective_no_memory(DType)
    {
        ObjectiveEvaluator<DType, Matrix, Vector> objective_evaluator(objective);
        user_code.define_problem(objective_evaluator);
    }

    void eval_variable_bounds_no_memory()
    {
        VariableBoundsEvaluator<Matrix, Vector> variable_bounds_evaluator(variable_bounds);
        user_code.define_problem(variable_bounds_evaluator);
    }

    template<typename DType>
    void eval_constraints_no_memory(DType)
    {
        VectorConstraintsEvaluator<DType, Matrix, Vector> constraints_evaluator(constraints);
        user_code.define_problem(constraints_evaluator);
    }

    template<typename DType>
    void eval_lagrangian_no_memory(DType, const scalar_t obj_factor = static_cast<scalar_t>(1))
    {
        ObjectiveEvaluator<DType, Matrix, Vector> lagrangian_objective_evaluator(lagrangian, obj_factor);
        user_code.define_problem(lagrangian_objective_evaluator);
        WeightedSumConstraintsEvaluator<DType, Matrix, Vector> lagrangian_constraints_evaluator(lagrangian);
        user_code.define_problem(lagrangian_constraints_evaluator);
    }
};

/**
 * Generates sparsity information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
SparsityInfo<UserCode> generate_sparsity(UserCode& user_code)
{
	Sparsity<UserCode, scalar_t> prob(user_code);

    Eigen::VectorX<scalar_t> var(prob.variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_no_memory(Jacobian{});
    prob.eval_variable_bounds_no_memory();
    prob.eval_objective_no_memory(Hessian{});
    prob.eval_lagrangian_no_memory(Hessian{});

    return prob.generate();
}

/**
 * Generates tape information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
TapeInfo<UserCode> generate_tape(UserCode& user_code, SparsityInfo<UserCode> sparsity)
{
	Tape<UserCode, scalar_t> prob(user_code, sparsity);

    Eigen::VectorX<scalar_t> var(prob.variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_no_memory(Jacobian{});
    prob.eval_variable_bounds_no_memory();
    prob.eval_objective_no_memory(Hessian{});
    prob.eval_lagrangian_no_memory(Hessian{});

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
 * a Problem if the memory isn't held externally.
 */
template<typename scalar_t>
struct ProblemMemory {
    WeightedSumFunctionMemory<scalar_t> objective;
    VectorFunctionMemory<scalar_t>      variable_bounds;
    VectorFunctionMemory<scalar_t>      constraints;
    WeightedSumFunctionMemory<scalar_t> lagrangian;

    Eigen::VectorX<scalar_t> var;

    template<typename Problem>
    explicit ProblemMemory(Problem &prob) :
        objective(prob.objective),
        variable_bounds(prob.variable_bounds),
        constraints(prob.constraints),
        lagrangian(prob.lagrangian),
        var(prob.variables())
    {
        prob.set_decision_variable(var);
    }
};

template <typename scalar_t>
std::ostream& operator<<(std::ostream &o, const ProblemInfo<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>> &info)
{
    o << "==== Problem Sparsity Information ====\n";
    o << "Variables    : " << info.objective.variables << std::endl;

    o << "Constraints  : " << info.constraints.rows << std::endl;
    o << "  Non-zeros  : " << info.constraints.jacobian.nonZeros() << std::endl;

    o << "Variable bnds: " << info.variable_bounds.rows << std::endl;

    o << "Objective    : " << info.objective.hessian.rows() << std::endl;
    o << "  Non-zeros  : " << info.objective.hessian.nonZeros() << std::endl;

    o << "Lagrangian   : " << info.lagrangian.hessian.rows() << std::endl;
    o << "  Non-zeros  : " << info.lagrangian.hessian.nonZeros() << std::endl;
    return o;
}

template <typename scalar_t>
std::ostream& operator<<(std::ostream& o, const ProblemInfo<BSMatrixTape, BSMatrixDenseConstruction<scalar_t>>& info)
{
	o << "==== Problem Tape Information ====\n";
    o << "Variables    : " << info.objective.variables << std::endl;
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

} // namespace laopt

#endif // LAOPT_PROBLEM_HPP
