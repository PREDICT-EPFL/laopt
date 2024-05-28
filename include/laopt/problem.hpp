#ifndef LAOPT_PROBLEM_HPP
#define LAOPT_PROBLEM_HPP

#include <numeric>
#include <iterator>
#include "laopt/problem_dispatch_types.hpp"
#include "laopt/problem_vector_function.hpp"
#include "laopt/problem_weighted_sum_function.hpp"
#include "laopt/opt_problem.hpp"
#include "laopt/problem_evaluators/problem_size_evaluator.hpp"
#include "laopt/problem_evaluators/objective_evaluator.hpp"
#include "laopt/problem_evaluators/decision_variable_setter.hpp"
#include "laopt/problem_evaluators/variable_bounds_evaluator.hpp"
#include "laopt/problem_evaluators/vector_constraints_evaluator.hpp"
#include "laopt/problem_evaluators/weighted_sum_constraints_evaluator.hpp"

namespace laopt
{

template<typename MatrixType, typename VectorType>
struct ProblemInfo
{
    WeightedSumFunctionInfo<MatrixType, VectorType> objective;
    VectorFunctionInfo<MatrixType, VectorType>      variable_bounds;
    VectorFunctionInfo<MatrixType, VectorType>      constraints;
	WeightedSumFunctionInfo<MatrixType, VectorType> lagrangian;
};

template<typename UserCode, typename Scalar, typename MatrixType, typename VectorType>
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
class Sparsity;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using SparsityInfo = typename Sparsity<UserCode, scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
class Tape;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using TapeInfo = typename Tape<UserCode, scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
class BSTape;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
using BSTapeInfo = typename BSTape<UserCode, scalar_t>::Info;

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
class Problem;
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
class BSProblem;

template<typename UserCode_, typename Scalar, typename MatrixType, typename VectorType>
class ProblemBase
{
public:
    using UserCode = UserCode_;
    using scalar_t = Scalar;
    using matrix_t = MatrixType;
    using vector_t = VectorType;

protected:
    std::shared_ptr<UserCode> user_code;

    /**
     * Constructor for sparsity
     */
    explicit ProblemBase(const std::shared_ptr<UserCode>& user_code) : user_code(user_code)
    {
        this->eval_problem_size();
    }

    /**
     * Constructor for tape recording and deployment
     */
    template<typename OtherMatrixType, typename OtherVectorType>
    explicit ProblemBase(const std::shared_ptr<UserCode>& user_code, const ProblemInfo<OtherMatrixType, OtherVectorType>& info) :
        user_code(user_code), objective(info.objective), variable_bounds(info.variable_bounds),
        constraints(info.constraints), lagrangian(info.lagrangian) {}

public:
    WeightedSumFunction<MatrixType, VectorType> objective;
    VectorFunction<MatrixType, VectorType>      variable_bounds;
    VectorFunction<MatrixType, VectorType>      constraints;
    WeightedSumFunction<MatrixType, VectorType> lagrangian;

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
        user_code->define_problem(decision_variable_setter);
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
    using Info = ProblemInfo<MatrixType, VectorType>;
    Info generate()
    {
        Info info;

        info.constraints = constraints.generate();
        info.objective = objective.generate();
        info.lagrangian = lagrangian.generate();
        info.variable_bounds = variable_bounds.generate();

        return info;
    }

    template<typename UserCodeTemp, typename ScalarTemp>
    friend SparsityInfo<UserCodeTemp> generate_sparsity(const std::shared_ptr<UserCodeTemp>& user_code);
    template<typename UserCodeTemp, typename ScalarTemp>
    friend TapeInfo<UserCodeTemp> generate_tape(const std::shared_ptr<UserCodeTemp>& user_code, const SparsityInfo<UserCodeTemp>& sparsity);
    template<typename UserCodeTemp, typename ScalarTemp>
    friend BSTapeInfo<UserCodeTemp> generate_bs_tape(const std::shared_ptr<UserCodeTemp>& user_code, const SparsityInfo<UserCodeTemp>& sparsity);

protected:
    /**
	 * Compute the various elements of the problem by calling the
	 * user-code. Stores the result in constraints/objective and lagrangian respectively
	 *
	 * These calls are only made internally when the Matrix and Vector types manage
	 * their own memory (i.e., during construction). They should not be called
	 * during deployment.
	 */
	void eval_problem_size()
    {
        ProblemSizeEvaluator<MatrixType, VectorType> problem_size_evaluator(variable_bounds, objective, constraints, lagrangian);
        user_code->define_problem(problem_size_evaluator);
    }

    template<typename DType>
    void eval_objective_no_memory(DType)
    {
        ObjectiveEvaluator<DType, MatrixType, VectorType> objective_evaluator(objective);
        user_code->define_problem(objective_evaluator);
    }

    void eval_variable_bounds_no_memory()
    {
        VariableBoundsEvaluator<MatrixType, VectorType> variable_bounds_evaluator(variable_bounds);
        user_code->define_problem(variable_bounds_evaluator);
    }

    template<typename DType>
    void eval_constraints_no_memory(DType)
    {
        VectorConstraintsEvaluator<DType, MatrixType, VectorType> constraints_evaluator(constraints);
        user_code->define_problem(constraints_evaluator);
    }

    template<typename DType>
    void eval_lagrangian_no_memory(DType, const scalar_t obj_factor = static_cast<scalar_t>(1))
    {
        ObjectiveEvaluator<DType, MatrixType, VectorType> lagrangian_objective_evaluator(lagrangian, obj_factor);
        user_code->define_problem(lagrangian_objective_evaluator);
        WeightedSumConstraintsEvaluator<DType, MatrixType, VectorType> lagrangian_constraints_evaluator(lagrangian);
        user_code->define_problem(lagrangian_constraints_evaluator);
    }
};

/**
 * Generates sparsity information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
SparsityInfo<UserCode> generate_sparsity(const std::shared_ptr<UserCode>& user_code)
{
	Sparsity<UserCode, scalar_t> prob(user_code);

    Eigen::VectorX<scalar_t> var(prob.variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_no_memory(Jacobian{});
    prob.eval_variable_bounds_no_memory();
    prob.eval_objective_no_memory(Gradient{});
    prob.eval_objective_no_memory(Hessian{});
    prob.eval_lagrangian_no_memory(Hessian{});

    return prob.generate();
}

/**
 * Generates tape information for the given user code.
 */
template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
TapeInfo<UserCode> generate_tape(const std::shared_ptr<UserCode>& user_code, const SparsityInfo<UserCode>& sparsity)
{
	Tape<UserCode, scalar_t> prob(user_code, sparsity);

    Eigen::VectorX<scalar_t> var(prob.variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_no_memory(Jacobian{});
    prob.eval_variable_bounds_no_memory();
    prob.eval_objective_no_memory(Gradient{});
    prob.eval_objective_no_memory(Hessian{});
    prob.eval_lagrangian_no_memory(Hessian{});

    return prob.generate();
}

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
TapeInfo<UserCode> generate_tape(const std::shared_ptr<UserCode>& user_code)
{
    return generate_tape(user_code, generate_sparsity(user_code));
}

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
BSTapeInfo<UserCode> generate_bs_tape(const std::shared_ptr<UserCode>& user_code, const SparsityInfo<UserCode>& sparsity)
{
    BSTape<UserCode, scalar_t> prob(user_code, sparsity);

    Eigen::VectorX<scalar_t> var(prob.variables());
    var.setZero();
    prob.set_decision_variable(var);

    prob.eval_constraints_no_memory(Jacobian{});
    prob.eval_variable_bounds_no_memory();
    prob.eval_objective_no_memory(Gradient{});
    prob.eval_objective_no_memory(Hessian{});
    prob.eval_lagrangian_no_memory(Hessian{});

    return prob.generate();
}

template<typename UserCode, typename scalar_t = typename UserCode::scalar_t>
BSTapeInfo<UserCode> generate_bs_tape(const std::shared_ptr<UserCode>& user_code)
{
    return generate_bs_tape(user_code, generate_sparsity(user_code));
}

template<typename UserCode, typename scalar_t>
class Sparsity : public ProblemBase<UserCode, scalar_t, BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>
{
public:
    using Base = ProblemBase<UserCode, scalar_t, BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>>;
    explicit Sparsity(const std::shared_ptr<UserCode>& user_code) : Base(user_code) {}
};

template<typename UserCode, typename scalar_t>
class Tape : public ProblemBase<UserCode, scalar_t, BSMatrixTape<scalar_t>, BSMatrixDenseConstruction<scalar_t>>
{
public:
    using Base = ProblemBase<UserCode, scalar_t, BSMatrixTape<scalar_t>, BSMatrixDenseConstruction<scalar_t>>;
    explicit Tape(const std::shared_ptr<UserCode>& user_code) : Base(user_code, generate_sparsity(user_code)) {}
    Tape(const std::shared_ptr<UserCode>& user_code, const SparsityInfo<UserCode, scalar_t>& info) : Base(user_code, info) {}
};

template<typename UserCode, typename scalar_t>
class BSTape : public ProblemBase<UserCode, scalar_t, BSBSMatrixTape<scalar_t>, BSMatrixDenseConstruction<scalar_t>>
{
public:
    using Base = ProblemBase<UserCode, scalar_t, BSBSMatrixTape<scalar_t>, BSMatrixDenseConstruction<scalar_t>>;
    explicit BSTape(const std::shared_ptr<UserCode>& user_code) : Base(user_code, generate_sparsity(user_code)) {}
    BSTape(const std::shared_ptr<UserCode>& user_code, const SparsityInfo<UserCode, scalar_t>& info) : Base(user_code, info) {}
};

template<typename UserCode, typename scalar_t>
class Problem : public ProblemBase<UserCode, scalar_t, BSMatrix<Eigen::SparseMatrix<scalar_t>>, BSMatrixDenseDeployment<scalar_t>>
{
public:
    using Base = ProblemBase<UserCode, scalar_t, BSMatrix<Eigen::SparseMatrix<scalar_t>>, BSMatrixDenseDeployment<scalar_t>>;
    explicit Problem(const std::shared_ptr<UserCode>& user_code) : Base(user_code, generate_tape(user_code)) {}
    Problem(const std::shared_ptr<UserCode>& user_code, const TapeInfo<UserCode, scalar_t>& info) : Base(user_code, info) {}
};

template<typename UserCode, typename scalar_t>
class BSProblem : public ProblemBase<UserCode, scalar_t, BSMatrix<Eigen::BlockSparseMatrix<scalar_t>>, BSMatrixDenseDeployment<scalar_t>>
{
public:
    using Base = ProblemBase<UserCode, scalar_t, BSMatrix<Eigen::BlockSparseMatrix<scalar_t>>, BSMatrixDenseDeployment<scalar_t>>;
    explicit BSProblem(const std::shared_ptr<UserCode>& user_code) : Base(user_code, generate_bs_tape(user_code)) {}
    BSProblem(const std::shared_ptr<UserCode>& user_code, const BSTapeInfo<UserCode, scalar_t>& info) : Base(user_code, info) {}
};

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
    explicit ProblemMemory(const std::shared_ptr<Problem>& prob) :
        objective(prob->objective),
        variable_bounds(prob->variable_bounds),
        constraints(prob->constraints),
        lagrangian(prob->lagrangian),
        var(prob->variables())
    {
        prob->set_decision_variable(var);
    }
};

template <typename scalar_t>
std::ostream& operator<<(std::ostream &o, const ProblemInfo<BSMatrixSparsity, BSMatrixDenseConstruction<scalar_t>> &info)
{
    o << "==== Problem Sparsity Information ====\n";
    o << "Variables    : " << info.objective.variables << std::endl;
    o << "Constraints  : " << info.constraints.rows << std::endl;
    o << "  Non-zeros  : " << info.constraints.jacobian.sparsity_pattern.nonZeros() << std::endl;
    o << "Variable bnds: " << info.variable_bounds.rows << std::endl;
    o << "Objective    : " << info.objective.hessian.sparsity_pattern.rows() << std::endl;
    o << "  Non-zeros  : " << info.objective.hessian.sparsity_pattern.nonZeros() << std::endl;
    o << "Lagrangian   : " << info.lagrangian.hessian.sparsity_pattern.rows() << std::endl;
    o << "  Non-zeros  : " << info.lagrangian.hessian.sparsity_pattern.nonZeros() << std::endl;
    return o;
}

template <typename scalar_t>
std::ostream& operator<<(std::ostream& o, const ProblemInfo<BSMatrixTape<scalar_t>, BSMatrixDenseConstruction<scalar_t>>& info)
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

template <typename scalar_t>
std::ostream& operator<<(std::ostream& o, const ProblemInfo<BSBSMatrixTape<scalar_t>, BSMatrixDenseConstruction<scalar_t>>& info)
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
