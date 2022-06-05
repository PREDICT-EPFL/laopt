/**
 * Unit test for the construction and computation of LAMPC problems.
 */

#include <iostream>
#include "lampc.hpp"
#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace {

template<typename _scalar_t, int N>
struct OCP_DoubleIntegrator
{  
  using scalar_t = _scalar_t; // lampc::Problem assumes that the user code will contain a scalar_t

  /**
   * Continuous-time dynamics - double integrator
   */
  struct sys_t
  {
    Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    impl( const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept        
    {
      return x(0)*(A.template cast<Scalar>() * x + B.template cast<Scalar>() * u);
    }
  };

  // Discretized dynamics
  using dsys_t = lampc::functions::RK4<sys_t, scalar_t>;

  struct stage_cost_t : public lampc::MakeDifferentiable<stage_cost_t>
  {
    OCP_DoubleIntegrator& p;
    stage_cost_t(OCP_DoubleIntegrator& p) : p(p) {}

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar impl(
      const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss,
      const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<U>& uss) noexcept
    {
      return (x-xss).dot(p.Q.template cast<Scalar>()*(x-xss)) + (u-uss).dot(p.R.template cast<Scalar>()*(u-uss));
    }

    // Bring all the calls that we're not overloading into scope
    using lampc::MakeDifferentiable<stage_cost_t>::weightedsum; 

    template<typename OutGradient, typename OutHessian, typename Weight, typename X, typename U>
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(lampc::Hessian,
                OutGradient&& outgradient, OutHessian&& outhessian,
                const Eigen::MatrixBase<Weight>& weight,
                const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss,
                const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<U>& uss) noexcept
    {
      outgradient(seqN(0,2)) +=  2*p.Q*(x-xss);
      outgradient(seqN(2,2)) += -2*p.Q*(x-xss);
      outgradient(seqN(4,1)) +=  2*p.R*(u-uss);
      outgradient(seqN(5,1)) += -2*p.R*(u-uss);

      outhessian(seqN(0,2),seqN(0,2)) +=  2*p.Q;
      outhessian(seqN(0,2),seqN(2,2)) += -2*p.Q;
      outhessian(seqN(2,2),seqN(2,2)) +=  2*p.Q;
      outhessian(seqN(2,2),seqN(0,2)) += -2*p.Q;
      outhessian(seqN(4,1),seqN(4,1)) +=  2*p.R;
      outhessian(seqN(5,1),seqN(4,1)) += -2*p.R;
      outhessian(seqN(5,1),seqN(5,1)) +=  2*p.R;
      outhessian(seqN(4,1),seqN(5,1)) += -2*p.R;

      return impl(x,xss,u,uss);
    }

  };

  struct terminal_cost_t : public lampc::MakeDifferentiable<terminal_cost_t>
  {
    OCP_DoubleIntegrator& p;
    terminal_cost_t(OCP_DoubleIntegrator& p) : p(p) {}

    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar impl(
      const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss) noexcept
    {
      return 20*(x-xss).dot(p.Q.template cast<Scalar>()*(x-xss));
    }
  };

  static constexpr int nx = 2;
  static constexpr int nu = 1;

  template<size_t n>
  using Variable = lampc::Variable<scalar_t, n>;
  std::array<Variable<nx>, N>   X;
  std::array<Variable<nu>, N-1> U;
  Variable<nx> xss;
  Variable<nu> uss;

  // Parameters shared across different elements of the cost function
  Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
  Eigen::Matrix<scalar_t, 1, 1> R{2};

  Eigen::Vector<scalar_t, 2> x0; // Initial state

  // Functions we use to define the problem
  sys_t sys; // Continuous-time dynamics
  dsys_t dsys; // Discrete-time dynamics

  lampc::functions::eq<dsys_t> dsys_eq; // dsys(x,u) - xp

  // Tuning parameters
  stage_cost_t stage_cost;
  terminal_cost_t terminal_cost;

  lampc::functions::id id; // Identity

  OCP_DoubleIntegrator() :
                      dsys(sys, 0.1),
                      dsys_eq(dsys),
                      stage_cost(*this),
                      terminal_cost(*this)
  {};

  template<typename OptProblem>
  void define_variables(OptProblem& problem)
  {
    for(int i=0; i<N-1; i++)
    {
      problem.add_variable(X[i]);
      problem.add_variable(U[i]);
    }
    problem.add_variable(X[N-1]);
    problem.add_variable(xss);
    problem.add_variable(uss);
  }

  template<typename Bounds>
  void eval_variable_bounds(Bounds& bnd)
  {
    // Add constraints on the variables
    for(auto& u : U) -1 <= bnd.add(u) <= 1;
    for(auto& x : X) -5 <= bnd.add(x) <= 5;
  }

  template<typename Constraints, typename Dtype>
  void eval_constraints(Dtype dtype, Constraints& con)
  {
    // Initial state
    con.add(dtype, id, X[0]) == x0;

    // Dynamics
    for(int i=0; i<N-1; i++) con.add(dtype, dsys_eq, X[i+1], X[i], U[i]) == 0;
  }

  template<typename Objective, typename Dtype>
  void eval_objective(Dtype dtype, Objective& obj)
  {
    for(int i=0; i<X.size()-1; i++)
      obj.add(dtype, stage_cost, X[i], xss, U[i], uss);
    obj.add(dtype, terminal_cost, X[N-1], xss);
  }
};



TEST(ProblemTest, DoubleIntegrator) {
  using scalar_t = double;
  const int N = 10;

  using OCP = OCP_DoubleIntegrator<scalar_t, N>;
  OCP ocp;

  auto sparsity = lampc::generate_sparsity(ocp);

  std::cout << "Jacobian sparsity structure \n" << sparsity.constraints.jacobian << std::endl;
  std::cout << "Hessian sparsity structure \n" << sparsity.objective.hessian << std::endl;
  std::cout << "Lagrangian sparsity structure \n" << sparsity.lagrangian.hessian << std::endl;
  std::cout << sparsity << std::endl;
  std::cout << "\n\n";

  auto tape = lampc::generate_tape(ocp, sparsity);
  std::cout << tape << std::endl;

  lampc::Problem<OCP> prob(ocp, tape);

  std::cout << std::setprecision(2) << std::defaultfloat;

  // Test computation into pre-allocated vectors
  Eigen::VectorX<scalar_t> con(prob.constraints.rows());
  Eigen::VectorX<scalar_t> lb(prob.constraints.rows());
  Eigen::VectorX<scalar_t> ub(prob.constraints.rows());

  Eigen::VectorX<scalar_t> lb_x(prob.num_variables());
  Eigen::VectorX<scalar_t> ub_x(prob.num_variables());

  scalar_t obj;
  Eigen::VectorX<scalar_t> gradient(prob.num_variables());
  Eigen::VectorX<scalar_t> weights(prob.objective.rows());
  Eigen::SparseMatrix<scalar_t> hessian;
  prob.objective.hessian.allocate_memory(hessian);

  scalar_t lag;
  Eigen::VectorX<scalar_t> lag_gradient(prob.num_variables());
  Eigen::VectorX<scalar_t> lag_weights(prob.lagrangian.rows());
  Eigen::SparseMatrix<scalar_t> lag_hessian;
  prob.lagrangian.hessian.allocate_memory(lag_hessian);

  Eigen::VectorX<scalar_t> var(prob.num_variables());
  prob.set_decision_variable(var);
  ocp.x0 << 1,2;
  for(int i=0; i<N; i++) ocp.X[i] << i,i+1;
  for(int i=0; i<N-1; i++) ocp.U[i] << i;
  ocp.xss << 3,4;
  ocp.uss << 1;

  lag_weights.array() = 1;
  weights.array() = 1;

  std::cout << "\n\n" << "=== Evaluating constraints into reference buffers ===" << std::endl;
  prob.eval_constraints(lampc::Eval(), var, con,lb,ub);
  obj = prob.eval_objective(lampc::Eval(), var, weights);
  obj = prob.eval_objective(lampc::Gradient(), var, weights,gradient);
  lag = prob.eval_lagrangian(lampc::Eval(), var, lag_weights);
  lag = prob.eval_lagrangian(lampc::Gradient(), var, lag_weights,lag_gradient);
  std::cout << "con = " << con.transpose() << std::endl;
  std::cout << "lb = " << lb.transpose() << std::endl;
  std::cout << "ub = " << ub.transpose() << std::endl;
  std::cout << "obj = " << obj << std::endl;
  std::cout << "gradient = " << gradient.transpose() << std::endl;
  std::cout << "lag = " << lag << std::endl;
  std::cout << "lag_gradient = " << lag_gradient.transpose() << std::endl;

  std::cout << "\n\n" << "=== Evaluating constraints and jacobian into reference buffers ===" << std::endl;
  Eigen::VectorX<scalar_t> jac_buffer(prob.constraints.jacobian.sparsity_structure.nonZeros());
  prob.eval_constraints(lampc::Jacobian(), var, con,lb,ub,jac_buffer);
  prob.eval_variable_bounds(lb_x, ub_x);
  obj = prob.eval_objective(lampc::Hessian(), var, weights,gradient,hessian);
  lag = prob.eval_lagrangian(lampc::Hessian(), var, lag_weights,lag_gradient,lag_hessian);
  std::cout << "con = " << con.transpose() << std::endl;
  std::cout << "lb = " << lb.transpose() << std::endl;
  std::cout << "ub = " << ub.transpose() << std::endl;
  std::cout << "jacobian_buffer = " << jac_buffer.transpose() << std::endl;
  std::cout << "obj = " << obj << std::endl;
  std::cout << "gradient = " << gradient.transpose() << std::endl;
  std::cout << "hessian_buffer = " << Eigen::Map<Eigen::VectorX<scalar_t>>(hessian.valuePtr(),hessian.nonZeros()).transpose() << std::endl;
  std::cout << "lag = " << lag << std::endl;
  std::cout << "lag_gradient = " << lag_gradient.transpose() << std::endl;
  std::cout << "lag_hessian_buffer = " << Eigen::Map<Eigen::VectorX<scalar_t>>(lag_hessian.valuePtr(),lag_hessian.nonZeros()).transpose() << std::endl;
  std::cout << "lb_x = " << lb_x.transpose() << std::endl;
  std::cout << "ub_x = " << ub_x.transpose() << std::endl;

  // Test computation into memory structure
  lampc::ProblemMemory<scalar_t> mem(prob);

  prob.set_decision_variable(mem.var);
  ocp.x0 << 1,2;
  for(int i=0; i<N; i++) ocp.X[i] << i,i+1;
  for(int i=0; i<N-1; i++) ocp.U[i] << i;
  ocp.xss << 3,4;
  ocp.uss << 1;

  mem.objective.weights.array() = 1;
  mem.lagrangian.weights.array() = 1;

  std::cout << "\n\n" << "=== Evaluating constraints into memory structure ===" << std::endl;
  prob.eval_constraints(lampc::Jacobian(), mem.var, mem.constraints.value, mem.constraints.lb, mem.constraints.ub, mem.constraints.jacobian);
  prob.eval_variable_bounds(mem.variable_bounds.lb, mem.variable_bounds.ub);
  obj = prob.eval_objective(lampc::Hessian(), mem.var, mem.objective.weights,mem.objective.gradient,mem.objective.hessian);
  lag = prob.eval_lagrangian(lampc::Hessian(), mem.var, mem.lagrangian.weights,mem.lagrangian.gradient,mem.lagrangian.hessian);
  std::cout << "con = " << mem.constraints.value.transpose() << std::endl;
  std::cout << "lb = " << mem.constraints.lb.transpose() << std::endl;
  std::cout << "ub = " << mem.constraints.ub.transpose() << std::endl;
  std::cout << "jacobian_buffer = " << mem.constraints.jacobian_buffer.transpose() << std::endl;
  std::cout << "obj = " << obj << std::endl;
  std::cout << "gradient = " << mem.objective.gradient.transpose() << std::endl;
  std::cout << "hessian_buffer = " << mem.objective.hessian_buffer.transpose() << std::endl;
  std::cout << "lag = " << lag << std::endl;
  std::cout << "lag_gradient = " << mem.lagrangian.gradient.transpose() << std::endl;
  std::cout << "lag_hessian_buffer = " << mem.lagrangian.hessian_buffer.transpose() << std::endl;

  std::cout << "lb_x = " << mem.variable_bounds.lb.transpose() << std::endl;
  std::cout << "ub_x = " << mem.variable_bounds.ub.transpose() << std::endl;
};

// #ifdef NDEBUG
// TEST(ProblemTest, SpeedTest) {
//   using scalar_t = double;
//   const int N = 10;

//   OCP_DoubleIntegrator<scalar_t, N> ocp;

//   auto sparsity = lampc::generate_problem(lampc::ProblemSparsity<scalar_t>(), ocp);
//   auto tape = lampc::generate_problem(lampc::ProblemTape<scalar_t>(sparsity), ocp);

//   lampc::Problem<scalar_t> prob(tape);  
//   lampc::ProblemMemory<scalar_t> mem(prob, tape);
//   ocp.define_variables(prob);
//   mem.objective.weights.array() = 1;
//   mem.lagrangian.weights.array() = 1;

//   for(int i=0; i<N; i++) ocp.X[i] << i,i+1;
//   for(int i=0; i<N-1; i++) ocp.U[i] << i;
//   ocp.xss << 3,4;
//   ocp.uss << 1;

//   prob.constraints.initialize();
//   prob.objective.initialize();
//   prob.lagrangian.initialize();
//   ocp.eval_constraints(lampc::Jacobian(), prob.constraints);
//   ocp.eval_objective(lampc::Hessian(), prob.objective);
//   ocp.eval_constraints(lampc::Hessian(), prob.lagrangian);
//   ocp.eval_objective(lampc::Hessian(), prob.lagrangian);

//   // Finally, the call!
//   const std::size_t NUM_EXP = 1000000;
//   double acc = 0;
//   auto start = std::chrono::steady_clock::now();
//   for(int i = 0; i < NUM_EXP; ++i)
//   {
//     ocp.X[3] << i,i+1;
//     prob.constraints.initialize();
//     prob.objective.initialize();
//     ocp.eval_constraints(lampc::Jacobian(), prob.constraints);
//     ocp.eval_objective(lampc::Hessian(), prob.objective);
//     acc += mem.constraints.value(4);
//   }
//   auto end = std::chrono::steady_clock::now();

//   std::cout << "LAMPC time: "
//       << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count()
//       << " ns  (acc = " << acc << ")" << std::endl;

//   double lampc_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count();

//   Eigen::Vector<scalar_t,2> xp(2);
//   Eigen::Vector<scalar_t,2> x(2);
//   Eigen::Vector<scalar_t,1> u(1);
//   Eigen::Vector<scalar_t,2> xss(2);
//   Eigen::Vector<scalar_t,1> uss(1);
//   Eigen::Vector<scalar_t,1> weight_stage(1);
//   Eigen::Vector<scalar_t,1> weight_term(1);
//   xp << 1,2; x << 3,4; u << 5; xss << 6,7; uss << 8;
//   weight_stage << 1;
//   weight_term << 1;

//   Eigen::MatrixX<scalar_t> J(2,5);
//   Eigen::VectorX<scalar_t> V(2);

//   Eigen::VectorX<scalar_t> G_stage(6);
//   Eigen::MatrixX<scalar_t> H_stage(6,6);
//   Eigen::VectorX<scalar_t> G_term(4);
//   Eigen::MatrixX<scalar_t> H_term(4,4);

//   Eigen::MatrixX<scalar_t> Target(20,20);

//   start = std::chrono::steady_clock::now();
//   acc = 0;
//   for(int i = 0; i < NUM_EXP; ++i)
//   {
//     for(int j=0; j<N-1; j++)
//     {
//       x(0) = i + j;
//       ocp.dsys_eq(lampc::Jacobian(), V,J, xp,x,u);
//       acc += V(0);

//       Target(seqN(3,2),4) = V;
//       for(int k=0; k<J.cols(); k++) Target(seqN(3,2),k) = J(all,k);

//       acc += ocp.stage_cost.weightedsum(lampc::Hessian(), G_stage,H_stage, weight_stage, x,xss,u,uss);

//       Target(seqN(3,6),1) = G_stage;
//       for(int k=0; k<H_stage.cols(); k++) Target(seqN(3,2),k) = H_stage(all,k);
//     }
//     // acc += ocp.terminal_cost.weightedsum(x, xss, weight_term, G_term, H_term);
//   }
//   end = std::chrono::steady_clock::now();
//   std::cout << "acc = " << acc << std::endl;

//   std::cout << "Direct computation of problem elements: "
//       << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count()
//       << " ns" << std::endl;

//   double direct_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count();


//   std::cout << "LAMPC overhead = " << (lampc_time - direct_time) / direct_time * 100 << std::endl;
// };
// #else

// #include "colormod.hpp"
// TEST(ProblemTest, SpeedTest) {
//   std::cout << Color::Modifier(Color::FG_RED) << "Skipping speed test because we're in debug mode";
//   std::cout << Color::Modifier(Color::FG_DEFAULT) << std::endl;
// }

// #endif
}