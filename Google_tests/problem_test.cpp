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

template<typename scalar_t, int N>
struct OCP_DoubleIntegrator
{
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

    // template<typename OutGradient, typename OutHessian, typename Weight, typename X, typename U>
    // EIGEN_STRONG_INLINE scalar_t
    // weightedsum(lampc::Hessian,
    //             OutGradient&& outgradient, OutHessian&& outhessian,
    //             const Eigen::MatrixBase<Weight>& weight,
    //             const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss,
    //             const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<U>& uss) noexcept
    // {
    //   outgradient(seqN(0,2)) =  2*p.Q*(x-xss);
    //   outgradient(seqN(2,2)) = -2*p.Q*(x-xss);
    //   outgradient(seqN(4,1)) =  2*p.R*(u-uss);
    //   outgradient(seqN(5,1)) = -2*p.R*(u-uss);

    //   outhessian(seqN(0,2),seqN(0,2)) =  2*p.Q;
    //   outhessian(seqN(0,2),seqN(2,2)) = -2*p.Q;
    //   outhessian(seqN(2,2),seqN(2,2)) =  2*p.Q;
    //   outhessian(seqN(2,2),seqN(0,2)) = -2*p.Q;
    //   outhessian(seqN(4,1),seqN(4,1)) =  2*p.R;
    //   outhessian(seqN(5,1),seqN(4,1)) = -2*p.R;
    //   outhessian(seqN(5,1),seqN(5,1)) =  2*p.R;
    //   outhessian(seqN(4,1),seqN(5,1)) = -2*p.R;

    //   return impl(x,xss,u,uss);
    // }

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

  template<typename Constraints, typename Dtype>
  void eval_constraints(Dtype dtype, Constraints& con)
  {
    // Initial state
    con.add(dtype, id, X[0]) == x0;

    // Dynamics
    for(int i=0; i<N-1; i++) con.add(dtype, dsys_eq, X[i+1], X[i], U[i]) == 0;

    // Add constraints on the variables
    for(auto& u : U) -1 <= con.add(dtype, id, u) <= 1;
    for(auto& x : X) -5 <= con.add(dtype, id, x) <= 5;
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

  OCP_DoubleIntegrator<scalar_t, N> ocp;

  auto sparsity = lampc::generate_problem(lampc::ProblemSparsity<scalar_t>(), ocp);

  std::cout << "Jacobian sparsity structure \n" << sparsity.constraints.jacobian << std::endl;
  std::cout << "Hessian sparsity structure \n" << sparsity.objective.hessian << std::endl;
  std::cout << "Lagrangian sparsity structure \n" << sparsity.lagrangian.hessian << std::endl;
  std::cout << sparsity << std::endl;
  std::cout << "\n\n";

  auto tape = lampc::generate_problem(lampc::ProblemTape<scalar_t>(sparsity), ocp);
  std::cout << tape << std::endl;

  lampc::Problem<scalar_t> prob(tape);
  lampc::ProblemMemory<scalar_t> mem(prob, tape);

  ocp.define_variables(prob);
  mem.objective.weights.array() = 1;
  mem.lagrangian.weights.array() = 1;

  ocp.x0 << 1,2;
  for(int i=0; i<N; i++) ocp.X[i] << i,i+1;
  for(int i=0; i<N-1; i++) ocp.U[i] << i;
  ocp.xss << 3,4;
  ocp.uss << 1;

  prob.constraints.initialize();
  prob.objective.initialize();
  prob.lagrangian.initialize();
  ocp.eval_constraints(lampc::Jacobian(), prob.constraints);
  ocp.eval_objective(lampc::Hessian(), prob.objective);
  ocp.eval_objective(lampc::Hessian(), prob.lagrangian);
  ocp.eval_constraints(lampc::Hessian(), prob.lagrangian);
 
  std::cout << "prob.constraints.ub = " << mem.constraints.ub.transpose() << std::endl;
  std::cout << "prob.constraints.lb = " << mem.constraints.lb.transpose() << std::endl;

  std::cout << std::setprecision(2) << std::defaultfloat;
  std::cout << "constraints\n";
  std::cout << "\tvalue = " << mem.constraints.value.transpose() << std::endl;
  std::cout << "\tjacobian\n" << Eigen::MatrixX<scalar_t>(mem.constraints.jacobian) << std::endl;

  std::cout << "objective\n";
  std::cout << "\tvalue = " << prob.objective.value << std::endl;
  std::cout << "\tgradient\n" << mem.objective.gradient.transpose() << std::endl;
  std::cout << "\thessian\n" << Eigen::MatrixX<scalar_t>(mem.objective.hessian) << std::endl;

  std::cout << "lagrangian\n";
  std::cout << "\tvalue = " << prob.lagrangian.value << std::endl;
  std::cout << "\tgradient\n" << mem.lagrangian.gradient.transpose() << std::endl;
  std::cout << "\thessian\n" << Eigen::MatrixX<scalar_t>(mem.lagrangian.hessian) << std::endl;
};



#ifdef NDEBUG
TEST(ProblemTest, SpeedTest) {
  using scalar_t = double;
  const int N = 10;

  OCP_DoubleIntegrator<scalar_t, N> ocp;

  auto sparsity = lampc::generate_problem(lampc::ProblemSparsity<scalar_t>(), ocp);
  auto tape = lampc::generate_problem(lampc::ProblemTape<scalar_t>(sparsity), ocp);

  lampc::Problem<scalar_t> prob(tape);  
  lampc::ProblemMemory<scalar_t> mem(prob, tape);
  ocp.define_variables(prob);
  mem.objective.weights.array() = 1;
  mem.lagrangian.weights.array() = 1;

  for(int i=0; i<N; i++) ocp.X[i] << i,i+1;
  for(int i=0; i<N-1; i++) ocp.U[i] << i;
  ocp.xss << 3,4;
  ocp.uss << 1;

  prob.constraints.initialize();
  prob.objective.initialize();
  prob.lagrangian.initialize();
  ocp.eval_constraints(lampc::Jacobian(), prob.constraints);
  ocp.eval_objective(lampc::Hessian(), prob.objective);
  ocp.eval_constraints(lampc::Hessian(), prob.lagrangian);
  ocp.eval_objective(lampc::Hessian(), prob.lagrangian);

  // Finally, the call!
  const std::size_t NUM_EXP = 1000000;
  double acc = 0;
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < NUM_EXP; ++i)
  {
    ocp.X[3] << i,i+1;
    prob.constraints.initialize();
    prob.objective.initialize();
    ocp.eval_constraints(lampc::Jacobian(), prob.constraints);
    ocp.eval_objective(lampc::Hessian(), prob.objective);
    acc += mem.constraints.value(4);
  }
  auto end = std::chrono::steady_clock::now();

  std::cout << "LAMPC time: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count()
      << " ns  (acc = " << acc << ")" << std::endl;

  double lampc_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count();

  Eigen::Vector<scalar_t,2> xp(2);
  Eigen::Vector<scalar_t,2> x(2);
  Eigen::Vector<scalar_t,1> u(1);
  Eigen::Vector<scalar_t,2> xss(2);
  Eigen::Vector<scalar_t,1> uss(1);
  Eigen::Vector<scalar_t,1> weight_stage(1);
  Eigen::Vector<scalar_t,1> weight_term(1);
  xp << 1,2; x << 3,4; u << 5; xss << 6,7; uss << 8;
  weight_stage << 1;
  weight_term << 1;

  Eigen::MatrixX<scalar_t> J(2,5);
  Eigen::VectorX<scalar_t> V(2);

  Eigen::VectorX<scalar_t> G_stage(6);
  Eigen::MatrixX<scalar_t> H_stage(6,6);
  Eigen::VectorX<scalar_t> G_term(4);
  Eigen::MatrixX<scalar_t> H_term(4,4);

  Eigen::MatrixX<scalar_t> Target(20,20);

  start = std::chrono::steady_clock::now();
  acc = 0;
  for(int i = 0; i < NUM_EXP; ++i)
  {
    for(int j=0; j<N-1; j++)
    {
      x(0) = i + j;
      ocp.dsys_eq(lampc::Jacobian(), V,J, xp,x,u);
      acc += V(0);

      Target(seqN(3,2),4) = V;
      for(int k=0; k<J.cols(); k++) Target(seqN(3,2),k) = J(all,k);

      acc += ocp.stage_cost.weightedsum(lampc::Hessian(), G_stage,H_stage, weight_stage, x,xss,u,uss);

      Target(seqN(3,6),1) = G_stage;
      for(int k=0; k<H_stage.cols(); k++) Target(seqN(3,2),k) = H_stage(all,k);
    }
    // acc += ocp.terminal_cost.weightedsum(x, xss, weight_term, G_term, H_term);
  }
  end = std::chrono::steady_clock::now();
  std::cout << "acc = " << acc << std::endl;

  std::cout << "Direct computation of problem elements: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count()
      << " ns" << std::endl;

  double direct_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count();


  std::cout << "LAMPC overhead = " << (lampc_time - direct_time) / direct_time * 100 << std::endl;
};
#else

#include "colormod.hpp"
TEST(ProblemTest, SpeedTest) {
  std::cout << Color::Modifier(Color::FG_RED) << "Skipping speed test because we're in debug mode";
  std::cout << Color::Modifier(Color::FG_DEFAULT) << std::endl;
}

#endif
}