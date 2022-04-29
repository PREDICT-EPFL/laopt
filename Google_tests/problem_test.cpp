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

/**
 * Continuous-time dynamics - double integrator
 */
template<typename scalar_t>
struct sys_t
{
    Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

    template<typename diff_t>
    EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> 
    impl( const Eigen::Ref< const Eigen::Vector<diff_t, 2> >& x, 
          const Eigen::Ref< const Eigen::Vector<diff_t, 1> >& u) noexcept        
    {
        return x(0)*(A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
    }
};

// Discretized dynamics
template<typename scalar_t>
using dsys_t = lampc::functions::RK4<sys_t<scalar_t>, scalar_t, 2, 1>;


// Parameters shared across different elements of the cost function
template<typename scalar_t>
struct cost_param_t
{
  Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
  Eigen::Matrix<scalar_t, 1, 1> R{2};
};

template<typename scalar_t>
struct stage_cost_t : public lampc::MakeDifferentiable<stage_cost_t<scalar_t>, scalar_t, 1, 2,2,1,1>
{
  cost_param_t<scalar_t>& p;
  stage_cost_t(cost_param_t<scalar_t>& p) : p(p) {}

  template<typename diff_t>
  EIGEN_STRONG_INLINE Eigen::Vector<diff_t,1> impl(
          const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& x,
          const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& xss,
          const Eigen::Ref<const Eigen::Vector<diff_t, 1>>& u,
          const Eigen::Ref<const Eigen::Vector<diff_t, 1>>& uss) noexcept
  {
    return (x-xss).transpose()*(p.Q.template cast<diff_t>())*(x-xss) + (u-uss).transpose()*(p.R.template cast<diff_t>())*(u-uss);
  }
};

template<typename scalar_t>
struct terminal_cost_t : public lampc::MakeDifferentiable<terminal_cost_t<scalar_t>, scalar_t, 1, 2,2>
{
  cost_param_t<scalar_t>& p;
  terminal_cost_t(cost_param_t<scalar_t>& p) : p(p) {}

  template<typename diff_t>
  EIGEN_STRONG_INLINE Eigen::Vector<diff_t,1> impl(
          const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& x,
          const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& xss) noexcept
  {
    return static_cast<diff_t>(20)*(x-xss).transpose()*(p.Q.template cast<diff_t>())*(x-xss);
  }
};


// Test the individual functions making up the problem
TEST(ProblemTest, OCP_DoubleIntegrator_Functions) {
  using scalar_t = double;
  sys_t<scalar_t> sys; // Continuous-time dynamics
  dsys_t<scalar_t> dsys(sys, 0.1); // Discrete-time dynamics

  cost_param_t<scalar_t> cost_params;
  stage_cost_t<scalar_t> stage_cost(cost_params);
  terminal_cost_t<scalar_t> terminal_cost(cost_params);

  Eigen::Vector<scalar_t, 2> x;
  Eigen::Vector<scalar_t, 1> u;
  x << 1,2;
  u << 3;

  dsys(lampc::Jacobian(), x,u);

  Eigen::Vector<scalar_t, 2> xss;
  Eigen::Vector<scalar_t, 1> uss;
  xss << 4,5;
  uss << 6;

  stage_cost(lampc::Hessian(), x,xss,u,uss);
  terminal_cost(lampc::Hessian(), x,xss);

};

template<typename OptProblem, typename scalar_t>
struct OCP_DoubleIntegrator
{
  using Variable = typename OptProblem::Variable;

  // Order here defines the order in the optimization problem
  Variable& x;
  Variable& u;
  Variable& xss;
  Variable& uss;
  const int N;

  // Functions we use to define the problem
  sys_t<scalar_t> sys; // Continuous-time dynamics
  dsys_t<scalar_t> dsys; // Discrete-time dynamics
  lampc::functions::eq<dsys_t<scalar_t>, 2, 2,1> dsys_eq; // dsys(x,u) - xp

  // Tuning parameters
  cost_param_t<scalar_t> cost_param;
  stage_cost_t<scalar_t> stage_cost;
  terminal_cost_t<scalar_t> terminal_cost;

  lampc::functions::id<scalar_t, 2> idx;

  OCP_DoubleIntegrator(OptProblem& opt, int N) :
                                N(N),
                                u(opt.variable(1,N)),
                                xss(opt.variable(2)), 
                                uss(opt.variable(1)),
                                x(opt.variable(2,N)), 
                                dsys(sys, 0.1),
                                dsys_eq(dsys),
                                stage_cost(cost_param),
                                terminal_cost(cost_param)
  {};

  void eval_constraints(OptProblem& opt)
  {
    opt.constraint(idx, x(0));
    opt.constraint(idx, x(0));
    for(int i=0; i<N-1; i++)
    {
      x(i)(0) = i;
      opt.constraint(dsys_eq, x(i+1), x(i),u(i));
    }
    opt.constraint(idx, x(0));
  }

  // template<typename D = lampc::Hessian>
  // void eval_objective()
  // {
  //   for(int i=0; i<x.cols()-1; i++)
  //     objective.add(D(), stage_cost, x(i), xss(0), u(i), uss(0));
  //   objective.add(D(), terminal_cost, x(N-1), xss(0));
  // }
};


// template<template<typename> class UserCode, typename... Args>
// auto makeProblem(Args... args)
// {
//   // Compute the sparsity structure
//   ProblemSparsity sparsity; // Sparsity capture object
//   UserCode<ProblemSparsity> user_sparsity(sparsity, args...);
//   Eigen::VectorX<scalar_t> var(sparsity.size);
//   sparsity.set_variable(var);
//   user_sparsity.eval_constraints();

//   return user_sparsity;

//   // ProblemTape tape(sparsity.generate());
//   // UserCode<ProblemTape> user_tape(tape, args...);
//   // user_tape.set_variable(var);

//   // Problem prob(tape.generate());
//   // UserCode<Problem> user(prob, args...);
//   // return user;
// }


// template<typename scalar_t, template<typename,typename> class UserProblem, typename... Args>
// auto makeProblem(Args... args)
// {
//   // Extract sparsity pattern
//   UserProblem<scalar_t, lampc::ProblemSparsity<scalar_t>> sparsity(args...);
//   sparsity.initialize();
//   sparsity.eval_constraints();

//   std::cout << "sparsity =\n" << sparsity.constraint.jacobian.get_sparsity() << std::endl;

//   // Compute copy sequence
//   UserProblem<scalar_t, lampc::ProblemTape<scalar_t>> tape(args...);
//   tape.initialize(sparsity.generate());
//   tape.eval_constraints();

//   // OCP_DoubleIntegrator<scalar_t, lampc::Problem<scalar_t>> problem(args...);
//   // problem(tape.generate());

//   // return problem;
// }


TEST(ProblemTest, Constraint) {
  using scalar_t = double;

  // Compute the sparsity structure
  lampc::ProblemSparsity<scalar_t> sparsity; // Sparsity capture object
  OCP_DoubleIntegrator<lampc::ProblemSparsity<scalar_t>, scalar_t> ocp_sparsity(sparsity, 10);
  ocp_sparsity.eval_constraints(sparsity);

  std::cout << "sparsity = \n" << sparsity.constraint.jacobian.get_sparsity() << std::endl;

  lampc::ProblemTape<scalar_t> tape(sparsity.generate());
  OCP_DoubleIntegrator<lampc::ProblemTape<scalar_t>, scalar_t> ocp_tape(tape, 10);
  ocp_tape.eval_constraints(tape);

  std::cout << "jacobian copies = " << tape.constraint.jacobian.copy_segments << std::endl;
  std::cout << "value copies = " << tape.constraint.value.copy_segments << std::endl;
  std::cout << "lb copies = " << tape.constraint.lb.copy_segments << std::endl;

  lampc::Problem<scalar_t> prob(tape.generate());
  OCP_DoubleIntegrator<lampc::Problem<scalar_t>, scalar_t> ocp(prob, 10);

  auto mem = prob.make_problem_memory();

  // Finally, the call!
  const std::size_t NUM_EXP = 1000;
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < NUM_EXP; ++i)
  {
    ocp.eval_constraints(prob);
  }
  auto end = std::chrono::steady_clock::now();

  std::cout << "Jacobian time: "
      << std::chrono::duration_cast<std::chrono::microseconds>((end - start)/(double)NUM_EXP).count()
      << " us" << std::endl;

  std::cout << "g = " << mem.g.transpose() << std::endl;
  std::cout << "g_jacobian = \n" << Eigen::MatrixX<double>(mem.g_jacobian) << std::endl;

};


TEST(ProblemTest, OCP_DoubleIntegrator) {
  using scalar_t = double;
  int N = 10;
  // auto prob = lampc::make_problem<OCP_DoubleIntegrator, scalar_t>(N);
  // auto mem = lampc::make_problem_memory<scalar_t>(prob);

  // for(int i=0; i<N; i++)
  // {
  //   prob.x(i) << 1+i,1+2*i;
  //   prob.u(i) << 3*i+2;
  // }
  // prob.xss(0) << 4,5;
  // prob.uss(0) << 6;

  // // Evaluate the value only
  // prob.eval_constraints<lampc::Eval>();
  // {
  //   Eigen::MatrixX<scalar_t> ground(20,1);
  //   ground << 1,1,-0.88332,-1.7889,-0.14519,-0.81134,1.719,1.2707,5.7423,5.0093,14.1,11.295,31.399,21.536,67.268,37.913,141.34,63.72,292.43,103.82;
  //   EXPECT_TRUE(ground.isApprox(mem.g, 1e-4));
  // }

  // // Evaluate the jacobian and the value
  // prob.eval_constraints<lampc::Jacobian>();
  // {
  //   Eigen::MatrixX<scalar_t> ground(20,1);
  //   ground << 1,1,-0.88332,-1.7889,-0.14519,-0.81134,1.719,1.2707,5.7423,5.0093,14.1,11.295,31.399,21.536,67.268,37.913,141.34,63.72,292.43,103.82;
  //   EXPECT_TRUE(ground.isApprox(mem.g, 1e-4));
  // }
  // {
  //   auto ground = triplet_to_sparse<double>(20,29,{{0,0,1},{1,0,0},{2,0,1.1283},{3,0,0.21182},{0,1,0},{1,1,1},{2,1,0.11206},{3,1,1.0108},{2,2,-1},{3,2,0},{4,2,1.5079},{5,2,0.60573},{2,3,0},{3,3,-1},{4,3,0.29091},{5,3,1.0632},{4,4,-1},{5,4,0},{6,4,2.1889},{7,4,1.1445},{4,5,0},{5,5,-1},{6,5,0.59904},{7,5,1.1815},{6,6,-1},{7,6,0},{8,6,3.4822},{9,6,1.9222},{6,7,0},{7,7,-1},{8,7,1.1666},{9,7,1.4042},{8,8,-1},{9,8,0},{10,8,6.0522},{11,8,3.0864},{8,9,0},{9,9,-1},{10,9,2.2646},{11,9,1.7904},{10,10,-1},{11,10,0},{12,10,11.291},{13,10,4.8618},{10,11,0},{11,11,-1},{12,11,4.4481},{13,11,2.4283},{12,12,-1},{13,12,0},{14,12,22.022},{15,12,7.5798},{12,13,0},{13,13,-1},{14,13,8.8211},{15,13,3.4454},{14,14,-1},{15,14,0},{16,14,43.775},{17,14,11.716},{14,15,0},{15,15,-1},{16,15,17.506},{17,15,5.0217},{16,16,-1},{17,16,0},{18,16,86.959},{19,16,17.936},{16,17,0},{17,17,-1},{18,17,34.435},{19,17,7.4058},{18,18,-1},{19,18,0},{18,19,0},{19,19,-1},{2,20,0.0057945},{3,20,0.10591},{4,21,0.032191},{5,21,0.24229},{6,22,0.10596},{7,22,0.42919},{8,23,0.28967},{9,23,0.69898},{10,24,0.72574},{11,24,1.1023},{12,25,1.7261},{13,25,1.7159},{14,26,3.9444},{15,26,2.6529},{16,27,8.6895},{17,27,4.0751},{18,28,18.469},{19,28,6.2086}});
  //   EXPECT_TRUE(ground.isApprox(mem.g_jacobian, 1e-4));
  //   std::cout << "Jg = \n" << Eigen::MatrixX<scalar_t>(mem.g_jacobian) << std::endl; 
  //   std::cout << "Jg = \n" << Eigen::MatrixX<scalar_t>(ground) << std::endl; 
  // }

  // prob.eval_objective<lampc::Eval>();
  // {
  //   EXPECT_NEAR(prob.objective.value,7325,1e-4);
  // }

  // prob.eval_objective<lampc::Jacobian>();
  // {
  //   Eigen::Vector<scalar_t,29> ground;
  //   ground << -6,-8,-4,-4,-2,0,0,4,2,8,4,12,6,16,8,20,10,24,240,560,-16,-4,8,20,32,44,56,68,80;
  //   EXPECT_TRUE(ground.isApprox(mem.obj_gradient, 1e-4));
  // }

  // prob.eval_objective(); // Default to hessian
  // {
  //   EXPECT_NEAR(prob.objective.value,7325,1e-4);
  // }

  // {
  //   Eigen::MatrixX<scalar_t> ground(29,1);
  //   ground << -6,-8,-4,-4,-2,0,0,4,2,8,4,12,6,16,8,20,10,24,240,560,-16,-4,8,20,32,44,56,68,80;
  //   EXPECT_TRUE(ground.isApprox(mem.obj_gradient, 1e-4));
  // }

  // {
  //   auto ground = triplet_to_sparse<double>(33,33,{{0,0,2},{1,0,0},{20,0,0},{30,0,-2},{31,0,0},{32,0,0},{0,1,0},{1,1,2},{20,1,0},{30,1,0},{31,1,-2},{32,1,0},{2,2,2},{3,2,0},{21,2,0},{30,2,-2},{31,2,0},{32,2,0},{2,3,0},{3,3,2},{21,3,0},{30,3,0},{31,3,-2},{32,3,0},{4,4,2},{5,4,0},{22,4,0},{30,4,-2},{31,4,0},{32,4,0},{4,5,0},{5,5,2},{22,5,0},{30,5,0},{31,5,-2},{32,5,0},{6,6,2},{7,6,0},{23,6,0},{30,6,-2},{31,6,0},{32,6,0},{6,7,0},{7,7,2},{23,7,0},{30,7,0},{31,7,-2},{32,7,0},{8,8,2},{9,8,0},{24,8,0},{30,8,-2},{31,8,0},{32,8,0},{8,9,0},{9,9,2},{24,9,0},{30,9,0},{31,9,-2},{32,9,0},{10,10,2},{11,10,0},{25,10,0},{30,10,-2},{31,10,0},{32,10,0},{10,11,0},{11,11,2},{25,11,0},{30,11,0},{31,11,-2},{32,11,0},{12,12,2},{13,12,0},{26,12,0},{30,12,-2},{31,12,0},{32,12,0},{12,13,0},{13,13,2},{26,13,0},{30,13,0},{31,13,-2},{32,13,0},{14,14,2},{15,14,0},{27,14,0},{30,14,-2},{31,14,0},{32,14,0},{14,15,0},{15,15,2},{27,15,0},{30,15,0},{31,15,-2},{32,15,0},{16,16,2},{17,16,0},{28,16,0},{30,16,-2},{31,16,0},{32,16,0},{16,17,0},{17,17,2},{28,17,0},{30,17,0},{31,17,-2},{32,17,0},{18,18,40},{19,18,0},{30,18,-40},{31,18,0},{18,19,0},{19,19,40},{30,19,0},{31,19,-40},{0,20,0},{1,20,0},{20,20,4},{30,20,0},{31,20,0},{32,20,-4},{2,21,0},{3,21,0},{21,21,4},{30,21,0},{31,21,0},{32,21,-4},{4,22,0},{5,22,0},{22,22,4},{30,22,0},{31,22,0},{32,22,-4},{6,23,0},{7,23,0},{23,23,4},{30,23,0},{31,23,0},{32,23,-4},{8,24,0},{9,24,0},{24,24,4},{30,24,0},{31,24,0},{32,24,-4},{10,25,0},{11,25,0},{25,25,4},{30,25,0},{31,25,0},{32,25,-4},{12,26,0},{13,26,0},{26,26,4},{30,26,0},{31,26,0},{32,26,-4},{14,27,0},{15,27,0},{27,27,4},{30,27,0},{31,27,0},{32,27,-4},{16,28,0},{17,28,0},{28,28,4},{30,28,0},{31,28,0},{32,28,-4},{0,30,-2},{1,30,0},{2,30,-2},{3,30,0},{4,30,-2},{5,30,0},{6,30,-2},{7,30,0},{8,30,-2},{9,30,0},{10,30,-2},{11,30,0},{12,30,-2},{13,30,0},{14,30,-2},{15,30,0},{16,30,-2},{17,30,0},{18,30,-40},{19,30,0},{20,30,0},{21,30,0},{22,30,0},{23,30,0},{24,30,0},{25,30,0},{26,30,0},{27,30,0},{28,30,0},{30,30,58},{31,30,0},{32,30,0},{0,31,0},{1,31,-2},{2,31,0},{3,31,-2},{4,31,0},{5,31,-2},{6,31,0},{7,31,-2},{8,31,0},{9,31,-2},{10,31,0},{11,31,-2},{12,31,0},{13,31,-2},{14,31,0},{15,31,-2},{16,31,0},{17,31,-2},{18,31,0},{19,31,-40},{20,31,0},{21,31,0},{22,31,0},{23,31,0},{24,31,0},{25,31,0},{26,31,0},{27,31,0},{28,31,0},{30,31,0},{31,31,58},{32,31,0},{0,32,0},{1,32,0},{2,32,0},{3,32,0},{4,32,0},{5,32,0},{6,32,0},{7,32,0},{8,32,0},{9,32,0},{10,32,0},{11,32,0},{12,32,0},{13,32,0},{14,32,0},{15,32,0},{16,32,0},{17,32,0},{20,32,-4},{21,32,-4},{22,32,-4},{23,32,-4},{24,32,-4},{25,32,-4},{26,32,-4},{27,32,-4},{28,32,-4},{30,32,0},{31,32,0},{32,32,36}});
  //   EXPECT_TRUE(ground.isApprox(mem.obj_hessian, 1e-4));
  // }
}












  // {
  //   auto self = this;
  //   auto q = lampc::rk4<scalar_t, diff_t, 2>([self](
  //     const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& x,
  //     const Eigen::Ref<const Eigen::Vector<diff_t, 1>>& u)
  //   {
  //     return self->template sys<diff_t>(x,u);
  //   }, 0.1, x, u);
  //   return q;
  // }
  // make_jacobian(dsys, 2, 2,1);

  // /** Dynamics constraints */
  // template<typename diff_t=scalar_t>
  // EIGEN_STRONG_INLINE Vector<diff_t, 2> dsys_equality(
  //         const Ref<const Vector<diff_t, 2>>& xp, 
  //         const Ref<const Vector<diff_t, 2>>& x, 
  //         const Ref<const Vector<diff_t, 1>>& u) noexcept
  // {
  //     return this->dsys(x,u) - xp;
  // }
  // make_jacobian(dsys_equality, 2, 2,2,1); // Automatic jacobian

  // // Custom jacobian
  // EIGEN_STRONG_INLINE auto dsys_equality(lampc::Jacobian, 
  //         std::pair<lampc::Segment, const Ref<const Vector<scalar_t, 2>>&> xp, 
  //         std::pair<lampc::Segment, const Ref<const Vector<scalar_t, 2>>&> x, 
  //         std::pair<lampc::Segment, const Ref<const Vector<scalar_t, 1>>&> u) noexcept
  // {
  //   std::cout << "Here!\n";
  //   // Eigen::Matrix<scalar_t,2,5> J;
  //   // Eigen::Vector<scalar_t,2> val;

  //   // // dsys(x,u)
  //   // auto R = J(all,seqN(2,3));
  //   // std::tie(val, R) = this->dsys(lampc::Jacobian(), x, u);

  //   // // -xp 
  //   // val -= xp;
  //   // J(all,seqN(0,2)) = -Eigen::Matrix<scalar_t,2,2>::Identity();

  //   return std::make_pair(val, J);
  // }  

  
//   template<typename diff_t=scalar_t>
//   EIGEN_STRONG_INLINE Vector<diff_t,1> stage_cost(
//           const Ref<const Vector<diff_t, 2>>& x,
//           const Ref<const Vector<diff_t, 2>>& xss,
//           const Ref<const Vector<diff_t, 1>>& u,
//           const Ref<const Vector<diff_t, 1>>& uss) noexcept
//   {
//     return (x-xss).transpose()*(Q.template cast<diff_t>())*(x-xss) + (u-uss).transpose()*(R.template cast<diff_t>())*(u-uss);
//   }
//   make_hessian(stage_cost, 1, 2,2,1,1);

//   template<typename diff_t=scalar_t>
//   EIGEN_STRONG_INLINE Vector<diff_t,1> terminal_cost(
//           const Ref<const Vector<diff_t, 2>>& x,
//           const Ref<const Vector<diff_t, 2>>& xss) noexcept
//   {
//     return static_cast<diff_t>(20)*(x-xss).transpose()*(Q.template cast<diff_t>())*(x-xss);
//   }
//   make_hessian(terminal_cost, 1, 2,2);
// };


// template<typename scalar_t, typename Problem=lampc::Problem<scalar_t>>
// struct OCP_DoubleIntegrator : public DoubleIntegrator<scalar_t>, public Problem
// {
//   // References into the decision variable
//   using Variable = typename Problem::Variable;
//   Variable& x;
//   Variable& u;
//   Variable& xss;
//   Variable& uss;
//   const int N;

//   Eigen::Vector<scalar_t, 2> x_lb; // Upper and lower state bounds
//   Eigen::Vector<scalar_t, 2> x_ub;
//   Eigen::Vector<scalar_t, 2> x0; // Initial state

//   scalar_t u_ub() { return x0.sum() + 3; }

//   OCP_DoubleIntegrator(int N) : N(N),
//                                 x(this->variable(2,N)), // Order here defines the order
//                                 u(this->variable(1,N)), // in the optimization problem
//                                 xss(this->variable(2)), 
//                                 uss(this->variable(1))
//   {
//     x_lb << -5,-5;
//     x_ub << 5,5;
//   };

//   template<typename D = lampc::Jacobian>
//   void eval_constraints()
//   {
//     this->constraints << (lampc::id<2>(D(), x(0)) == x_lb);

//     auto t = (lampc::id<2>(D(), x(0)) == x_lb);
//     std::cout << "t.bounds = \n" << t.bounds << std::endl;

//     for(int i=0; i<N-1; i++)
//       this->constraints << this->dsys_equality(D(), x(i+1),x(i),u(i));

//     // this->constraints << (lampc::id<2>(D(), x(0)) == x0);
//     // for(int i=0; i<N-1; i++)
//     //   this->constraints << (this->dsys_equality(D(), x(i+1),x(i),u(i)) == 0);

//     // for(int i=0; i<N-1; i++)
//     // {      
//     //   this->constraints << (-x_lb <= lampc::id(D(), x(i)) <= x_ub)
//     //                     << (-1 <= lampc::id(D(), u(i)) <= u_ub(x0));
//     // }
//   }

//   template<typename D = lampc::Hessian>
//   void eval_objective()
//   {
//     this->objective = 0; // Must be called to initialize the object each time

//     for(int i=0; i<x.cols()-1; i++)
//       this->objective += this->stage_cost(D(), x(i), xss(0), u(i), uss(0));
//     this->objective += this->terminal_cost(D(), x(N-1), xss(0));
//   }
// };



}