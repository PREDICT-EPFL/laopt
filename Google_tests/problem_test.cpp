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
 * Define the dynamics for a simple double integrator model discretized by RK4
 */
template<typename scalar_t>
struct DoubleIntegrator
{
  // Example : Double integrator
  Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
  Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

  Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
  Eigen::Matrix<scalar_t, 1, 1> R{2};

  /** Continuous system dynamics */
  template<typename diff_t=scalar_t>
  EIGEN_STRONG_INLINE Vector<diff_t, 2> 
      sys(const Ref<const Matrix<diff_t, 2, 1>>& x, const Ref<const Vector<diff_t, 1>>& u) noexcept
  {
    return x(0) * (A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
  }
  // make_jacobian(sys, 2, 2,1);

  /** Discrete system dynamics */
  template<typename diff_t=scalar_t>
  EIGEN_STRONG_INLINE Vector<diff_t, 2> 
      dsys(const Ref<const Matrix<diff_t, 2, 1>>& x, const Ref<const Vector<diff_t, 1>>& u) noexcept
  {
    auto self = this;
    auto q = lampc::rk4<scalar_t, diff_t, 2>([self](
      const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& x,
      const Eigen::Ref<const Eigen::Vector<diff_t, 1>>& u)
    {
      return self->template sys<diff_t>(x,u);
    }, 0.1, x, u);
    return q;
  }
  make_jacobian(dsys, 2, 2,1);

  /** Dynamics constraints */
  template<typename diff_t=scalar_t>
  EIGEN_STRONG_INLINE Vector<diff_t, 2> dsys_equality(
          const Ref<const Vector<diff_t, 2>>& xp, 
          const Ref<const Vector<diff_t, 2>>& x, 
          const Ref<const Vector<diff_t, 1>>& u) noexcept
  {
      return this->dsys(x,u) - xp;
  }
  make_jacobian(dsys_equality, 2, 2,2,1); // Automatic jacobian

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

  
  template<typename diff_t=scalar_t>
  EIGEN_STRONG_INLINE Vector<diff_t,1> stage_cost(
          const Ref<const Vector<diff_t, 2>>& x,
          const Ref<const Vector<diff_t, 2>>& xss,
          const Ref<const Vector<diff_t, 1>>& u,
          const Ref<const Vector<diff_t, 1>>& uss) noexcept
  {
    return (x-xss).transpose()*(Q.template cast<diff_t>())*(x-xss) + (u-uss).transpose()*(R.template cast<diff_t>())*(u-uss);
  }
  make_hessian(stage_cost, 1, 2,2,1,1);

  template<typename diff_t=scalar_t>
  EIGEN_STRONG_INLINE Vector<diff_t,1> terminal_cost(
          const Ref<const Vector<diff_t, 2>>& x,
          const Ref<const Vector<diff_t, 2>>& xss) noexcept
  {
    return static_cast<diff_t>(20)*(x-xss).transpose()*(Q.template cast<diff_t>())*(x-xss);
  }
  make_hessian(terminal_cost, 1, 2,2);
};


template<typename scalar_t, typename Problem=lampc::Problem<scalar_t>>
struct OCP_DoubleIntegrator : public DoubleIntegrator<scalar_t>, public Problem
{
  // References into the decision variable
  using Variable = typename Problem::Variable;
  Variable& x;
  Variable& u;
  Variable& xss;
  Variable& uss;
  const int N;

  OCP_DoubleIntegrator(int N) : N(N),
                                x(this->variable(2,N)), // Order here defines the order
                                u(this->variable(1,N)), // in the optimization problem
                                xss(this->variable(2)), 
                                uss(this->variable(1))
  {};

  template<typename D = lampc::Jacobian>
  void eval_constraints()
  {
    this->constraints << lampc::id(D(), x(0));
    for(int i=0; i<x.cols()-1; i++)
      this->constraints << this->dsys_equality(D(), x(i+1),x(i),u(i));
  }

  template<typename D = lampc::Hessian>
  void eval_objective()
  {
    this->objective = 0; // Must be called to initialize the object each time

    for(int i=0; i<x.cols()-1; i++)
      this->objective += this->stage_cost(D(), x(i), xss(0), u(i), uss(0));
    this->objective += this->terminal_cost(D(), x(N-1), xss(0));
  }
};

TEST(ProblemTest, OCP_DoubleIntegrator) {
  using scalar_t = double;
  int N = 10;
  auto prob = lampc::makeProblem<OCP_DoubleIntegrator, scalar_t>(N);
  auto mem = lampc::makeProblemMemory<scalar_t>(prob);

  std::cout << "mem.obj_weight.data = " << mem.obj_weight.data() << std::endl;

  for(int i=0; i<N; i++)
  {
    prob.x(i) << 1,1;
    prob.u(i) << 1;
  }
  prob.xss(0) << 0,0;
  prob.uss(0) << 0;
  prob.eval_constraints();

  std::cout << std::setprecision(2);
  std::cout << "g_jacobian = \n" << Eigen::MatrixX<scalar_t>(mem.g_jacobian) << std::endl;
  std::cout << "g = \n" << mem.g << std::endl;

  prob.eval_objective();

  std::cout << std::setprecision(10);
  std::cout << "obj = " << prob.objective.value << std::endl;
  std::cout << "obj gradient = " << mem.obj_gradient.transpose() << std::endl;
  std::cout << "obj hessian = \n" << Eigen::MatrixX<scalar_t>(mem.obj_hessian) << std::endl;
 
    // Eigen::Vector<double,18> ground_val;
    // ground_val << -0.3,-0.4,-0.287757,-0.4,-0.250028,-0.4,-0.185253,-0.4,-0.091788,-0.4,0.0321,-0.4,0.188236,-0.4,0.378541,-0.4,0.605036,-0.4;
    // EXPECT_TRUE(ground_val.isApprox(t.val, 1e-4));


    // // Test sparsity structure
    // {
    //     auto ground = triplet_to_sparse<double>(18,29,{{0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,1},{2,2,1},{3,2,1},{0,3,1},{1,3,1},{2,3,1},{3,3,1},{2,4,1},{3,4,1},{4,4,1},{5,4,1},{2,5,1},{3,5,1},{4,5,1},{5,5,1},{4,6,1},{5,6,1},{6,6,1},{7,6,1},{4,7,1},{5,7,1},{6,7,1},{7,7,1},{6,8,1},{7,8,1},{8,8,1},{9,8,1},{6,9,1},{7,9,1},{8,9,1},{9,9,1},{8,10,1},{9,10,1},{10,10,1},{11,10,1},{8,11,1},{9,11,1},{10,11,1},{11,11,1},{10,12,1},{11,12,1},{12,12,1},{13,12,1},{10,13,1},{11,13,1},{12,13,1},{13,13,1},{12,14,1},{13,14,1},{14,14,1},{15,14,1},{12,15,1},{13,15,1},{14,15,1},{15,15,1},{14,16,1},{15,16,1},{16,16,1},{17,16,1},{14,17,1},{15,17,1},{16,17,1},{17,17,1},{16,18,1},{17,18,1},{16,19,1},{17,19,1},{0,20,1},{1,20,1},{2,21,1},{3,21,1},{4,22,1},{5,22,1},{6,23,1},{7,23,1},{8,24,1},{9,24,1},{10,25,1},{11,25,1},{12,26,1},{13,26,1},{14,27,1},{15,27,1},{16,28,1},{17,28,1}});
    //     EXPECT_TRUE(ground.isApprox(t.jacobian, 0));
    // }

    // // Compute value and jacobian
    // {
    //   for(int i=0; i<10; i++)
    //   {
    //     t.x(i) << 0.3*i,0.4*i;
    //     t.u(i) << 0;
    //   }
    //   t.eval_equality_constraints();
    //   auto ground = triplet_to_sparse<double>(18,29,{{0,0,1},{1,0,0},{0,1,0},{1,1,1},{0,2,-1},{1,2,-0},{2,2,1.0408},{3,2,0},{0,3,-0},{1,3,-1},{2,3,0.031224},{3,3,1},{2,4,-1},{3,4,-0},{4,4,1.0833},{5,4,0},{2,5,-0},{3,5,-1},{4,5,0.064997},{5,5,1},{4,6,-1},{5,6,-0},{6,6,1.1275},{7,6,0},{4,7,-0},{5,7,-1},{6,7,0.10147},{7,7,1},{6,8,-1},{7,8,-0},{8,8,1.1735},{9,8,0},{6,9,-0},{7,9,-1},{8,9,0.14082},{9,9,1},{8,10,-1},{9,10,-0},{10,10,1.2214},{11,10,0},{8,11,-0},{9,11,-1},{10,11,0.1832},{11,11,1},{10,12,-1},{11,12,-0},{12,12,1.2712},{13,12,0},{10,13,-0},{11,13,-1},{12,13,0.2288},{13,13,1},{12,14,-1},{13,14,-0},{14,14,1.3231},{15,14,0},{12,15,-0},{13,15,-1},{14,15,0.2778},{15,15,1},{14,16,-1},{15,16,-0},{16,16,1.3771},{17,16,0},{14,17,-0},{15,17,-1},{16,17,0.3304},{17,17,1},{16,18,-1},{17,18,-0},{16,19,-0},{17,19,-1},{0,20,0},{1,20,0},{2,21,0.00047467},{3,21,0.030608},{4,22,0.002003},{5,22,0.062465},{6,23,0.0047545},{7,23,0.095622},{8,24,0.0089178},{9,24,0.13013},{10,25,0.014702},{11,25,0.16605},{12,26,0.022339},{13,26,0.20343},{14,27,0.032083},{15,27,0.24234},{16,28,0.044218},{17,28,0.28282}});
    //   EXPECT_TRUE(ground.isApprox(t.jacobian, 1e-4));

    //   Eigen::Vector<double,18> ground_val;
    //   ground_val << -0.3,-0.4,-0.287757,-0.4,-0.250028,-0.4,-0.185253,-0.4,-0.091788,-0.4,0.0321,-0.4,0.188236,-0.4,0.378541,-0.4,0.605036,-0.4;
    //   EXPECT_TRUE(ground_val.isApprox(t.val, 1e-4));
    // }
}


}