#ifndef __DOUBLE_INTEGRATOR__HPP
#define __DOUBLE_INTEGRATOR__HPP

#include <iostream>
#include "lampc.hpp"

/**
 * Simple linear MPC problem
 */
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
      return A.template cast<Scalar>() * x + B.template cast<Scalar>() * u;
    }
  };

  // Return the output (position) of the system
  struct output_t : public lampc::MakeDifferentiable<output_t>
  {
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar impl(
      const Eigen::MatrixBase<X>& x) noexcept
    {
      return x(0);
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
      return 100*(x-xss).dot(p.Q.template cast<Scalar>()*(x-xss));
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
  scalar_t ref; // Output target

  // Functions we use to define the problem
  sys_t sys; // Continuous-time dynamics
  dsys_t dsys; // Discrete-time dynamics
  output_t output;

  lampc::functions::eq<dsys_t> dsys_eq; // dsys(x,u) - xp
  lampc::functions::dsteady_state_t<dsys_t> steady_state; // dsys(xss,uss) - xss

  // Tuning parameters
  stage_cost_t stage_cost;
  terminal_cost_t terminal_cost;

  lampc::functions::id id; // Identity

  OCP_DoubleIntegrator(scalar_t step_size) :
                      dsys(sys, step_size),
                      dsys_eq(dsys),
                      steady_state(dsys),
                      stage_cost(*this),
                      terminal_cost(*this)
  {
    x0 << 0,0;
    ref = 0;
  };

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
    for(auto& x : X) -50 <= bnd.add(x) <= 50;
    for(auto& x : X) -2 <= bnd.add(x(seqN(1,1))) <= 2;
    -50 <= bnd.add(xss) <= 50;
    -1 <= bnd.add(uss) <= 1;
  }

  template<typename Constraints, typename Dtype>
  void eval_constraints(Dtype dtype, Constraints& con)
  {
    // Initial state
    con.add(dtype, id, X[0]) == x0;

    // Dynamics
    for(int i=0; i<N-1; i++) con.add(dtype, dsys_eq, X[i+1], X[i], U[i]) == 0;

    con.add(dtype, steady_state, xss, uss) == 0;
    con.add(dtype, output, xss) == ref;
  }

  template<typename Objective, typename Dtype>
  void eval_objective(Dtype dtype, Objective& obj)
  {
    for(int i=0; i<X.size()-1; i++)
      obj.add(dtype, stage_cost, X[i], xss, U[i], uss);
    obj.add(dtype, terminal_cost, X[N-1], xss);
  }
};

using scalar_t = double;
constexpr int N = 20;
using OCP = OCP_DoubleIntegrator<scalar_t, N>;
using Problem = lampc::Problem<OCP>;

Problem tape_to_problem(lampc::TapeInfo<OCP>& tape, OCP& ocp);

#endif
