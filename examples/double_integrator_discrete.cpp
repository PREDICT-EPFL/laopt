#ifndef __DOUBLE_INTEGRATOR__HPP
#define __DOUBLE_INTEGRATOR__HPP

#include "laopt/lampc.hpp"

/**
 * Simple linear MPC problem
 */
template<typename _scalar_t, int N>
struct OCP_DoubleIntegrator
{  
  using scalar_t = _scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

  /**
   * Discrete-time dynamics - double integrator
   */
  struct sys_t
  {
    Eigen::Matrix<scalar_t, 2, 2> A;
    Eigen::Matrix<scalar_t, 2, 1> B;

    sys_t(scalar_t h)
    {
      A << 1, h, 0, 1;
      B << h*h/2.0, h;
    }

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    impl( const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept        
    {
      return A.template cast<Scalar>() * x + B.template cast<Scalar>() * u;
    }
  };

  struct stage_cost_t : public laopt::MakeDifferentiable<stage_cost_t>
  {
    Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
    Eigen::Matrix<scalar_t, 1, 1> R{0.01};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar 
    impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
      return x.dot(p.Q.template cast<Scalar>()*x) + u.dot(p.R.template cast<Scalar>()*u);
    }
  };

  static constexpr int nx = 2;
  static constexpr int nu = 1;

  template<size_t n>
  using Variable = laopt::Variable<scalar_t, n>;
  std::array<Variable<nx>, N>   X;
  std::array<Variable<nu>, N-1> U;

  Eigen::Vector<scalar_t, 2> x0; // Initial state

  // Functions we use to define the problem
  sys_t sys;
  laopt::functions::eq<sys_t> sys_eq; // sys(x,u) - xp
  stage_cost_t stage_cost;

  laopt::functions::id id; // Identity

  OCP_DoubleIntegrator(scalar_t step_size) :
                      sys(step_size),
                      sys_eq(sys)
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
  }

  template<typename Bounds>
  void eval_variable_bounds(Bounds& bnd)
  {
    // Add constraints on the variables
    for(auto& u : U) -1 <= bnd.add(u) <= 1;
    for(auto& x : X) -50 <= bnd.add(x) <= 50;
  }

  template<typename Constraints, typename Dtype>
  void eval_constraints(Dtype dtype, Constraints& con)
  {
    // Initial state
    con.add(dtype, id, X[0]) == x0;

    // Dynamics
    for(int i=0; i<N-1; i++) con.add(dtype, sys_eq, X[i+1], X[i], U[i]) == 0;
  }

  template<typename Objective, typename Dtype>
  void eval_objective(Dtype dtype, Objective& obj)
  {
    for(int i=0; i<X.size()-1; i++)
      obj.add(dtype, stage_cost, X[i], U[i]);
  }
};

using scalar_t = double;
constexpr int N = 10;
using OCP = OCP_DoubleIntegrator<scalar_t, N>;
using Problem = laopt::Problem<OCP>;

Problem tape_to_problem(laopt::TapeInfo<OCP>& tape, OCP& ocp);

#endif
