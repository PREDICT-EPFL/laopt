#include <iostream>
#include "lampc.hpp"

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
      // std::cout << "impl type = " << type_name<diff_t>() << std::endl;
        return (A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
    }
};

// Discretized dynamics
template<typename scalar_t>
using dsys_t = lampc::functions::RK4<sys_t<scalar_t>, scalar_t, 2, 1>;


/**
 * Discrete-time dynamics - example with custom jacobian
 */
template<typename scalar_t>
struct dsys_custom_t : public lampc::MakeDifferentiable<dsys_custom_t<scalar_t>, scalar_t, 2, 2,1>
{
    Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
    Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};

    template<typename diff_t>
    EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> 
    impl( const Eigen::Ref< const Eigen::Vector<diff_t, 2> >& x, 
          const Eigen::Ref< const Eigen::Vector<diff_t, 1> >& u) noexcept        
    {
      // std::cout << "impl type = " << type_name<diff_t>() << std::endl;
        return (A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
    }

    // Specify custom jacobian
    template<typename OutValue, typename OutJacobian>
    EIGEN_STRONG_INLINE void
    operator()(
      const Eigen::Ref< const Eigen::Vector<scalar_t, 2> >& x,
      const Eigen::Ref< const Eigen::Vector<scalar_t, 1> >& u,
      OutValue&& outvalue, OutJacobian&& outjacobian) noexcept
    {
      outvalue = impl<scalar_t>(x,u);
      outjacobian(all,seqN(0,2)) = A;
      outjacobian(all,seqN(2,1)) = B;
    }
};


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


template<typename OptProblem>
struct OCP_DoubleIntegrator
{
  using Variable = typename OptProblem::Variable;
  using scalar_t = typename OptProblem::scalar_t;

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
    opt.constraint.initialize();

    // opt.constraint(idx, x(0));
    // opt.constraint(idx, x(0));
    for(int i=0; i<N-1; i++)
    {
      x(i)(0) = i;
      opt.constraint(dsys_eq, x(i+1), x(i),u(i));
    }
    // opt.constraint(idx, x(0));
  }

  void eval_objective(OptProblem& opt)
  {
    opt.objective.initialize();

    for(int i=0; i<x.cols()-1; i++)
      opt.objective(stage_cost, x(i), xss(0), u(i), uss(0));
    opt.objective(terminal_cost, x(N-1), xss(0));
  }
};


int main()
{
  using scalar_t = double;

  // Compute the sparsity structure
  lampc::ProblemSparsity<scalar_t> sparsity; // Sparsity capture object
  OCP_DoubleIntegrator<lampc::ProblemSparsity<scalar_t>> ocp_sparsity(sparsity, 10);
  ocp_sparsity.eval_constraints(sparsity);
  // ocp_sparsity.eval_objective(sparsity);

  std::cout << "sparsity = \n" << sparsity.constraint.jacobian.get_sparsity() << std::endl;

  lampc::ProblemTape<scalar_t> tape(sparsity.generate());
  OCP_DoubleIntegrator<lampc::ProblemTape<scalar_t>> ocp_tape(tape, 10);
  ocp_tape.eval_constraints(tape);
  // ocp_tape.eval_objective(tape);

  std::cout << "jacobian copies = " << tape.constraint.jacobian.copy_segments << std::endl;

  lampc::Problem<scalar_t> prob(tape.generate());
  OCP_DoubleIntegrator<lampc::Problem<scalar_t>> ocp(prob, 10);

  auto mem = prob.make_problem_memory();

  // Finally, the call!
  const std::size_t NUM_EXP = 10000000;
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < NUM_EXP; ++i)
  {
    ocp.eval_constraints(prob);
  }
  auto end = std::chrono::steady_clock::now();

  std::cout << "Jacobian time: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count()
      << " ns" << std::endl;

  Eigen::VectorX<scalar_t> xp(2);
  Eigen::VectorX<scalar_t> x(2);
  Eigen::VectorX<scalar_t> u(1);
  xp << 1,2; x << 3,4; u << 5;
  Eigen::MatrixX<scalar_t> J(2,5);
  Eigen::VectorX<scalar_t> V(2);

  Eigen::MatrixX<scalar_t> out(20,40);

  start = std::chrono::steady_clock::now();
  scalar_t acc = 0;
  for(int i = 0; i < NUM_EXP; ++i)
  {
    for(int j=0; j<9; j++)
    {
      x(0) = i + j;
      ocp.dsys_eq(xp,x,u,V,J);
      acc += V(0);
      out({2*j,2*j+1},{2*j,2*j+1,2*j+3,2*j+4,20+j}) = J;
    }
  }
  end = std::chrono::steady_clock::now();
  std::cout << "acc = " << acc << std::endl;
  std::cout << "out = \n" << out << std::endl;

  std::cout << "Direct computation of constraints time: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)/(double)NUM_EXP).count()
      << " ns" << std::endl;


  std::cout << "g = " << mem.g.transpose() << std::endl;
  std::cout << "g_jacobian = \n" << Eigen::MatrixX<double>(mem.g_jacobian) << std::endl;

};  
