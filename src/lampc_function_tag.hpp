
// namespace lampc
// {

struct Eigen_autodiff{};

// Tag used if the user has written a custom jacobian, etc
// This triggers an _impl call into the Derived class
struct USER {};

// Tag used if we have a custom implementation not written by the user
// This triggers an _impl call in the Differentiable class
struct CUSTOM {};

/**
 * Base class of all function tags
 */
struct Tag
{
  // Overload if user or custom versions are defined
  using function = USER;
  using jacobian = Eigen_autodiff;
  using gradient = Eigen_autodiff;
  using hessian  = Eigen_autodiff;
};


/**********************************
 * Define a set of useful functions
 * 
 * (_impl are in Differentiable)
 **********************************/

/**
 * For a given function F, EQ<F> is the function eq(xp, x...) = -xp + F(x...)
 */
template<typename F>
struct EQ : Tag
{
  F f; 
  template<typename... Args>
  EQ(Args&&... args) : f(std::forward<Args>(args)...) {} // If parameters are needed
  EQ() {} // Parameter free version

  using function = CUSTOM;
  using jacobian = CUSTOM;
};


/**
 * Apply RK4 to the given ODE
 */
template<typename Ode, typename _scalar_t>
struct RK4 : Tag
{
  using scalar_t = _scalar_t;

  Ode ode;
  template<typename... Args>
  RK4(scalar_t step_size, Args&&... args) : 
        ode(std::forward<Args>(args)...),
        step_size(step_size)
     {} // If parameters are needed
  RK4(scalar_t step_size) : step_size(step_size) {} // Parameter free version

  scalar_t step_size;

  using function = CUSTOM;
};


// /**
//  * Discrete-time steady-state
//  *  f(xss,uxx) = sys(xss,uss) - xss
//  */
// template<typename Sys>
// struct DSteadyState : Tag
// {
//   Sys sys; 
//   template<typename... Args>
//   EQ(Args&&... args) : sys(std::forward<Args>(args)...) {} // If parameters are needed
//   EQ() {} // Parameter free version

//   using function = CUSTOM;
//   // using jacobian = CUSTOM;
// };


/**
 * Base class (CRTP) that provides the functions:
 * - function
 * - jacobian 
 * - hessian
 * - gradient
 * 
 * Tag-dispatching is used to determine which function_impl, jacobian_impl, etc 
 * to call in the Derived class to call the user-defined functions.
 * 
 * The types jacobian, hessian, etc in the Tag are used to dispatch
 * to specific function types to implement these functions. 
 * 
 * Eigen_autodiff is included as a differentition method by default, and others
 * can be added by the user when inheriting from Differentiable
 */
template<typename Derived>
struct Differentiable
{
  // TODO: Allow hot-swapping of special function and gradient approaches

  /**
   * Call the function
   */
  template<typename Tag, typename OutValue, typename... Args>
  EIGEN_STRONG_INLINE 
  typename std::enable_if<std::is_same<typename Tag::function, USER>::value == true, void>::type
  function(Tag&& tag, OutValue&& outvalue, Args&&... args) noexcept
  {
    using Scalar = lampc::meta::get_scalar_t<std::decay_t<Args>...>;
    static_cast<Derived*>(this)->template function_impl<Scalar>(
        std::forward<Tag>(tag),
        std::forward<OutValue>(outvalue),
        std::forward<Args>(args)...);
  }

  template<typename Tag, typename OutValue, typename... Args>
  EIGEN_STRONG_INLINE 
  typename std::enable_if<std::is_same<typename Tag::function, USER>::value == false, void>::type
  function(Tag&& tag, OutValue&& outvalue, Args&&... args) noexcept
  {
    using Scalar = lampc::meta::get_scalar_t<std::decay_t<Args>...>;
    this->template function_impl<Scalar>(
        std::forward<Tag>(tag),
        std::forward<OutValue>(outvalue),
        std::forward<Args>(args)...);
  }

  /**
   * We call the jacobian function specified in the tag.
   * 
   * If it's USER, then we cast to Derived and call the user's function.
   * Otherwise we call the type specified in the tag, under the assumption that
   * it's in the scope of this class.
   * 
   * Better would be if we had a way to bring all jacobian_impl calls into the scope 
   * of Derived, in which case we wouldn't need these enable_if's
   */

  // User-specified jacobian code
  template <typename Tag, typename... Args>
  typename std::enable_if<std::is_same<typename Tag::jacobian, USER>::value == true, void>::type
  jacobian(Tag&& tag, Args&&... args)
  {
    static_cast<Derived*>(this)->jacobian_impl(std::forward<Tag>(tag), USER{}, std::forward<Args>(args)...);
  }

  // Tag-dispatch based on Tag::jacobian
  template <typename Tag, typename... Args>
  typename std::enable_if<std::is_same<typename Tag::jacobian, USER>::value == false, void>::type
  jacobian(Tag&& tag, Args&&... args)
  {
    jacobian_impl(std::forward<Tag>(tag), typename Tag::jacobian(), std::forward<Args>(args)...);
  }


private:

  // 
  // For a given function F, EQ<F> is the function eq(xp, x...) = -xp + F(x...)
  // 

  template<typename T, typename F, typename OutValue, typename XP, typename... X>
  void function_impl(EQ<F>&& tag, OutValue&& value,
   const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>&... x) noexcept
  {
    static_cast<Derived*>(this)->function(std::forward<std::decay_t<F>>(tag.f), value, x...);
    value -= xp;
  }

  template<typename F, typename OutValue, typename OutJacobian, typename XP, typename... X>
  void jacobian_impl(EQ<F>&& tag, CUSTOM, OutValue&& value, OutJacobian&& jac,
   const Eigen::MatrixBase<XP>& xp, const Eigen::MatrixBase<X>&... x) noexcept
  {
    // Jacobian is [-I jac_f]
    constexpr int nx = XP::RowsAtCompileTime;
    auto F_jac = jac(all,seq(nx,last));
    jacobian(std::forward<std::decay_t<F>>(tag.f), value,F_jac, x...);
    value -= xp;

    using scalar_t = typename Eigen::MatrixBase<XP>::Scalar;
    for(int i=0; i<value.rows(); i++)
      jac(seqN(i,fix<1>), seqN(i, fix<1>)) = Eigen::Matrix<scalar_t,1,1>::Constant(-1);
  }


  // 
  // Apply RK4 to the given Ode tag
  // 

  template<typename T, typename Ode, typename scalar_t, typename OutValue, typename X, typename... Params>
  inline void function_impl(RK4<Ode, scalar_t>&& tag, OutValue&& xplus,
    const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
  {
    using Vec = typename X::PlainObject;
    Vec k1, k2, k3, k4;
    scalar_t h = tag.step_size;

    function(std::forward<Ode>(tag.ode), k1, x,          params...);
    function(std::forward<Ode>(tag.ode), k2, x+h*0.5*k1, params...);
    function(std::forward<Ode>(tag.ode), k3, x+h*0.5*k2, params...);
    function(std::forward<Ode>(tag.ode), k4, x+h*k3,     params...);
    xplus = x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
  }


  // // 
  // // Discrete-time steady-state
  // //  f(xss,params...) = sys(xss,params...) - xss
  // // 

  // template<typename T, typename Sys, typename OutValue, typename X, typename... Params>
  // inline void function_impl(DSteadyState<Sys>&& tag, OutValue&& val,
  //   const Eigen::MatrixBase<X>& xss, const Eigen::MatrixBase<Params>&... params) noexcept
  // {
  //   function(std::forward<Sys>(tag.sys), val, xss, params...);
  //   val -= xss;
  // }

  //
  // Eigen Autodiff Implementations
  //


  /**
   * Class that provides implementations of jacobian_impl, hessian_impl, gradient_impl
   * using the Eigen AutoDiff tool
   */
  // Compute the jacobian with eigen autodiff
  template<typename Tag, typename OutValue, typename OutJacobian, typename... Args>
  EIGEN_STRONG_INLINE void
  jacobian_impl(
     Tag&& tag, // Function to call
     Eigen_autodiff, // Type of autodiff method
     OutValue&& outvalue, OutJacobian&& outjacobian, // Outputs
     const Eigen::MatrixBase<Args>&... args) noexcept // Function arguments
  {
    // Compute the scalar type
    using Scalar = lampc::meta::get_scalar_t<Eigen::MatrixBase<Args>...>;

    // Get the total number of inputs
    constexpr size_t num_inputs = lampc::meta::sum_template<Eigen::MatrixBase<Args>::RowsAtCompileTime...>();

    // First order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<Scalar, num_inputs>>;
    using AD_Output = Eigen::Vector<AD_scalar,std::remove_reference_t<OutValue>::RowsAtCompileTime>;

    // Convert the arguments to AD variables, and call the function
    AD_Output out;
    seed_and_call(tag, out, make_ad<num_inputs>(args)...);

    // Copy out into output variables
    for(int i=0; i<out.rows(); i++)
    {
        outvalue(i) = out[i].value();
        outjacobian(i,Eigen::all) = out[i].derivatives().transpose();
    }
  }

  // Sets the input derivatives to the identity. 
  // Assumes that the derivative matrix is initially zero
  template<typename Arg>
  static EIGEN_STRONG_INLINE int 
  AD_Seed(Eigen::MatrixBase<Arg>& x, int offset)
  {
      for (int i=0; i<x.rows(); i++)
          x[i].derivatives().coeffRef(i + offset) = 1;
      return offset + x.rows();
  }

  // Take a vector input and return a AD version of the vector
  template<size_t num_inputs, typename X, 
           typename AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<typename X::Scalar, num_inputs>>>
  static EIGEN_STRONG_INLINE auto
  make_ad(const Eigen::MatrixBase<X>& x)
  {
      constexpr size_t n = X::RowsAtCompileTime;
      Eigen::Vector<AD_scalar, n> y;
      y = x;
      for (int i=0; i<y.rows(); i++) {
          y[i].derivatives().setZero();
      }
      return y;
  }

  template<typename Tag, typename Output, typename... Args>
  EIGEN_STRONG_INLINE void
  seed_and_call(Tag&& tag, Output& out, Eigen::MatrixBase<Args>&&... args)
  {
    // Set derivative equal to identity
    int offset = 0;
    (void)std::initializer_list<int>{ 
      (
        offset = AD_Seed(args, offset), // Set to unit vectors
        0
      )...
    };

    function(std::forward<std::decay_t<Tag>>(tag), out, std::forward<Args>(args)...);
  }


};

// }; // namespace lampc