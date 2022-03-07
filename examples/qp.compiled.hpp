struct QP
{
  static constexpr int num_variables = 32;
  using param_t = MyFunctions<double>::param_t_;
  using scalar_t = double;
  using variable_t = Eigen::Vector<scalar_t, num_variables>;
  
  // Variable accessors
  static Eigen::Ref<Eigen::Vector<scalar_t, 2>> x(Eigen::Ref<variable_t> var, int ind) {return var.template segment<2>(0+2*ind);};
  static Eigen::Ref<Eigen::Matrix<scalar_t, 2, 10>> x(Eigen::Ref<variable_t> var) {return Eigen::Map<Eigen::Matrix<scalar_t, 2, 10>>(var.template segment<20>(0).data());};
  static Eigen::Ref<Eigen::Vector<scalar_t, 1>> u(Eigen::Ref<variable_t> var, int ind) {return var.template segment<1>(20+1*ind);};
  static Eigen::Ref<Eigen::Matrix<scalar_t, 1, 9>> u(Eigen::Ref<variable_t> var) {return Eigen::Map<Eigen::Matrix<scalar_t, 1, 9>>(var.template segment<9>(20).data());};
  static Eigen::Ref<Eigen::Vector<scalar_t, 2>> xss(Eigen::Ref<variable_t> var) {return var.template segment<2>(29);};
  static Eigen::Ref<Eigen::Vector<scalar_t, 1>> uss(Eigen::Ref<variable_t> var) {return var.template segment<1>(31);};
  
  // Define convenience names for all differentiable functions
  using dynamics = lampc::Jacobian<MyFunctions<double>::dynamics_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using dynamics_eq = lampc::Jacobian<MyFunctions<double>::dynamics_eq_, double, MyFunctions<double>::param_t_, 2, 2, 2, 1>;
  using dynamics_ss = lampc::Jacobian<MyFunctions<double>::dynamics_ss_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using stage_cost = lampc::Jacobian<MyFunctions<double>::stage_cost_, double, MyFunctions<double>::param_t_, 1, 2, 1, 2, 1>;
  
  struct equalities : public function_util_t<scalar_t, Eigen::Vector<scalar_t, 22>, Eigen::SparseMatrix<scalar_t>>
  
  {
    using out_t = Eigen::Vector<scalar_t, 22>;
    using jacobian_t = Eigen::SparseMatrix<scalar_t>;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;
    
    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out)
    {
      out.SEG(2,0) = dynamics::eval(param, x.SEG(2,2), x.SEG(1,20)); // dynamics(x_1, u_0)
      out.SEG(2,2) = dynamics_eq::eval(param, x.SEG(2,2), x.SEG(2,0), x.SEG(1,20)); // dynamics_eq(x_1, x_0, u_0)
      out.SEG(2,4) = dynamics_eq::eval(param, x.SEG(2,4), x.SEG(2,2), x.SEG(1,21)); // dynamics_eq(x_2, x_1, u_1)
      out.SEG(2,6) = dynamics_eq::eval(param, x.SEG(2,6), x.SEG(2,4), x.SEG(1,22)); // dynamics_eq(x_3, x_2, u_2)
      out.SEG(2,8) = dynamics_eq::eval(param, x.SEG(2,8), x.SEG(2,6), x.SEG(1,23)); // dynamics_eq(x_4, x_3, u_3)
      out.SEG(2,10) = dynamics_eq::eval(param, x.SEG(2,10), x.SEG(2,8), x.SEG(1,24)); // dynamics_eq(x_5, x_4, u_4)
      out.SEG(2,12) = dynamics_eq::eval(param, x.SEG(2,12), x.SEG(2,10), x.SEG(1,25)); // dynamics_eq(x_6, x_5, u_5)
      out.SEG(2,14) = dynamics_eq::eval(param, x.SEG(2,14), x.SEG(2,12), x.SEG(1,26)); // dynamics_eq(x_7, x_6, u_6)
      out.SEG(2,16) = dynamics_eq::eval(param, x.SEG(2,16), x.SEG(2,14), x.SEG(1,27)); // dynamics_eq(x_8, x_7, u_7)
      out.SEG(2,18) = dynamics_eq::eval(param, x.SEG(2,18), x.SEG(2,16), x.SEG(1,28)); // dynamics_eq(x_9, x_8, u_8)
      out.SEG(2,20) = dynamics_ss::eval(param, x.SEG(2,29), x.SEG(1,31)); // dynamics_ss(xss, uss)
    };
    
    static void initialize_jacobian(Eigen::SparseMatrix<scalar_t> &J)
    {
      J.resize(22,32);
      J.reserve(102);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,102> tripletList = {T{2,0,1},{3,0,1},{2,1,1},{3,1,1},{0,2,1},{1,2,1},{2,2,1},{3,2,1},{4,2,1},{5,2,1},{0,3,1},{1,3,1},{2,3,1},{3,3,1},{4,3,1},{5,3,1},{4,4,1},{5,4,1},{6,4,1},{7,4,1},{4,5,1},{5,5,1},{6,5,1},{7,5,1},{6,6,1},{7,6,1},{8,6,1},{9,6,1},{6,7,1},{7,7,1},{8,7,1},{9,7,1},{8,8,1},{9,8,1},{10,8,1},{11,8,1},{8,9,1},{9,9,1},{10,9,1},{11,9,1},{10,10,1},{11,10,1},{12,10,1},{13,10,1},{10,11,1},{11,11,1},{12,11,1},{13,11,1},{12,12,1},{13,12,1},{14,12,1},{15,12,1},{12,13,1},{13,13,1},{14,13,1},{15,13,1},{14,14,1},{15,14,1},{16,14,1},{17,14,1},{14,15,1},{15,15,1},{16,15,1},{17,15,1},{16,16,1},{17,16,1},{18,16,1},{19,16,1},{16,17,1},{17,17,1},{18,17,1},{19,17,1},{18,18,1},{19,18,1},{18,19,1},{19,19,1},{0,20,1},{1,20,1},{2,20,1},{3,20,1},{4,21,1},{5,21,1},{6,22,1},{7,22,1},{8,23,1},{9,23,1},{10,24,1},{11,24,1},{12,25,1},{13,25,1},{14,26,1},{15,26,1},{16,27,1},{17,27,1},{18,28,1},{19,28,1},{20,29,1},{21,29,1},{20,30,1},{21,30,1},{20,31,1},{21,31,1}};
      J.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    
    /** 
     * Compute the jacobian of the overall function
     */
    static constexpr seqinfo jac_seq[47] = {{4,2},{10,2},{76,2},{6,2},{12,2},{0,4},{78,2},{16,2},{20,2},{8,2},{14,2},{80,2},{24,2},{28,2},{18,2},{22,2},{82,2},{32,2},{36,2},{26,2},{30,2},{84,2},{40,2},{44,2},{34,2},{38,2},{86,2},{48,2},{52,2},{42,2},{46,2},{88,2},{56,2},{60,2},{50,2},{54,2},{90,2},{64,2},{68,2},{58,2},{62,2},{92,2},{72,4},{66,2},{70,2},{94,2},{96,6}};
    static void eval(param_t &param, variable_t x, out_t &out, jacobian_t &jacobian)
    {
    };
    
    
  };
  
  
  struct objective : public weightedsum_util_t<scalar_t, Eigen::Vector<scalar_t, num_variables>, Eigen::Vector<scalar_t, 9>>
  {
    using weight_t = Eigen::Vector<scalar_t, 9>;
    using gradient_t = Eigen::Vector<scalar_t, num_variables>;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;
    
    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> &w, const Eigen::Ref<const variable_t> &x)
    {
      scalar_t val = 0;
      val += w.SEG(1,0).dot(stage_cost::eval(param, x.SEG(2,0), x.SEG(1,20), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_0, u_0, xss, uss)
      val += w.SEG(1,1).dot(stage_cost::eval(param, x.SEG(2,2), x.SEG(1,21), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_1, u_1, xss, uss)
      val += w.SEG(1,2).dot(stage_cost::eval(param, x.SEG(2,4), x.SEG(1,22), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_2, u_2, xss, uss)
      val += w.SEG(1,3).dot(stage_cost::eval(param, x.SEG(2,6), x.SEG(1,23), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_3, u_3, xss, uss)
      val += w.SEG(1,4).dot(stage_cost::eval(param, x.SEG(2,8), x.SEG(1,24), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_4, u_4, xss, uss)
      val += w.SEG(1,5).dot(stage_cost::eval(param, x.SEG(2,10), x.SEG(1,25), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_5, u_5, xss, uss)
      val += w.SEG(1,6).dot(stage_cost::eval(param, x.SEG(2,12), x.SEG(1,26), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_6, u_6, xss, uss)
      val += w.SEG(1,7).dot(stage_cost::eval(param, x.SEG(2,14), x.SEG(1,27), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_7, u_7, xss, uss)
      val += w.SEG(1,8).dot(stage_cost::eval(param, x.SEG(2,16), x.SEG(1,28), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x_8, u_8, xss, uss)
      return val;
    };
    
    /**
     * Initialize the hessian of the function
     */
    static void initialize_hessian(Eigen::SparseMatrix<scalar_t> &H)
    {
      H.resize(32,32);
      H.reserve(252);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,252> tripletList = {T{0,0,1},{1,0,1},{20,0,1},{29,0,1},{30,0,1},{31,0,1},{0,1,1},{1,1,1},{20,1,1},{29,1,1},{30,1,1},{31,1,1},{2,2,1},{3,2,1},{21,2,1},{29,2,1},{30,2,1},{31,2,1},{2,3,1},{3,3,1},{21,3,1},{29,3,1},{30,3,1},{31,3,1},{4,4,1},{5,4,1},{22,4,1},{29,4,1},{30,4,1},{31,4,1},{4,5,1},{5,5,1},{22,5,1},{29,5,1},{30,5,1},{31,5,1},{6,6,1},{7,6,1},{23,6,1},{29,6,1},{30,6,1},{31,6,1},{6,7,1},{7,7,1},{23,7,1},{29,7,1},{30,7,1},{31,7,1},{8,8,1},{9,8,1},{24,8,1},{29,8,1},{30,8,1},{31,8,1},{8,9,1},{9,9,1},{24,9,1},{29,9,1},{30,9,1},{31,9,1},{10,10,1},{11,10,1},{25,10,1},{29,10,1},{30,10,1},{31,10,1},{10,11,1},{11,11,1},{25,11,1},{29,11,1},{30,11,1},{31,11,1},{12,12,1},{13,12,1},{26,12,1},{29,12,1},{30,12,1},{31,12,1},{12,13,1},{13,13,1},{26,13,1},{29,13,1},{30,13,1},{31,13,1},{14,14,1},{15,14,1},{27,14,1},{29,14,1},{30,14,1},{31,14,1},{14,15,1},{15,15,1},{27,15,1},{29,15,1},{30,15,1},{31,15,1},{16,16,1},{17,16,1},{28,16,1},{29,16,1},{30,16,1},{31,16,1},{16,17,1},{17,17,1},{28,17,1},{29,17,1},{30,17,1},{31,17,1},{0,20,1},{1,20,1},{20,20,1},{29,20,1},{30,20,1},{31,20,1},{2,21,1},{3,21,1},{21,21,1},{29,21,1},{30,21,1},{31,21,1},{4,22,1},{5,22,1},{22,22,1},{29,22,1},{30,22,1},{31,22,1},{6,23,1},{7,23,1},{23,23,1},{29,23,1},{30,23,1},{31,23,1},{8,24,1},{9,24,1},{24,24,1},{29,24,1},{30,24,1},{31,24,1},{10,25,1},{11,25,1},{25,25,1},{29,25,1},{30,25,1},{31,25,1},{12,26,1},{13,26,1},{26,26,1},{29,26,1},{30,26,1},{31,26,1},{14,27,1},{15,27,1},{27,27,1},{29,27,1},{30,27,1},{31,27,1},{16,28,1},{17,28,1},{28,28,1},{29,28,1},{30,28,1},{31,28,1},{0,29,1},{1,29,1},{2,29,1},{3,29,1},{4,29,1},{5,29,1},{6,29,1},{7,29,1},{8,29,1},{9,29,1},{10,29,1},{11,29,1},{12,29,1},{13,29,1},{14,29,1},{15,29,1},{16,29,1},{17,29,1},{20,29,1},{21,29,1},{22,29,1},{23,29,1},{24,29,1},{25,29,1},{26,29,1},{27,29,1},{28,29,1},{29,29,1},{30,29,1},{31,29,1},{0,30,1},{1,30,1},{2,30,1},{3,30,1},{4,30,1},{5,30,1},{6,30,1},{7,30,1},{8,30,1},{9,30,1},{10,30,1},{11,30,1},{12,30,1},{13,30,1},{14,30,1},{15,30,1},{16,30,1},{17,30,1},{20,30,1},{21,30,1},{22,30,1},{23,30,1},{24,30,1},{25,30,1},{26,30,1},{27,30,1},{28,30,1},{29,30,1},{30,30,1},{31,30,1},{0,31,1},{1,31,1},{2,31,1},{3,31,1},{4,31,1},{5,31,1},{6,31,1},{7,31,1},{8,31,1},{9,31,1},{10,31,1},{11,31,1},{12,31,1},{13,31,1},{14,31,1},{15,31,1},{16,31,1},{17,31,1},{20,31,1},{21,31,1},{22,31,1},{23,31,1},{24,31,1},{25,31,1},{26,31,1},{27,31,1},{28,31,1},{29,31,1},{30,31,1},{31,31,1}};
      H.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    
    /**
     * Copy the hessian of <w, f> into the right place
     * 
     * Input:
     *   hessian_return_t (value, jacobian and hessian of the vector-valued function f)
     * 
     * Output:
     *   gradient += w' * jacobian f(x) 
     *   value += w' * f(x)
     *   hessian += sum wi * hessian fi(x)
     */
    
  };
  
};
