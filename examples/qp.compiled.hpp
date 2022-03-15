#ifndef __QP_HPP
#define __QP_HPP
struct QP
{
  static constexpr int num_variables = 32;
  using param_t = MyFunctions<double>::param_t;
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
  using dynamics_0 = lampc::Jacobian<MyFunctions<double>::dynamics_0_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using dynamics_eq = lampc::Jacobian<MyFunctions<double>::dynamics_eq_, double, MyFunctions<double>::param_t_, 2, 2, 2, 1>;
  using dynamics_ss = lampc::Jacobian<MyFunctions<double>::dynamics_ss_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using output = lampc::Jacobian<MyFunctions<double>::output_, double, MyFunctions<double>::param_t_, 1, 2>;
  using stage_cost = lampc::Jacobian<MyFunctions<double>::stage_cost_, double, MyFunctions<double>::param_t_, 1, 2, 1, 2, 1>;
  using terminal_cost = lampc::Jacobian<MyFunctions<double>::terminal_cost_, double, MyFunctions<double>::param_t_, 1, 2, 2>;
  
  struct constraints : public function_util_t<scalar_t, Eigen::Vector<scalar_t, 32>, Eigen::SparseMatrix<scalar_t>>
  
  {
    using out_t = Eigen::Vector<scalar_t, 32>;
    using jacobian_t = Eigen::SparseMatrix<scalar_t>;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;
    
    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out)
    {
      out.SEG(2,0) = dynamics_0::eval(param, x.SEG(2,2), x.SEG(1,20)); // dynamics_0(x1, u0)
      out.SEG(2,2) = dynamics_eq::eval(param, x.SEG(2,2), x.SEG(2,0), x.SEG(1,20)); // dynamics_eq(x1, x0, u0)
      out.SEG(1,4) = output::eval(param, x.SEG(2,0)); // output(x0)
      out.SEG(2,5) = dynamics_eq::eval(param, x.SEG(2,4), x.SEG(2,2), x.SEG(1,21)); // dynamics_eq(x2, x1, u1)
      out.SEG(1,7) = output::eval(param, x.SEG(2,2)); // output(x1)
      out.SEG(2,8) = dynamics_eq::eval(param, x.SEG(2,6), x.SEG(2,4), x.SEG(1,22)); // dynamics_eq(x3, x2, u2)
      out.SEG(1,10) = output::eval(param, x.SEG(2,4)); // output(x2)
      out.SEG(2,11) = dynamics_eq::eval(param, x.SEG(2,8), x.SEG(2,6), x.SEG(1,23)); // dynamics_eq(x4, x3, u3)
      out.SEG(1,13) = output::eval(param, x.SEG(2,6)); // output(x3)
      out.SEG(2,14) = dynamics_eq::eval(param, x.SEG(2,10), x.SEG(2,8), x.SEG(1,24)); // dynamics_eq(x5, x4, u4)
      out.SEG(1,16) = output::eval(param, x.SEG(2,8)); // output(x4)
      out.SEG(2,17) = dynamics_eq::eval(param, x.SEG(2,12), x.SEG(2,10), x.SEG(1,25)); // dynamics_eq(x6, x5, u5)
      out.SEG(1,19) = output::eval(param, x.SEG(2,10)); // output(x5)
      out.SEG(2,20) = dynamics_eq::eval(param, x.SEG(2,14), x.SEG(2,12), x.SEG(1,26)); // dynamics_eq(x7, x6, u6)
      out.SEG(1,22) = output::eval(param, x.SEG(2,12)); // output(x6)
      out.SEG(2,23) = dynamics_eq::eval(param, x.SEG(2,16), x.SEG(2,14), x.SEG(1,27)); // dynamics_eq(x8, x7, u7)
      out.SEG(1,25) = output::eval(param, x.SEG(2,14)); // output(x7)
      out.SEG(2,26) = dynamics_eq::eval(param, x.SEG(2,18), x.SEG(2,16), x.SEG(1,28)); // dynamics_eq(x9, x8, u8)
      out.SEG(1,28) = output::eval(param, x.SEG(2,16)); // output(x8)
      out.SEG(2,29) = dynamics_ss::eval(param, x.SEG(2,29), x.SEG(1,31)); // dynamics_ss(xss, uss)
      out.SEG(1,31) = output::eval(param, x.SEG(2,18)); // output(x9)
    };
    
    static void initialize_jacobian(Eigen::SparseMatrix<scalar_t> &J)
    {
      J.resize(32,32);
      J.reserve(122);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,122> tripletList = {T{2,0,1},{3,0,1},{4,0,1},{2,1,1},{3,1,1},{4,1,1},{0,2,1},{1,2,1},{2,2,1},{3,2,1},{5,2,1},{6,2,1},{7,2,1},{0,3,1},{1,3,1},{2,3,1},{3,3,1},{5,3,1},{6,3,1},{7,3,1},{5,4,1},{6,4,1},{8,4,1},{9,4,1},{10,4,1},{5,5,1},{6,5,1},{8,5,1},{9,5,1},{10,5,1},{8,6,1},{9,6,1},{11,6,1},{12,6,1},{13,6,1},{8,7,1},{9,7,1},{11,7,1},{12,7,1},{13,7,1},{11,8,1},{12,8,1},{14,8,1},{15,8,1},{16,8,1},{11,9,1},{12,9,1},{14,9,1},{15,9,1},{16,9,1},{14,10,1},{15,10,1},{17,10,1},{18,10,1},{19,10,1},{14,11,1},{15,11,1},{17,11,1},{18,11,1},{19,11,1},{17,12,1},{18,12,1},{20,12,1},{21,12,1},{22,12,1},{17,13,1},{18,13,1},{20,13,1},{21,13,1},{22,13,1},{20,14,1},{21,14,1},{23,14,1},{24,14,1},{25,14,1},{20,15,1},{21,15,1},{23,15,1},{24,15,1},{25,15,1},{23,16,1},{24,16,1},{26,16,1},{27,16,1},{28,16,1},{23,17,1},{24,17,1},{26,17,1},{27,17,1},{28,17,1},{26,18,1},{27,18,1},{31,18,1},{26,19,1},{27,19,1},{31,19,1},{0,20,1},{1,20,1},{2,20,1},{3,20,1},{5,21,1},{6,21,1},{8,22,1},{9,22,1},{11,23,1},{12,23,1},{14,24,1},{15,24,1},{17,25,1},{18,25,1},{20,26,1},{21,26,1},{23,27,1},{24,27,1},{26,28,1},{27,28,1},{29,29,1},{30,29,1},{29,30,1},{30,30,1},{29,31,1},{30,31,1}};
      J.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    
    /** 
     * Compute the jacobian of the overall function
     */
    static constexpr seqinfo jac_seq[69] = {
      {6,2},{13,2},{96,2},
      {8,2},{15,2},{0,2},{3,2},{98,2},
      {2,1},{5,1},
      {20,2},{25,2},{10,2},{17,2},{100,2},
      {12,1},{19,1},
      {30,2},{35,2},{22,2},{27,2},{102,2},
      {24,1},{29,1},
      {40,2},{45,2},{32,2},{37,2},{104,2},
      {34,1},{39,1},
      {50,2},{55,2},{42,2},{47,2},{106,2},
      {44,1},{49,1},
      {60,2},{65,2},{52,2},{57,2},{108,2},
      {54,1},{59,1},
      {70,2},{75,2},{62,2},{67,2},{110,2},
      {64,1},{69,1},
      {80,2},{85,2},{72,2},{77,2},{112,2},
      {74,1},{79,1},
      {90,2},{93,2},{82,2},{87,2},{114,2},
      {84,1},{89,1},
      {116,6},
      {92,1},{95,1}};
    static void eval(param_t &param, variable_t x, out_t &out, jacobian_t &jacobian)
    {
      setJ(out, jacobian, 0, jac_seq+0, 3, dynamics_0::jac(param, x.SEG(2,2), x.SEG(1,20))); // dynamics_0(x1, u0)
      setJ(out, jacobian, 2, jac_seq+3, 5, dynamics_eq::jac(param, x.SEG(2,2), x.SEG(2,0), x.SEG(1,20))); // dynamics_eq(x1, x0, u0)
      setJ(out, jacobian, 4, jac_seq+8, 2, output::jac(param, x.SEG(2,0))); // output(x0)
      setJ(out, jacobian, 5, jac_seq+10, 5, dynamics_eq::jac(param, x.SEG(2,4), x.SEG(2,2), x.SEG(1,21))); // dynamics_eq(x2, x1, u1)
      setJ(out, jacobian, 7, jac_seq+15, 2, output::jac(param, x.SEG(2,2))); // output(x1)
      setJ(out, jacobian, 8, jac_seq+17, 5, dynamics_eq::jac(param, x.SEG(2,6), x.SEG(2,4), x.SEG(1,22))); // dynamics_eq(x3, x2, u2)
      setJ(out, jacobian, 10, jac_seq+22, 2, output::jac(param, x.SEG(2,4))); // output(x2)
      setJ(out, jacobian, 11, jac_seq+24, 5, dynamics_eq::jac(param, x.SEG(2,8), x.SEG(2,6), x.SEG(1,23))); // dynamics_eq(x4, x3, u3)
      setJ(out, jacobian, 13, jac_seq+29, 2, output::jac(param, x.SEG(2,6))); // output(x3)
      setJ(out, jacobian, 14, jac_seq+31, 5, dynamics_eq::jac(param, x.SEG(2,10), x.SEG(2,8), x.SEG(1,24))); // dynamics_eq(x5, x4, u4)
      setJ(out, jacobian, 16, jac_seq+36, 2, output::jac(param, x.SEG(2,8))); // output(x4)
      setJ(out, jacobian, 17, jac_seq+38, 5, dynamics_eq::jac(param, x.SEG(2,12), x.SEG(2,10), x.SEG(1,25))); // dynamics_eq(x6, x5, u5)
      setJ(out, jacobian, 19, jac_seq+43, 2, output::jac(param, x.SEG(2,10))); // output(x5)
      setJ(out, jacobian, 20, jac_seq+45, 5, dynamics_eq::jac(param, x.SEG(2,14), x.SEG(2,12), x.SEG(1,26))); // dynamics_eq(x7, x6, u6)
      setJ(out, jacobian, 22, jac_seq+50, 2, output::jac(param, x.SEG(2,12))); // output(x6)
      setJ(out, jacobian, 23, jac_seq+52, 5, dynamics_eq::jac(param, x.SEG(2,16), x.SEG(2,14), x.SEG(1,27))); // dynamics_eq(x8, x7, u7)
      setJ(out, jacobian, 25, jac_seq+57, 2, output::jac(param, x.SEG(2,14))); // output(x7)
      setJ(out, jacobian, 26, jac_seq+59, 5, dynamics_eq::jac(param, x.SEG(2,18), x.SEG(2,16), x.SEG(1,28))); // dynamics_eq(x9, x8, u8)
      setJ(out, jacobian, 28, jac_seq+64, 2, output::jac(param, x.SEG(2,16))); // output(x8)
      setJ(out, jacobian, 29, jac_seq+66, 1, dynamics_ss::jac(param, x.SEG(2,29), x.SEG(1,31))); // dynamics_ss(xss, uss)
      setJ(out, jacobian, 31, jac_seq+67, 2, output::jac(param, x.SEG(2,18))); // output(x9)
    };
    
    
    static void bounds(Eigen::Ref<out_t> lb, Eigen::Ref<out_t> ub)
    {
      constexpr scalar_t inf = std::numeric_limits<double>::infinity();
      lb << 0,0,3,3,-1,3,3,-1,3,3,-1,3,3,-1,3,3,-1,3,3,-1,3,3,-1,3,3,-1,3,3,-1,0,0,-2;
      ub << 0,0,3,3,1,3,3,1,3,3,1,3,3,1,3,3,1,3,3,1,3,3,1,3,3,1,3,3,1,0,0,2;
    }
  };
  
  
  struct objective : public weightedsum_util_t<scalar_t, Eigen::Vector<scalar_t, num_variables>, Eigen::Vector<scalar_t, 10>>
  {
    using weight_t = Eigen::Vector<scalar_t, 10>;
    using gradient_t = Eigen::Vector<scalar_t, num_variables>;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;
    
    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> &w, const Eigen::Ref<const variable_t> &x)
    {
      scalar_t val = 0;
      val += w.SEG(1,0).dot(stage_cost::eval(param, x.SEG(2,0), x.SEG(1,20), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x0, u0, xss, uss)
      val += w.SEG(1,1).dot(stage_cost::eval(param, x.SEG(2,2), x.SEG(1,21), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x1, u1, xss, uss)
      val += w.SEG(1,2).dot(stage_cost::eval(param, x.SEG(2,4), x.SEG(1,22), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x2, u2, xss, uss)
      val += w.SEG(1,3).dot(stage_cost::eval(param, x.SEG(2,6), x.SEG(1,23), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x3, u3, xss, uss)
      val += w.SEG(1,4).dot(stage_cost::eval(param, x.SEG(2,8), x.SEG(1,24), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x4, u4, xss, uss)
      val += w.SEG(1,5).dot(stage_cost::eval(param, x.SEG(2,10), x.SEG(1,25), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x5, u5, xss, uss)
      val += w.SEG(1,6).dot(stage_cost::eval(param, x.SEG(2,12), x.SEG(1,26), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x6, u6, xss, uss)
      val += w.SEG(1,7).dot(stage_cost::eval(param, x.SEG(2,14), x.SEG(1,27), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x7, u7, xss, uss)
      val += w.SEG(1,8).dot(stage_cost::eval(param, x.SEG(2,16), x.SEG(1,28), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x8, u8, xss, uss)
      val += w.SEG(1,9).dot(terminal_cost::eval(param, x.SEG(2,18), x.SEG(2,29))); // terminal_cost(x9, xss)
      return val;
    };
    
    /** 
     * Compute the gradient of the weighted sum
     */
    static constexpr seqinfo grad_seq[38] = {
      {0,2},{20,1},{29,2},{31,1},
      {2,2},{21,1},{29,2},{31,1},
      {4,2},{22,1},{29,2},{31,1},
      {6,2},{23,1},{29,2},{31,1},
      {8,2},{24,1},{29,2},{31,1},
      {10,2},{25,1},{29,2},{31,1},
      {12,2},{26,1},{29,2},{31,1},
      {14,2},{27,1},{29,2},{31,1},
      {16,2},{28,1},{29,2},{31,1},
      {18,2},{29,2}};
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient)
    {
      gradient.array() = 0;
      scalar_t val = 0;
      accGrad(val, gradient, grad_seq+0, 4, w.SEG(1,0), stage_cost::jac(param, x.SEG(2,0), x.SEG(1,20), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x0, u0, xss, uss)
      accGrad(val, gradient, grad_seq+4, 4, w.SEG(1,1), stage_cost::jac(param, x.SEG(2,2), x.SEG(1,21), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x1, u1, xss, uss)
      accGrad(val, gradient, grad_seq+8, 4, w.SEG(1,2), stage_cost::jac(param, x.SEG(2,4), x.SEG(1,22), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x2, u2, xss, uss)
      accGrad(val, gradient, grad_seq+12, 4, w.SEG(1,3), stage_cost::jac(param, x.SEG(2,6), x.SEG(1,23), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x3, u3, xss, uss)
      accGrad(val, gradient, grad_seq+16, 4, w.SEG(1,4), stage_cost::jac(param, x.SEG(2,8), x.SEG(1,24), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x4, u4, xss, uss)
      accGrad(val, gradient, grad_seq+20, 4, w.SEG(1,5), stage_cost::jac(param, x.SEG(2,10), x.SEG(1,25), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x5, u5, xss, uss)
      accGrad(val, gradient, grad_seq+24, 4, w.SEG(1,6), stage_cost::jac(param, x.SEG(2,12), x.SEG(1,26), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x6, u6, xss, uss)
      accGrad(val, gradient, grad_seq+28, 4, w.SEG(1,7), stage_cost::jac(param, x.SEG(2,14), x.SEG(1,27), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x7, u7, xss, uss)
      accGrad(val, gradient, grad_seq+32, 4, w.SEG(1,8), stage_cost::jac(param, x.SEG(2,16), x.SEG(1,28), x.SEG(2,29), x.SEG(1,31))); // stage_cost(x8, u8, xss, uss)
      accGrad(val, gradient, grad_seq+36, 2, w.SEG(1,9), terminal_cost::jac(param, x.SEG(2,18), x.SEG(2,29))); // terminal_cost(x9, xss)
      return val;
    };
    
    
    /**
     * Initialize the hessian of the function
     */
    static void initialize_hessian(Eigen::SparseMatrix<scalar_t> &H)
    {
      H.resize(32,32);
      H.reserve(264);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,264> tripletList = {T{0,0,1},{1,0,1},{20,0,1},{29,0,1},{30,0,1},{31,0,1},{0,1,1},{1,1,1},{20,1,1},{29,1,1},{30,1,1},{31,1,1},{2,2,1},{3,2,1},{21,2,1},{29,2,1},{30,2,1},{31,2,1},{2,3,1},{3,3,1},{21,3,1},{29,3,1},{30,3,1},{31,3,1},{4,4,1},{5,4,1},{22,4,1},{29,4,1},{30,4,1},{31,4,1},{4,5,1},{5,5,1},{22,5,1},{29,5,1},{30,5,1},{31,5,1},{6,6,1},{7,6,1},{23,6,1},{29,6,1},{30,6,1},{31,6,1},{6,7,1},{7,7,1},{23,7,1},{29,7,1},{30,7,1},{31,7,1},{8,8,1},{9,8,1},{24,8,1},{29,8,1},{30,8,1},{31,8,1},{8,9,1},{9,9,1},{24,9,1},{29,9,1},{30,9,1},{31,9,1},{10,10,1},{11,10,1},{25,10,1},{29,10,1},{30,10,1},{31,10,1},{10,11,1},{11,11,1},{25,11,1},{29,11,1},{30,11,1},{31,11,1},{12,12,1},{13,12,1},{26,12,1},{29,12,1},{30,12,1},{31,12,1},{12,13,1},{13,13,1},{26,13,1},{29,13,1},{30,13,1},{31,13,1},{14,14,1},{15,14,1},{27,14,1},{29,14,1},{30,14,1},{31,14,1},{14,15,1},{15,15,1},{27,15,1},{29,15,1},{30,15,1},{31,15,1},{16,16,1},{17,16,1},{28,16,1},{29,16,1},{30,16,1},{31,16,1},{16,17,1},{17,17,1},{28,17,1},{29,17,1},{30,17,1},{31,17,1},{18,18,1},{19,18,1},{29,18,1},{30,18,1},{18,19,1},{19,19,1},{29,19,1},{30,19,1},{0,20,1},{1,20,1},{20,20,1},{29,20,1},{30,20,1},{31,20,1},{2,21,1},{3,21,1},{21,21,1},{29,21,1},{30,21,1},{31,21,1},{4,22,1},{5,22,1},{22,22,1},{29,22,1},{30,22,1},{31,22,1},{6,23,1},{7,23,1},{23,23,1},{29,23,1},{30,23,1},{31,23,1},{8,24,1},{9,24,1},{24,24,1},{29,24,1},{30,24,1},{31,24,1},{10,25,1},{11,25,1},{25,25,1},{29,25,1},{30,25,1},{31,25,1},{12,26,1},{13,26,1},{26,26,1},{29,26,1},{30,26,1},{31,26,1},{14,27,1},{15,27,1},{27,27,1},{29,27,1},{30,27,1},{31,27,1},{16,28,1},{17,28,1},{28,28,1},{29,28,1},{30,28,1},{31,28,1},{0,29,1},{1,29,1},{2,29,1},{3,29,1},{4,29,1},{5,29,1},{6,29,1},{7,29,1},{8,29,1},{9,29,1},{10,29,1},{11,29,1},{12,29,1},{13,29,1},{14,29,1},{15,29,1},{16,29,1},{17,29,1},{18,29,1},{19,29,1},{20,29,1},{21,29,1},{22,29,1},{23,29,1},{24,29,1},{25,29,1},{26,29,1},{27,29,1},{28,29,1},{29,29,1},{30,29,1},{31,29,1},{0,30,1},{1,30,1},{2,30,1},{3,30,1},{4,30,1},{5,30,1},{6,30,1},{7,30,1},{8,30,1},{9,30,1},{10,30,1},{11,30,1},{12,30,1},{13,30,1},{14,30,1},{15,30,1},{16,30,1},{17,30,1},{18,30,1},{19,30,1},{20,30,1},{21,30,1},{22,30,1},{23,30,1},{24,30,1},{25,30,1},{26,30,1},{27,30,1},{28,30,1},{29,30,1},{30,30,1},{31,30,1},{0,31,1},{1,31,1},{2,31,1},{3,31,1},{4,31,1},{5,31,1},{6,31,1},{7,31,1},{8,31,1},{9,31,1},{10,31,1},{11,31,1},{12,31,1},{13,31,1},{14,31,1},{15,31,1},{16,31,1},{17,31,1},{20,31,1},{21,31,1},{22,31,1},{23,31,1},{24,31,1},{25,31,1},{26,31,1},{27,31,1},{28,31,1},{29,31,1},{30,31,1},{31,31,1}};
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
    static constexpr seqinfo hessian_seq[99] = {
      {0,12},{116,6},{170,2},{190,1},{199,5},{222,1},{231,5},{252,1},{261,3},
      {12,12},{122,6},{172,2},{191,1},{199,3},{204,2},{223,1},{231,3},{236,2},{253,1},{261,3},
      {24,12},{128,6},{174,2},{192,1},{199,3},{206,2},{224,1},{231,3},{238,2},{254,1},{261,3},
      {36,12},{134,6},{176,2},{193,1},{199,3},{208,2},{225,1},{231,3},{240,2},{255,1},{261,3},
      {48,12},{140,6},{178,2},{194,1},{199,3},{210,2},{226,1},{231,3},{242,2},{256,1},{261,3},
      {60,12},{146,6},{180,2},{195,1},{199,3},{212,2},{227,1},{231,3},{244,2},{257,1},{261,3},
      {72,12},{152,6},{182,2},{196,1},{199,3},{214,2},{228,1},{231,3},{246,2},{258,1},{261,3},
      {84,12},{158,6},{184,2},{197,1},{199,3},{216,2},{229,1},{231,3},{248,2},{259,1},{261,3},
      {96,12},{164,6},{186,2},{198,4},{218,2},{230,4},{250,2},{260,4},
      {108,8},{188,2},{199,2},{220,2},{231,2}};
    static constexpr int hessian_seq_len[10] = {9,11,11,11,11,11,11,11,8,5};
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient, Eigen::Ref<hessian_t> hessian)
    {
      gradient.array() = 0;
      scalar_t val = 0;
      auto ptr = hessian.valuePtr();
      for(int i=0; i<hessian.nonZeros(); i++) ptr[i] = 0;
      accHessian(val, gradient, hessian, grad_seq+0, 4, hessian_seq+0, hessian_seq_len+0, w.SEG(1,0), stage_cost::hessian(param, x.SEG(2,0), x.SEG(1,20), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+4, 4, hessian_seq+9, hessian_seq_len+1, w.SEG(1,1), stage_cost::hessian(param, x.SEG(2,2), x.SEG(1,21), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+8, 4, hessian_seq+20, hessian_seq_len+2, w.SEG(1,2), stage_cost::hessian(param, x.SEG(2,4), x.SEG(1,22), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+12, 4, hessian_seq+31, hessian_seq_len+3, w.SEG(1,3), stage_cost::hessian(param, x.SEG(2,6), x.SEG(1,23), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+16, 4, hessian_seq+42, hessian_seq_len+4, w.SEG(1,4), stage_cost::hessian(param, x.SEG(2,8), x.SEG(1,24), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+20, 4, hessian_seq+53, hessian_seq_len+5, w.SEG(1,5), stage_cost::hessian(param, x.SEG(2,10), x.SEG(1,25), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+24, 4, hessian_seq+64, hessian_seq_len+6, w.SEG(1,6), stage_cost::hessian(param, x.SEG(2,12), x.SEG(1,26), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+28, 4, hessian_seq+75, hessian_seq_len+7, w.SEG(1,7), stage_cost::hessian(param, x.SEG(2,14), x.SEG(1,27), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+32, 4, hessian_seq+86, hessian_seq_len+8, w.SEG(1,8), stage_cost::hessian(param, x.SEG(2,16), x.SEG(1,28), x.SEG(2,29), x.SEG(1,31)));
      accHessian(val, gradient, hessian, grad_seq+36, 2, hessian_seq+94, hessian_seq_len+9, w.SEG(1,9), terminal_cost::hessian(param, x.SEG(2,18), x.SEG(2,29)));
      return val;
    }
    
  };
  
  static void variable_bounds(Eigen::Ref<variable_t> lb, Eigen::Ref<variable_t> ub)
  {
    constexpr scalar_t inf = std::numeric_limits<double>::infinity();
    lb << -12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-inf,-inf,0,-2,-4,-6,-8,-10,-12,-14,-16,-inf,-inf,-1;
    ub << inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,1,1,1,1,1,1,1,1,1,inf,inf,1;
  }
};
constexpr seqinfo QP::constraints::jac_seq[];
constexpr seqinfo QP::objective::grad_seq[];
constexpr seqinfo QP::objective::hessian_seq[];
constexpr int QP::objective::hessian_seq_len[];
#endif
