#ifndef __QP_HPP
#define __QP_HPP

struct QP
{
  static constexpr int num_variables = 77;
  using param_t = MyFunctions<double>::param_t;
  using scalar_t = double;
  using variable_t = Eigen::Vector<scalar_t, num_variables>;

  // Variable accessors
  static Eigen::Ref<Eigen::Vector<scalar_t, 2>> x(Eigen::Ref<variable_t> var, int ind) {return var.template segment<2>(0+2*ind);};
  static Eigen::Ref<Eigen::Matrix<scalar_t, 2, 25>> x(Eigen::Ref<variable_t> var) {return Eigen::Map<Eigen::Matrix<scalar_t, 2, 25>>(var.template segment<50>(0).data());};
  static Eigen::Ref<Eigen::Vector<scalar_t, 1>> u(Eigen::Ref<variable_t> var, int ind) {return var.template segment<1>(50+1*ind);};
  static Eigen::Ref<Eigen::Matrix<scalar_t, 1, 24>> u(Eigen::Ref<variable_t> var) {return Eigen::Map<Eigen::Matrix<scalar_t, 1, 24>>(var.template segment<24>(50).data());};
  static Eigen::Ref<Eigen::Vector<scalar_t, 2>> xss(Eigen::Ref<variable_t> var) {return var.template segment<2>(74);};
  static Eigen::Ref<Eigen::Vector<scalar_t, 1>> uss(Eigen::Ref<variable_t> var) {return var.template segment<1>(76);};

  // Define convenience names for all differentiable functions
  using dynamics = lampc::Jacobian<MyFunctions<double>::dynamics_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using dynamics_eq = lampc::Jacobian<MyFunctions<double>::dynamics_eq_, double, MyFunctions<double>::param_t_, 2, 2, 2, 1>;
  using dynamics_0 = lampc::Jacobian<MyFunctions<double>::dynamics_0_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using dynamics_ss = lampc::Jacobian<MyFunctions<double>::dynamics_ss_, double, MyFunctions<double>::param_t_, 2, 2, 1>;
  using output = lampc::Jacobian<MyFunctions<double>::output_, double, MyFunctions<double>::param_t_, 1, 2>;
  using stage_cost = lampc::Jacobian<MyFunctions<double>::stage_cost_, double, MyFunctions<double>::param_t_, 1, 2, 1, 2, 1>;
  using terminal_cost = lampc::Jacobian<MyFunctions<double>::terminal_cost_, double, MyFunctions<double>::param_t_, 1, 2, 2>;

  struct constraints : public function_util_t<scalar_t, Eigen::Vector<scalar_t, 50>, Eigen::SparseMatrix<scalar_t>>
  {
    static constexpr std::size_t output_size = 50;
    static constexpr std::size_t nnz_jacobian = 242;
    using out_t = Eigen::Vector<scalar_t, output_size>;
    using jacobian_t = Eigen::SparseMatrix<scalar_t>;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;

    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out)
    {
      out.SEG(2,0) = dynamics_0::eval(param, x.SEG(2,0),x.SEG(1,50)); // dynamics_0(x0,u0)
      out.SEG(2,2) = dynamics_eq::eval(param, x.SEG(2,4),x.SEG(2,2),x.SEG(1,51)); // dynamics_eq(x2,x1,u1)
      out.SEG(2,4) = dynamics_eq::eval(param, x.SEG(2,6),x.SEG(2,4),x.SEG(1,52)); // dynamics_eq(x3,x2,u2)
      out.SEG(2,6) = dynamics_eq::eval(param, x.SEG(2,8),x.SEG(2,6),x.SEG(1,53)); // dynamics_eq(x4,x3,u3)
      out.SEG(2,8) = dynamics_eq::eval(param, x.SEG(2,10),x.SEG(2,8),x.SEG(1,54)); // dynamics_eq(x5,x4,u4)
      out.SEG(2,10) = dynamics_eq::eval(param, x.SEG(2,12),x.SEG(2,10),x.SEG(1,55)); // dynamics_eq(x6,x5,u5)
      out.SEG(2,12) = dynamics_eq::eval(param, x.SEG(2,14),x.SEG(2,12),x.SEG(1,56)); // dynamics_eq(x7,x6,u6)
      out.SEG(2,14) = dynamics_eq::eval(param, x.SEG(2,16),x.SEG(2,14),x.SEG(1,57)); // dynamics_eq(x8,x7,u7)
      out.SEG(2,16) = dynamics_eq::eval(param, x.SEG(2,18),x.SEG(2,16),x.SEG(1,58)); // dynamics_eq(x9,x8,u8)
      out.SEG(2,18) = dynamics_eq::eval(param, x.SEG(2,20),x.SEG(2,18),x.SEG(1,59)); // dynamics_eq(x10,x9,u9)
      out.SEG(2,20) = dynamics_eq::eval(param, x.SEG(2,22),x.SEG(2,20),x.SEG(1,60)); // dynamics_eq(x11,x10,u10)
      out.SEG(2,22) = dynamics_eq::eval(param, x.SEG(2,24),x.SEG(2,22),x.SEG(1,61)); // dynamics_eq(x12,x11,u11)
      out.SEG(2,24) = dynamics_eq::eval(param, x.SEG(2,26),x.SEG(2,24),x.SEG(1,62)); // dynamics_eq(x13,x12,u12)
      out.SEG(2,26) = dynamics_eq::eval(param, x.SEG(2,28),x.SEG(2,26),x.SEG(1,63)); // dynamics_eq(x14,x13,u13)
      out.SEG(2,28) = dynamics_eq::eval(param, x.SEG(2,30),x.SEG(2,28),x.SEG(1,64)); // dynamics_eq(x15,x14,u14)
      out.SEG(2,30) = dynamics_eq::eval(param, x.SEG(2,32),x.SEG(2,30),x.SEG(1,65)); // dynamics_eq(x16,x15,u15)
      out.SEG(2,32) = dynamics_eq::eval(param, x.SEG(2,34),x.SEG(2,32),x.SEG(1,66)); // dynamics_eq(x17,x16,u16)
      out.SEG(2,34) = dynamics_eq::eval(param, x.SEG(2,36),x.SEG(2,34),x.SEG(1,67)); // dynamics_eq(x18,x17,u17)
      out.SEG(2,36) = dynamics_eq::eval(param, x.SEG(2,38),x.SEG(2,36),x.SEG(1,68)); // dynamics_eq(x19,x18,u18)
      out.SEG(2,38) = dynamics_eq::eval(param, x.SEG(2,40),x.SEG(2,38),x.SEG(1,69)); // dynamics_eq(x20,x19,u19)
      out.SEG(2,40) = dynamics_eq::eval(param, x.SEG(2,42),x.SEG(2,40),x.SEG(1,70)); // dynamics_eq(x21,x20,u20)
      out.SEG(2,42) = dynamics_eq::eval(param, x.SEG(2,44),x.SEG(2,42),x.SEG(1,71)); // dynamics_eq(x22,x21,u21)
      out.SEG(2,44) = dynamics_eq::eval(param, x.SEG(2,46),x.SEG(2,44),x.SEG(1,72)); // dynamics_eq(x23,x22,u22)
      out.SEG(2,46) = dynamics_eq::eval(param, x.SEG(2,48),x.SEG(2,46),x.SEG(1,73)); // dynamics_eq(x24,x23,u23)
      out.SEG(2,48) = dynamics_ss::eval(param, x.SEG(2,74),x.SEG(1,76)); // dynamics_ss(xss,uss)
    };

    static void initialize_jacobian(Eigen::SparseMatrix<scalar_t> &J)
    {
      J.resize(50,77);
      J.reserve(242);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,242> tripletList = {T{0,1,1},{0,50,1},{1,0,1},{1,1,1},{1,50,1},{2,2,1},{2,3,1},{2,4,1},{2,5,1},{2,51,1},{3,2,1},{3,3,1},{3,4,1},{3,5,1},{3,51,1},{4,4,1},{4,5,1},{4,6,1},{4,7,1},{4,52,1},{5,4,1},{5,5,1},{5,6,1},{5,7,1},{5,52,1},{6,6,1},{6,7,1},{6,8,1},{6,9,1},{6,53,1},{7,6,1},{7,7,1},{7,8,1},{7,9,1},{7,53,1},{8,8,1},{8,9,1},{8,10,1},{8,11,1},{8,54,1},{9,8,1},{9,9,1},{9,10,1},{9,11,1},{9,54,1},{10,10,1},{10,11,1},{10,12,1},{10,13,1},{10,55,1},{11,10,1},{11,11,1},{11,12,1},{11,13,1},{11,55,1},{12,12,1},{12,13,1},{12,14,1},{12,15,1},{12,56,1},{13,12,1},{13,13,1},{13,14,1},{13,15,1},{13,56,1},{14,14,1},{14,15,1},{14,16,1},{14,17,1},{14,57,1},{15,14,1},{15,15,1},{15,16,1},{15,17,1},{15,57,1},{16,16,1},{16,17,1},{16,18,1},{16,19,1},{16,58,1},{17,16,1},{17,17,1},{17,18,1},{17,19,1},{17,58,1},{18,18,1},{18,19,1},{18,20,1},{18,21,1},{18,59,1},{19,18,1},{19,19,1},{19,20,1},{19,21,1},{19,59,1},{20,20,1},{20,21,1},{20,22,1},{20,23,1},{20,60,1},{21,20,1},{21,21,1},{21,22,1},{21,23,1},{21,60,1},{22,22,1},{22,23,1},{22,24,1},{22,25,1},{22,61,1},{23,22,1},{23,23,1},{23,24,1},{23,25,1},{23,61,1},{24,24,1},{24,25,1},{24,26,1},{24,27,1},{24,62,1},{25,24,1},{25,25,1},{25,26,1},{25,27,1},{25,62,1},{26,26,1},{26,27,1},{26,28,1},{26,29,1},{26,63,1},{27,26,1},{27,27,1},{27,28,1},{27,29,1},{27,63,1},{28,28,1},{28,29,1},{28,30,1},{28,31,1},{28,64,1},{29,28,1},{29,29,1},{29,30,1},{29,31,1},{29,64,1},{30,30,1},{30,31,1},{30,32,1},{30,33,1},{30,65,1},{31,30,1},{31,31,1},{31,32,1},{31,33,1},{31,65,1},{32,32,1},{32,33,1},{32,34,1},{32,35,1},{32,66,1},{33,32,1},{33,33,1},{33,34,1},{33,35,1},{33,66,1},{34,34,1},{34,35,1},{34,36,1},{34,37,1},{34,67,1},{35,34,1},{35,35,1},{35,36,1},{35,37,1},{35,67,1},{36,36,1},{36,37,1},{36,38,1},{36,39,1},{36,68,1},{37,36,1},{37,37,1},{37,38,1},{37,39,1},{37,68,1},{38,38,1},{38,39,1},{38,40,1},{38,41,1},{38,69,1},{39,38,1},{39,39,1},{39,40,1},{39,41,1},{39,69,1},{40,40,1},{40,41,1},{40,42,1},{40,43,1},{40,70,1},{41,40,1},{41,41,1},{41,42,1},{41,43,1},{41,70,1},{42,42,1},{42,43,1},{42,44,1},{42,45,1},{42,71,1},{43,42,1},{43,43,1},{43,44,1},{43,45,1},{43,71,1},{44,44,1},{44,45,1},{44,46,1},{44,47,1},{44,72,1},{45,44,1},{45,45,1},{45,46,1},{45,47,1},{45,72,1},{46,46,1},{46,47,1},{46,48,1},{46,49,1},{46,73,1},{47,46,1},{47,47,1},{47,48,1},{47,49,1},{47,73,1},{48,74,1},{48,75,1},{48,76,1},{49,74,1},{49,75,1},{49,76,1}};
      J.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    /**
     * Compute the jacobian of the overall function
     */
    static constexpr seqinfo jac_seq[116] = {
      {0,4},{188,2},
      {8,2},{12,2},{4,4},{190,2},
      {16,2},{20,2},{10,2},{14,2},{192,2},
      {24,2},{28,2},{18,2},{22,2},{194,2},
      {32,2},{36,2},{26,2},{30,2},{196,2},
      {40,2},{44,2},{34,2},{38,2},{198,2},
      {48,2},{52,2},{42,2},{46,2},{200,2},
      {56,2},{60,2},{50,2},{54,2},{202,2},
      {64,2},{68,2},{58,2},{62,2},{204,2},
      {72,2},{76,2},{66,2},{70,2},{206,2},
      {80,2},{84,2},{74,2},{78,2},{208,2},
      {88,2},{92,2},{82,2},{86,2},{210,2},
      {96,2},{100,2},{90,2},{94,2},{212,2},
      {104,2},{108,2},{98,2},{102,2},{214,2},
      {112,2},{116,2},{106,2},{110,2},{216,2},
      {120,2},{124,2},{114,2},{118,2},{218,2},
      {128,2},{132,2},{122,2},{126,2},{220,2},
      {136,2},{140,2},{130,2},{134,2},{222,2},
      {144,2},{148,2},{138,2},{142,2},{224,2},
      {152,2},{156,2},{146,2},{150,2},{226,2},
      {160,2},{164,2},{154,2},{158,2},{228,2},
      {168,2},{172,2},{162,2},{166,2},{230,2},
      {176,2},{180,2},{170,2},{174,2},{232,2},
      {184,4},{178,2},{182,2},{234,2},
      {236,6}};
    static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out, Eigen::Ref<jacobian_t> jacobian)
    {
      setJ(out, jacobian, 0, jac_seq+0, 2, dynamics_0::jac(param, x.SEG(2,0),x.SEG(1,50))); // dynamics_0(x0,u0)
      setJ(out, jacobian, 2, jac_seq+2, 4, dynamics_eq::jac(param, x.SEG(2,4),x.SEG(2,2),x.SEG(1,51))); // dynamics_eq(x2,x1,u1)
      setJ(out, jacobian, 4, jac_seq+6, 5, dynamics_eq::jac(param, x.SEG(2,6),x.SEG(2,4),x.SEG(1,52))); // dynamics_eq(x3,x2,u2)
      setJ(out, jacobian, 6, jac_seq+11, 5, dynamics_eq::jac(param, x.SEG(2,8),x.SEG(2,6),x.SEG(1,53))); // dynamics_eq(x4,x3,u3)
      setJ(out, jacobian, 8, jac_seq+16, 5, dynamics_eq::jac(param, x.SEG(2,10),x.SEG(2,8),x.SEG(1,54))); // dynamics_eq(x5,x4,u4)
      setJ(out, jacobian, 10, jac_seq+21, 5, dynamics_eq::jac(param, x.SEG(2,12),x.SEG(2,10),x.SEG(1,55))); // dynamics_eq(x6,x5,u5)
      setJ(out, jacobian, 12, jac_seq+26, 5, dynamics_eq::jac(param, x.SEG(2,14),x.SEG(2,12),x.SEG(1,56))); // dynamics_eq(x7,x6,u6)
      setJ(out, jacobian, 14, jac_seq+31, 5, dynamics_eq::jac(param, x.SEG(2,16),x.SEG(2,14),x.SEG(1,57))); // dynamics_eq(x8,x7,u7)
      setJ(out, jacobian, 16, jac_seq+36, 5, dynamics_eq::jac(param, x.SEG(2,18),x.SEG(2,16),x.SEG(1,58))); // dynamics_eq(x9,x8,u8)
      setJ(out, jacobian, 18, jac_seq+41, 5, dynamics_eq::jac(param, x.SEG(2,20),x.SEG(2,18),x.SEG(1,59))); // dynamics_eq(x10,x9,u9)
      setJ(out, jacobian, 20, jac_seq+46, 5, dynamics_eq::jac(param, x.SEG(2,22),x.SEG(2,20),x.SEG(1,60))); // dynamics_eq(x11,x10,u10)
      setJ(out, jacobian, 22, jac_seq+51, 5, dynamics_eq::jac(param, x.SEG(2,24),x.SEG(2,22),x.SEG(1,61))); // dynamics_eq(x12,x11,u11)
      setJ(out, jacobian, 24, jac_seq+56, 5, dynamics_eq::jac(param, x.SEG(2,26),x.SEG(2,24),x.SEG(1,62))); // dynamics_eq(x13,x12,u12)
      setJ(out, jacobian, 26, jac_seq+61, 5, dynamics_eq::jac(param, x.SEG(2,28),x.SEG(2,26),x.SEG(1,63))); // dynamics_eq(x14,x13,u13)
      setJ(out, jacobian, 28, jac_seq+66, 5, dynamics_eq::jac(param, x.SEG(2,30),x.SEG(2,28),x.SEG(1,64))); // dynamics_eq(x15,x14,u14)
      setJ(out, jacobian, 30, jac_seq+71, 5, dynamics_eq::jac(param, x.SEG(2,32),x.SEG(2,30),x.SEG(1,65))); // dynamics_eq(x16,x15,u15)
      setJ(out, jacobian, 32, jac_seq+76, 5, dynamics_eq::jac(param, x.SEG(2,34),x.SEG(2,32),x.SEG(1,66))); // dynamics_eq(x17,x16,u16)
      setJ(out, jacobian, 34, jac_seq+81, 5, dynamics_eq::jac(param, x.SEG(2,36),x.SEG(2,34),x.SEG(1,67))); // dynamics_eq(x18,x17,u17)
      setJ(out, jacobian, 36, jac_seq+86, 5, dynamics_eq::jac(param, x.SEG(2,38),x.SEG(2,36),x.SEG(1,68))); // dynamics_eq(x19,x18,u18)
      setJ(out, jacobian, 38, jac_seq+91, 5, dynamics_eq::jac(param, x.SEG(2,40),x.SEG(2,38),x.SEG(1,69))); // dynamics_eq(x20,x19,u19)
      setJ(out, jacobian, 40, jac_seq+96, 5, dynamics_eq::jac(param, x.SEG(2,42),x.SEG(2,40),x.SEG(1,70))); // dynamics_eq(x21,x20,u20)
      setJ(out, jacobian, 42, jac_seq+101, 5, dynamics_eq::jac(param, x.SEG(2,44),x.SEG(2,42),x.SEG(1,71))); // dynamics_eq(x22,x21,u21)
      setJ(out, jacobian, 44, jac_seq+106, 5, dynamics_eq::jac(param, x.SEG(2,46),x.SEG(2,44),x.SEG(1,72))); // dynamics_eq(x23,x22,u22)
      setJ(out, jacobian, 46, jac_seq+111, 4, dynamics_eq::jac(param, x.SEG(2,48),x.SEG(2,46),x.SEG(1,73))); // dynamics_eq(x24,x23,u23)
      setJ(out, jacobian, 48, jac_seq+115, 1, dynamics_ss::jac(param, x.SEG(2,74),x.SEG(1,76))); // dynamics_ss(xss,uss)
    };

    static void bounds(param_t &param, Eigen::Ref<out_t> lb, Eigen::Ref<out_t> ub)
    {
      constexpr scalar_t inf = std::numeric_limits<double>::infinity();
      lb << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
      ub << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
    };
  };

  struct objective : public weightedsum_util_t<scalar_t, Eigen::Vector<scalar_t, num_variables>, Eigen::Vector<scalar_t, 27>>
  {
    static constexpr std::size_t num_weights = 27;
    using weight_t = Eigen::Vector<scalar_t, num_weights>;
    using gradient_t = Eigen::Vector<scalar_t, num_variables>;
    static constexpr std::size_t hessian_nnz = 673;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;


    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> &w, const Eigen::Ref<const variable_t> &x)
    {
      scalar_t val = 0;
      val += w.SEG(2,0).dot(dynamics_0::eval(param, x.SEG(2,4),x.SEG(1,51))); // dynamics_0(x2,u1)
      val += w.SEG(1,2).dot(terminal_cost::eval(param, x.SEG(2,48),x.SEG(2,74))); // terminal_cost(x24,xss)
      val += w.SEG(1,3).dot(stage_cost::eval(param, x.SEG(2,0),x.SEG(1,50),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x0,u0,xss,uss)
      val += w.SEG(1,4).dot(stage_cost::eval(param, x.SEG(2,2),x.SEG(1,51),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x1,u1,xss,uss)
      val += w.SEG(1,5).dot(stage_cost::eval(param, x.SEG(2,4),x.SEG(1,52),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x2,u2,xss,uss)
      val += w.SEG(1,6).dot(stage_cost::eval(param, x.SEG(2,6),x.SEG(1,53),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x3,u3,xss,uss)
      val += w.SEG(1,7).dot(stage_cost::eval(param, x.SEG(2,8),x.SEG(1,54),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x4,u4,xss,uss)
      val += w.SEG(1,8).dot(stage_cost::eval(param, x.SEG(2,10),x.SEG(1,55),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x5,u5,xss,uss)
      val += w.SEG(1,9).dot(stage_cost::eval(param, x.SEG(2,12),x.SEG(1,56),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x6,u6,xss,uss)
      val += w.SEG(1,10).dot(stage_cost::eval(param, x.SEG(2,14),x.SEG(1,57),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x7,u7,xss,uss)
      val += w.SEG(1,11).dot(stage_cost::eval(param, x.SEG(2,16),x.SEG(1,58),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x8,u8,xss,uss)
      val += w.SEG(1,12).dot(stage_cost::eval(param, x.SEG(2,18),x.SEG(1,59),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x9,u9,xss,uss)
      val += w.SEG(1,13).dot(stage_cost::eval(param, x.SEG(2,20),x.SEG(1,60),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x10,u10,xss,uss)
      val += w.SEG(1,14).dot(stage_cost::eval(param, x.SEG(2,22),x.SEG(1,61),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x11,u11,xss,uss)
      val += w.SEG(1,15).dot(stage_cost::eval(param, x.SEG(2,24),x.SEG(1,62),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x12,u12,xss,uss)
      val += w.SEG(1,16).dot(stage_cost::eval(param, x.SEG(2,26),x.SEG(1,63),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x13,u13,xss,uss)
      val += w.SEG(1,17).dot(stage_cost::eval(param, x.SEG(2,28),x.SEG(1,64),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x14,u14,xss,uss)
      val += w.SEG(1,18).dot(stage_cost::eval(param, x.SEG(2,30),x.SEG(1,65),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x15,u15,xss,uss)
      val += w.SEG(1,19).dot(stage_cost::eval(param, x.SEG(2,32),x.SEG(1,66),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x16,u16,xss,uss)
      val += w.SEG(1,20).dot(stage_cost::eval(param, x.SEG(2,34),x.SEG(1,67),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x17,u17,xss,uss)
      val += w.SEG(1,21).dot(stage_cost::eval(param, x.SEG(2,36),x.SEG(1,68),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x18,u18,xss,uss)
      val += w.SEG(1,22).dot(stage_cost::eval(param, x.SEG(2,38),x.SEG(1,69),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x19,u19,xss,uss)
      val += w.SEG(1,23).dot(stage_cost::eval(param, x.SEG(2,40),x.SEG(1,70),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x20,u20,xss,uss)
      val += w.SEG(1,24).dot(stage_cost::eval(param, x.SEG(2,42),x.SEG(1,71),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x21,u21,xss,uss)
      val += w.SEG(1,25).dot(stage_cost::eval(param, x.SEG(2,44),x.SEG(1,72),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x22,u22,xss,uss)
      val += w.SEG(1,26).dot(stage_cost::eval(param, x.SEG(2,46),x.SEG(1,73),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x23,u23,xss,uss)
      return val;
    };

    /**
     * Compute the gradient of the weighted sum
     */
    static constexpr seqinfo grad_seq[100] = {
      {4,2},{51,1},
      {48,2},{74,2},
      {0,2},{50,1},{74,2},{76,1},
      {2,2},{51,1},{74,2},{76,1},
      {4,2},{52,1},{74,2},{76,1},
      {6,2},{53,1},{74,2},{76,1},
      {8,2},{54,1},{74,2},{76,1},
      {10,2},{55,1},{74,2},{76,1},
      {12,2},{56,1},{74,2},{76,1},
      {14,2},{57,1},{74,2},{76,1},
      {16,2},{58,1},{74,2},{76,1},
      {18,2},{59,1},{74,2},{76,1},
      {20,2},{60,1},{74,2},{76,1},
      {22,2},{61,1},{74,2},{76,1},
      {24,2},{62,1},{74,2},{76,1},
      {26,2},{63,1},{74,2},{76,1},
      {28,2},{64,1},{74,2},{76,1},
      {30,2},{65,1},{74,2},{76,1},
      {32,2},{66,1},{74,2},{76,1},
      {34,2},{67,1},{74,2},{76,1},
      {36,2},{68,1},{74,2},{76,1},
      {38,2},{69,1},{74,2},{76,1},
      {40,2},{70,1},{74,2},{76,1},
      {42,2},{71,1},{74,2},{76,1},
      {44,2},{72,1},{74,2},{76,1},
      {46,2},{73,1},{74,2},{76,1}};
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient)
    {
      gradient.array() = 0;
      scalar_t val = 0;
      accGrad(val, gradient, grad_seq+0, 2, w.SEG(2,0), dynamics_0::jac(param, x.SEG(2,4),x.SEG(1,51))); // {std(call)}
      accGrad(val, gradient, grad_seq+2, 2, w.SEG(1,2), terminal_cost::jac(param, x.SEG(2,48),x.SEG(2,74))); // {std(call)}
      accGrad(val, gradient, grad_seq+4, 4, w.SEG(1,3), stage_cost::jac(param, x.SEG(2,0),x.SEG(1,50),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+8, 4, w.SEG(1,4), stage_cost::jac(param, x.SEG(2,2),x.SEG(1,51),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+12, 4, w.SEG(1,5), stage_cost::jac(param, x.SEG(2,4),x.SEG(1,52),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+16, 4, w.SEG(1,6), stage_cost::jac(param, x.SEG(2,6),x.SEG(1,53),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+20, 4, w.SEG(1,7), stage_cost::jac(param, x.SEG(2,8),x.SEG(1,54),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+24, 4, w.SEG(1,8), stage_cost::jac(param, x.SEG(2,10),x.SEG(1,55),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+28, 4, w.SEG(1,9), stage_cost::jac(param, x.SEG(2,12),x.SEG(1,56),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+32, 4, w.SEG(1,10), stage_cost::jac(param, x.SEG(2,14),x.SEG(1,57),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+36, 4, w.SEG(1,11), stage_cost::jac(param, x.SEG(2,16),x.SEG(1,58),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+40, 4, w.SEG(1,12), stage_cost::jac(param, x.SEG(2,18),x.SEG(1,59),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+44, 4, w.SEG(1,13), stage_cost::jac(param, x.SEG(2,20),x.SEG(1,60),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+48, 4, w.SEG(1,14), stage_cost::jac(param, x.SEG(2,22),x.SEG(1,61),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+52, 4, w.SEG(1,15), stage_cost::jac(param, x.SEG(2,24),x.SEG(1,62),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+56, 4, w.SEG(1,16), stage_cost::jac(param, x.SEG(2,26),x.SEG(1,63),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+60, 4, w.SEG(1,17), stage_cost::jac(param, x.SEG(2,28),x.SEG(1,64),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+64, 4, w.SEG(1,18), stage_cost::jac(param, x.SEG(2,30),x.SEG(1,65),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+68, 4, w.SEG(1,19), stage_cost::jac(param, x.SEG(2,32),x.SEG(1,66),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+72, 4, w.SEG(1,20), stage_cost::jac(param, x.SEG(2,34),x.SEG(1,67),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+76, 4, w.SEG(1,21), stage_cost::jac(param, x.SEG(2,36),x.SEG(1,68),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+80, 4, w.SEG(1,22), stage_cost::jac(param, x.SEG(2,38),x.SEG(1,69),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+84, 4, w.SEG(1,23), stage_cost::jac(param, x.SEG(2,40),x.SEG(1,70),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+88, 4, w.SEG(1,24), stage_cost::jac(param, x.SEG(2,42),x.SEG(1,71),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+92, 4, w.SEG(1,25), stage_cost::jac(param, x.SEG(2,44),x.SEG(1,72),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+96, 4, w.SEG(1,26), stage_cost::jac(param, x.SEG(2,46),x.SEG(1,73),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      return val;
    };

    /**
     * Initialize the hessian of the function
     */
    static void initialize_hessian(Eigen::SparseMatrix<scalar_t> &H)
    {
      H.resize(77,77);
      H.reserve(673);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,673> tripletList = {T{0,1,1},{0,50,1},{0,74,1},{0,75,1},{0,76,1},{1,0,1},{1,1,1},{1,50,1},{1,74,1},{1,75,1},{1,76,1},{2,2,1},{2,3,1},{2,51,1},{2,74,1},{2,75,1},{2,76,1},{3,2,1},{3,3,1},{3,51,1},{3,74,1},{3,75,1},{3,76,1},{4,4,1},{4,5,1},{4,51,1},{4,52,1},{4,74,1},{4,75,1},{4,76,1},{5,4,1},{5,5,1},{5,51,1},{5,52,1},{5,74,1},{5,75,1},{5,76,1},{6,6,1},{6,7,1},{6,53,1},{6,74,1},{6,75,1},{6,76,1},{7,6,1},{7,7,1},{7,53,1},{7,74,1},{7,75,1},{7,76,1},{8,8,1},{8,9,1},{8,54,1},{8,74,1},{8,75,1},{8,76,1},{9,8,1},{9,9,1},{9,54,1},{9,74,1},{9,75,1},{9,76,1},{10,10,1},{10,11,1},{10,55,1},{10,74,1},{10,75,1},{10,76,1},{11,10,1},{11,11,1},{11,55,1},{11,74,1},{11,75,1},{11,76,1},{12,12,1},{12,13,1},{12,56,1},{12,74,1},{12,75,1},{12,76,1},{13,12,1},{13,13,1},{13,56,1},{13,74,1},{13,75,1},{13,76,1},{14,14,1},{14,15,1},{14,57,1},{14,74,1},{14,75,1},{14,76,1},{15,14,1},{15,15,1},{15,57,1},{15,74,1},{15,75,1},{15,76,1},{16,16,1},{16,17,1},{16,58,1},{16,74,1},{16,75,1},{16,76,1},{17,16,1},{17,17,1},{17,58,1},{17,74,1},{17,75,1},{17,76,1},{18,18,1},{18,19,1},{18,59,1},{18,74,1},{18,75,1},{18,76,1},{19,18,1},{19,19,1},{19,59,1},{19,74,1},{19,75,1},{19,76,1},{20,20,1},{20,21,1},{20,60,1},{20,74,1},{20,75,1},{20,76,1},{21,20,1},{21,21,1},{21,60,1},{21,74,1},{21,75,1},{21,76,1},{22,22,1},{22,23,1},{22,61,1},{22,74,1},{22,75,1},{22,76,1},{23,22,1},{23,23,1},{23,61,1},{23,74,1},{23,75,1},{23,76,1},{24,24,1},{24,25,1},{24,62,1},{24,74,1},{24,75,1},{24,76,1},{25,24,1},{25,25,1},{25,62,1},{25,74,1},{25,75,1},{25,76,1},{26,26,1},{26,27,1},{26,63,1},{26,74,1},{26,75,1},{26,76,1},{27,26,1},{27,27,1},{27,63,1},{27,74,1},{27,75,1},{27,76,1},{28,28,1},{28,29,1},{28,64,1},{28,74,1},{28,75,1},{28,76,1},{29,28,1},{29,29,1},{29,64,1},{29,74,1},{29,75,1},{29,76,1},{30,30,1},{30,31,1},{30,65,1},{30,74,1},{30,75,1},{30,76,1},{31,30,1},{31,31,1},{31,65,1},{31,74,1},{31,75,1},{31,76,1},{32,32,1},{32,33,1},{32,66,1},{32,74,1},{32,75,1},{32,76,1},{33,32,1},{33,33,1},{33,66,1},{33,74,1},{33,75,1},{33,76,1},{34,34,1},{34,35,1},{34,67,1},{34,74,1},{34,75,1},{34,76,1},{35,34,1},{35,35,1},{35,67,1},{35,74,1},{35,75,1},{35,76,1},{36,36,1},{36,37,1},{36,68,1},{36,74,1},{36,75,1},{36,76,1},{37,36,1},{37,37,1},{37,68,1},{37,74,1},{37,75,1},{37,76,1},{38,38,1},{38,39,1},{38,69,1},{38,74,1},{38,75,1},{38,76,1},{39,38,1},{39,39,1},{39,69,1},{39,74,1},{39,75,1},{39,76,1},{40,40,1},{40,41,1},{40,70,1},{40,74,1},{40,75,1},{40,76,1},{41,40,1},{41,41,1},{41,70,1},{41,74,1},{41,75,1},{41,76,1},{42,42,1},{42,43,1},{42,71,1},{42,74,1},{42,75,1},{42,76,1},{43,42,1},{43,43,1},{43,71,1},{43,74,1},{43,75,1},{43,76,1},{44,44,1},{44,45,1},{44,72,1},{44,74,1},{44,75,1},{44,76,1},{45,44,1},{45,45,1},{45,72,1},{45,74,1},{45,75,1},{45,76,1},{46,46,1},{46,47,1},{46,73,1},{46,74,1},{46,75,1},{46,76,1},{47,46,1},{47,47,1},{47,73,1},{47,74,1},{47,75,1},{47,76,1},{48,48,1},{48,49,1},{48,74,1},{48,75,1},{49,48,1},{49,49,1},{49,74,1},{49,75,1},{50,0,1},{50,1,1},{50,50,1},{50,74,1},{50,75,1},{50,76,1},{51,2,1},{51,3,1},{51,4,1},{51,5,1},{51,51,1},{51,74,1},{51,75,1},{51,76,1},{52,4,1},{52,5,1},{52,52,1},{52,74,1},{52,75,1},{52,76,1},{53,6,1},{53,7,1},{53,53,1},{53,74,1},{53,75,1},{53,76,1},{54,8,1},{54,9,1},{54,54,1},{54,74,1},{54,75,1},{54,76,1},{55,10,1},{55,11,1},{55,55,1},{55,74,1},{55,75,1},{55,76,1},{56,12,1},{56,13,1},{56,56,1},{56,74,1},{56,75,1},{56,76,1},{57,14,1},{57,15,1},{57,57,1},{57,74,1},{57,75,1},{57,76,1},{58,16,1},{58,17,1},{58,58,1},{58,74,1},{58,75,1},{58,76,1},{59,18,1},{59,19,1},{59,59,1},{59,74,1},{59,75,1},{59,76,1},{60,20,1},{60,21,1},{60,60,1},{60,74,1},{60,75,1},{60,76,1},{61,22,1},{61,23,1},{61,61,1},{61,74,1},{61,75,1},{61,76,1},{62,24,1},{62,25,1},{62,62,1},{62,74,1},{62,75,1},{62,76,1},{63,26,1},{63,27,1},{63,63,1},{63,74,1},{63,75,1},{63,76,1},{64,28,1},{64,29,1},{64,64,1},{64,74,1},{64,75,1},{64,76,1},{65,30,1},{65,31,1},{65,65,1},{65,74,1},{65,75,1},{65,76,1},{66,32,1},{66,33,1},{66,66,1},{66,74,1},{66,75,1},{66,76,1},{67,34,1},{67,35,1},{67,67,1},{67,74,1},{67,75,1},{67,76,1},{68,36,1},{68,37,1},{68,68,1},{68,74,1},{68,75,1},{68,76,1},{69,38,1},{69,39,1},{69,69,1},{69,74,1},{69,75,1},{69,76,1},{70,40,1},{70,41,1},{70,70,1},{70,74,1},{70,75,1},{70,76,1},{71,42,1},{71,43,1},{71,71,1},{71,74,1},{71,75,1},{71,76,1},{72,44,1},{72,45,1},{72,72,1},{72,74,1},{72,75,1},{72,76,1},{73,46,1},{73,47,1},{73,73,1},{73,74,1},{73,75,1},{73,76,1},{74,0,1},{74,1,1},{74,2,1},{74,3,1},{74,4,1},{74,5,1},{74,6,1},{74,7,1},{74,8,1},{74,9,1},{74,10,1},{74,11,1},{74,12,1},{74,13,1},{74,14,1},{74,15,1},{74,16,1},{74,17,1},{74,18,1},{74,19,1},{74,20,1},{74,21,1},{74,22,1},{74,23,1},{74,24,1},{74,25,1},{74,26,1},{74,27,1},{74,28,1},{74,29,1},{74,30,1},{74,31,1},{74,32,1},{74,33,1},{74,34,1},{74,35,1},{74,36,1},{74,37,1},{74,38,1},{74,39,1},{74,40,1},{74,41,1},{74,42,1},{74,43,1},{74,44,1},{74,45,1},{74,46,1},{74,47,1},{74,48,1},{74,49,1},{74,50,1},{74,51,1},{74,52,1},{74,53,1},{74,54,1},{74,55,1},{74,56,1},{74,57,1},{74,58,1},{74,59,1},{74,60,1},{74,61,1},{74,62,1},{74,63,1},{74,64,1},{74,65,1},{74,66,1},{74,67,1},{74,68,1},{74,69,1},{74,70,1},{74,71,1},{74,72,1},{74,73,1},{74,74,1},{74,75,1},{74,76,1},{75,0,1},{75,1,1},{75,2,1},{75,3,1},{75,4,1},{75,5,1},{75,6,1},{75,7,1},{75,8,1},{75,9,1},{75,10,1},{75,11,1},{75,12,1},{75,13,1},{75,14,1},{75,15,1},{75,16,1},{75,17,1},{75,18,1},{75,19,1},{75,20,1},{75,21,1},{75,22,1},{75,23,1},{75,24,1},{75,25,1},{75,26,1},{75,27,1},{75,28,1},{75,29,1},{75,30,1},{75,31,1},{75,32,1},{75,33,1},{75,34,1},{75,35,1},{75,36,1},{75,37,1},{75,38,1},{75,39,1},{75,40,1},{75,41,1},{75,42,1},{75,43,1},{75,44,1},{75,45,1},{75,46,1},{75,47,1},{75,48,1},{75,49,1},{75,50,1},{75,51,1},{75,52,1},{75,53,1},{75,54,1},{75,55,1},{75,56,1},{75,57,1},{75,58,1},{75,59,1},{75,60,1},{75,61,1},{75,62,1},{75,63,1},{75,64,1},{75,65,1},{75,66,1},{75,67,1},{75,68,1},{75,69,1},{75,70,1},{75,71,1},{75,72,1},{75,73,1},{75,74,1},{75,75,1},{75,76,1},{76,0,1},{76,1,1},{76,2,1},{76,3,1},{76,4,1},{76,5,1},{76,6,1},{76,7,1},{76,8,1},{76,9,1},{76,10,1},{76,11,1},{76,12,1},{76,13,1},{76,14,1},{76,15,1},{76,16,1},{76,17,1},{76,18,1},{76,19,1},{76,20,1},{76,21,1},{76,22,1},{76,23,1},{76,24,1},{76,25,1},{76,26,1},{76,27,1},{76,28,1},{76,29,1},{76,30,1},{76,31,1},{76,32,1},{76,33,1},{76,34,1},{76,35,1},{76,36,1},{76,37,1},{76,38,1},{76,39,1},{76,40,1},{76,41,1},{76,42,1},{76,43,1},{76,44,1},{76,45,1},{76,46,1},{76,47,1},{76,50,1},{76,51,1},{76,52,1},{76,53,1},{76,54,1},{76,55,1},{76,56,1},{76,57,1},{76,58,1},{76,59,1},{76,60,1},{76,61,1},{76,62,1},{76,63,1},{76,64,1},{76,65,1},{76,66,1},{76,67,1},{76,68,1},{76,69,1},{76,70,1},{76,71,1},{76,72,1},{76,73,1},{76,74,1},{76,75,1},{76,76,1}};
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

    static constexpr seqinfo hessian_seq[273] = {
      {24,3},{31,3},{306,3},{24,3},{31,3},{306,3},
      {290,8},{492,2},{518,2},{569,2},{595,2},
      {0,12},{298,6},{444,2},{494,1},{518,5},{571,1},{595,5},{646,1},{670,3},
      {12,12},{304,2},{308,4},{446,2},{495,1},{518,3},{523,2},{572,1},{595,3},{600,2},{647,1},{670,3},
      {24,2},{27,6},{34,4},{312,6},{448,2},{496,1},{518,3},{525,2},{573,1},{595,3},{602,2},{648,1},{670,3},
      {38,12},{318,6},{450,2},{497,1},{518,3},{527,2},{574,1},{595,3},{604,2},{649,1},{670,3},
      {50,12},{324,6},{452,2},{498,1},{518,3},{529,2},{575,1},{595,3},{606,2},{650,1},{670,3},
      {62,12},{330,6},{454,2},{499,1},{518,3},{531,2},{576,1},{595,3},{608,2},{651,1},{670,3},
      {74,12},{336,6},{456,2},{500,1},{518,3},{533,2},{577,1},{595,3},{610,2},{652,1},{670,3},
      {86,12},{342,6},{458,2},{501,1},{518,3},{535,2},{578,1},{595,3},{612,2},{653,1},{670,3},
      {98,12},{348,6},{460,2},{502,1},{518,3},{537,2},{579,1},{595,3},{614,2},{654,1},{670,3},
      {110,12},{354,6},{462,2},{503,1},{518,3},{539,2},{580,1},{595,3},{616,2},{655,1},{670,3},
      {122,12},{360,6},{464,2},{504,1},{518,3},{541,2},{581,1},{595,3},{618,2},{656,1},{670,3},
      {134,12},{366,6},{466,2},{505,1},{518,3},{543,2},{582,1},{595,3},{620,2},{657,1},{670,3},
      {146,12},{372,6},{468,2},{506,1},{518,3},{545,2},{583,1},{595,3},{622,2},{658,1},{670,3},
      {158,12},{378,6},{470,2},{507,1},{518,3},{547,2},{584,1},{595,3},{624,2},{659,1},{670,3},
      {170,12},{384,6},{472,2},{508,1},{518,3},{549,2},{585,1},{595,3},{626,2},{660,1},{670,3},
      {182,12},{390,6},{474,2},{509,1},{518,3},{551,2},{586,1},{595,3},{628,2},{661,1},{670,3},
      {194,12},{396,6},{476,2},{510,1},{518,3},{553,2},{587,1},{595,3},{630,2},{662,1},{670,3},
      {206,12},{402,6},{478,2},{511,1},{518,3},{555,2},{588,1},{595,3},{632,2},{663,1},{670,3},
      {218,12},{408,6},{480,2},{512,1},{518,3},{557,2},{589,1},{595,3},{634,2},{664,1},{670,3},
      {230,12},{414,6},{482,2},{513,1},{518,3},{559,2},{590,1},{595,3},{636,2},{665,1},{670,3},
      {242,12},{420,6},{484,2},{514,1},{518,3},{561,2},{591,1},{595,3},{638,2},{666,1},{670,3},
      {254,12},{426,6},{486,2},{515,1},{518,3},{563,2},{592,1},{595,3},{640,2},{667,1},{670,3},
      {266,12},{432,6},{488,2},{516,1},{518,3},{565,2},{593,1},{595,3},{642,2},{668,1},{670,3},
      {278,12},{438,6},{490,2},{517,4},{567,2},{594,4},{644,2},{669,4}};
    static constexpr int hessian_seq_len[27] = {3,3,5,9,12,13,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,8};
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient, Eigen::Ref<hessian_t> hessian)
    {
      gradient.array() = 0;
      scalar_t val = 0;
      auto ptr = hessian.valuePtr();
      for(int i=0; i<hessian.nonZeros(); i++) ptr[i] = 0;

      accHessian(val, gradient, hessian, grad_seq+0, 2, hessian_seq+0, hessian_seq_len+0, w.SEG(2,0), dynamics_0::hessian(param, x.SEG(2,4),x.SEG(1,51)));
      accHessian(val, gradient, hessian, grad_seq+2, 2, hessian_seq+6, hessian_seq_len+2, w.SEG(1,2), terminal_cost::hessian(param, x.SEG(2,48),x.SEG(2,74)));
      accHessian(val, gradient, hessian, grad_seq+4, 4, hessian_seq+11, hessian_seq_len+3, w.SEG(1,3), stage_cost::hessian(param, x.SEG(2,0),x.SEG(1,50),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+8, 4, hessian_seq+20, hessian_seq_len+4, w.SEG(1,4), stage_cost::hessian(param, x.SEG(2,2),x.SEG(1,51),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+12, 4, hessian_seq+32, hessian_seq_len+5, w.SEG(1,5), stage_cost::hessian(param, x.SEG(2,4),x.SEG(1,52),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+16, 4, hessian_seq+45, hessian_seq_len+6, w.SEG(1,6), stage_cost::hessian(param, x.SEG(2,6),x.SEG(1,53),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+20, 4, hessian_seq+56, hessian_seq_len+7, w.SEG(1,7), stage_cost::hessian(param, x.SEG(2,8),x.SEG(1,54),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+24, 4, hessian_seq+67, hessian_seq_len+8, w.SEG(1,8), stage_cost::hessian(param, x.SEG(2,10),x.SEG(1,55),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+28, 4, hessian_seq+78, hessian_seq_len+9, w.SEG(1,9), stage_cost::hessian(param, x.SEG(2,12),x.SEG(1,56),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+32, 4, hessian_seq+89, hessian_seq_len+10, w.SEG(1,10), stage_cost::hessian(param, x.SEG(2,14),x.SEG(1,57),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+36, 4, hessian_seq+100, hessian_seq_len+11, w.SEG(1,11), stage_cost::hessian(param, x.SEG(2,16),x.SEG(1,58),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+40, 4, hessian_seq+111, hessian_seq_len+12, w.SEG(1,12), stage_cost::hessian(param, x.SEG(2,18),x.SEG(1,59),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+44, 4, hessian_seq+122, hessian_seq_len+13, w.SEG(1,13), stage_cost::hessian(param, x.SEG(2,20),x.SEG(1,60),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+48, 4, hessian_seq+133, hessian_seq_len+14, w.SEG(1,14), stage_cost::hessian(param, x.SEG(2,22),x.SEG(1,61),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+52, 4, hessian_seq+144, hessian_seq_len+15, w.SEG(1,15), stage_cost::hessian(param, x.SEG(2,24),x.SEG(1,62),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+56, 4, hessian_seq+155, hessian_seq_len+16, w.SEG(1,16), stage_cost::hessian(param, x.SEG(2,26),x.SEG(1,63),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+60, 4, hessian_seq+166, hessian_seq_len+17, w.SEG(1,17), stage_cost::hessian(param, x.SEG(2,28),x.SEG(1,64),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+64, 4, hessian_seq+177, hessian_seq_len+18, w.SEG(1,18), stage_cost::hessian(param, x.SEG(2,30),x.SEG(1,65),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+68, 4, hessian_seq+188, hessian_seq_len+19, w.SEG(1,19), stage_cost::hessian(param, x.SEG(2,32),x.SEG(1,66),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+72, 4, hessian_seq+199, hessian_seq_len+20, w.SEG(1,20), stage_cost::hessian(param, x.SEG(2,34),x.SEG(1,67),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+76, 4, hessian_seq+210, hessian_seq_len+21, w.SEG(1,21), stage_cost::hessian(param, x.SEG(2,36),x.SEG(1,68),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+80, 4, hessian_seq+221, hessian_seq_len+22, w.SEG(1,22), stage_cost::hessian(param, x.SEG(2,38),x.SEG(1,69),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+84, 4, hessian_seq+232, hessian_seq_len+23, w.SEG(1,23), stage_cost::hessian(param, x.SEG(2,40),x.SEG(1,70),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+88, 4, hessian_seq+243, hessian_seq_len+24, w.SEG(1,24), stage_cost::hessian(param, x.SEG(2,42),x.SEG(1,71),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+92, 4, hessian_seq+254, hessian_seq_len+25, w.SEG(1,25), stage_cost::hessian(param, x.SEG(2,44),x.SEG(1,72),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+96, 4, hessian_seq+265, hessian_seq_len+26, w.SEG(1,26), stage_cost::hessian(param, x.SEG(2,46),x.SEG(1,73),x.SEG(2,74),x.SEG(1,76)));
      return val;
    }
  };
  struct lagrangian : public weightedsum_util_t<scalar_t, Eigen::Vector<scalar_t, num_variables>, Eigen::Vector<scalar_t, 77>>
  {
    static constexpr std::size_t num_weights = 77;
    using weight_t = Eigen::Vector<scalar_t, num_weights>;
    using gradient_t = Eigen::Vector<scalar_t, num_variables>;
    static constexpr std::size_t hessian_nnz = 945;
    using hessian_t = Eigen::SparseMatrix<scalar_t>;


    /**
     * Evalute the function for the parameter param and return the result in out
     */
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> &w, const Eigen::Ref<const variable_t> &x)
    {
      scalar_t val = 0;
      val += w.SEG(2,0).dot(dynamics_0::eval(param, x.SEG(2,4),x.SEG(1,51))); // dynamics_0(x2,u1)
      val += w.SEG(1,2).dot(terminal_cost::eval(param, x.SEG(2,48),x.SEG(2,74))); // terminal_cost(x24,xss)
      val += w.SEG(1,3).dot(stage_cost::eval(param, x.SEG(2,0),x.SEG(1,50),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x0,u0,xss,uss)
      val += w.SEG(1,4).dot(stage_cost::eval(param, x.SEG(2,2),x.SEG(1,51),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x1,u1,xss,uss)
      val += w.SEG(1,5).dot(stage_cost::eval(param, x.SEG(2,4),x.SEG(1,52),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x2,u2,xss,uss)
      val += w.SEG(1,6).dot(stage_cost::eval(param, x.SEG(2,6),x.SEG(1,53),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x3,u3,xss,uss)
      val += w.SEG(1,7).dot(stage_cost::eval(param, x.SEG(2,8),x.SEG(1,54),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x4,u4,xss,uss)
      val += w.SEG(1,8).dot(stage_cost::eval(param, x.SEG(2,10),x.SEG(1,55),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x5,u5,xss,uss)
      val += w.SEG(1,9).dot(stage_cost::eval(param, x.SEG(2,12),x.SEG(1,56),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x6,u6,xss,uss)
      val += w.SEG(1,10).dot(stage_cost::eval(param, x.SEG(2,14),x.SEG(1,57),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x7,u7,xss,uss)
      val += w.SEG(1,11).dot(stage_cost::eval(param, x.SEG(2,16),x.SEG(1,58),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x8,u8,xss,uss)
      val += w.SEG(1,12).dot(stage_cost::eval(param, x.SEG(2,18),x.SEG(1,59),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x9,u9,xss,uss)
      val += w.SEG(1,13).dot(stage_cost::eval(param, x.SEG(2,20),x.SEG(1,60),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x10,u10,xss,uss)
      val += w.SEG(1,14).dot(stage_cost::eval(param, x.SEG(2,22),x.SEG(1,61),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x11,u11,xss,uss)
      val += w.SEG(1,15).dot(stage_cost::eval(param, x.SEG(2,24),x.SEG(1,62),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x12,u12,xss,uss)
      val += w.SEG(1,16).dot(stage_cost::eval(param, x.SEG(2,26),x.SEG(1,63),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x13,u13,xss,uss)
      val += w.SEG(1,17).dot(stage_cost::eval(param, x.SEG(2,28),x.SEG(1,64),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x14,u14,xss,uss)
      val += w.SEG(1,18).dot(stage_cost::eval(param, x.SEG(2,30),x.SEG(1,65),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x15,u15,xss,uss)
      val += w.SEG(1,19).dot(stage_cost::eval(param, x.SEG(2,32),x.SEG(1,66),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x16,u16,xss,uss)
      val += w.SEG(1,20).dot(stage_cost::eval(param, x.SEG(2,34),x.SEG(1,67),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x17,u17,xss,uss)
      val += w.SEG(1,21).dot(stage_cost::eval(param, x.SEG(2,36),x.SEG(1,68),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x18,u18,xss,uss)
      val += w.SEG(1,22).dot(stage_cost::eval(param, x.SEG(2,38),x.SEG(1,69),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x19,u19,xss,uss)
      val += w.SEG(1,23).dot(stage_cost::eval(param, x.SEG(2,40),x.SEG(1,70),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x20,u20,xss,uss)
      val += w.SEG(1,24).dot(stage_cost::eval(param, x.SEG(2,42),x.SEG(1,71),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x21,u21,xss,uss)
      val += w.SEG(1,25).dot(stage_cost::eval(param, x.SEG(2,44),x.SEG(1,72),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x22,u22,xss,uss)
      val += w.SEG(1,26).dot(stage_cost::eval(param, x.SEG(2,46),x.SEG(1,73),x.SEG(2,74),x.SEG(1,76))); // stage_cost(x23,u23,xss,uss)
      val += w.SEG(2,27).dot(dynamics_0::eval(param, x.SEG(2,0),x.SEG(1,50))); // dynamics_0(x0,u0)
      val += w.SEG(2,29).dot(dynamics_eq::eval(param, x.SEG(2,4),x.SEG(2,2),x.SEG(1,51))); // dynamics_eq(x2,x1,u1)
      val += w.SEG(2,31).dot(dynamics_eq::eval(param, x.SEG(2,6),x.SEG(2,4),x.SEG(1,52))); // dynamics_eq(x3,x2,u2)
      val += w.SEG(2,33).dot(dynamics_eq::eval(param, x.SEG(2,8),x.SEG(2,6),x.SEG(1,53))); // dynamics_eq(x4,x3,u3)
      val += w.SEG(2,35).dot(dynamics_eq::eval(param, x.SEG(2,10),x.SEG(2,8),x.SEG(1,54))); // dynamics_eq(x5,x4,u4)
      val += w.SEG(2,37).dot(dynamics_eq::eval(param, x.SEG(2,12),x.SEG(2,10),x.SEG(1,55))); // dynamics_eq(x6,x5,u5)
      val += w.SEG(2,39).dot(dynamics_eq::eval(param, x.SEG(2,14),x.SEG(2,12),x.SEG(1,56))); // dynamics_eq(x7,x6,u6)
      val += w.SEG(2,41).dot(dynamics_eq::eval(param, x.SEG(2,16),x.SEG(2,14),x.SEG(1,57))); // dynamics_eq(x8,x7,u7)
      val += w.SEG(2,43).dot(dynamics_eq::eval(param, x.SEG(2,18),x.SEG(2,16),x.SEG(1,58))); // dynamics_eq(x9,x8,u8)
      val += w.SEG(2,45).dot(dynamics_eq::eval(param, x.SEG(2,20),x.SEG(2,18),x.SEG(1,59))); // dynamics_eq(x10,x9,u9)
      val += w.SEG(2,47).dot(dynamics_eq::eval(param, x.SEG(2,22),x.SEG(2,20),x.SEG(1,60))); // dynamics_eq(x11,x10,u10)
      val += w.SEG(2,49).dot(dynamics_eq::eval(param, x.SEG(2,24),x.SEG(2,22),x.SEG(1,61))); // dynamics_eq(x12,x11,u11)
      val += w.SEG(2,51).dot(dynamics_eq::eval(param, x.SEG(2,26),x.SEG(2,24),x.SEG(1,62))); // dynamics_eq(x13,x12,u12)
      val += w.SEG(2,53).dot(dynamics_eq::eval(param, x.SEG(2,28),x.SEG(2,26),x.SEG(1,63))); // dynamics_eq(x14,x13,u13)
      val += w.SEG(2,55).dot(dynamics_eq::eval(param, x.SEG(2,30),x.SEG(2,28),x.SEG(1,64))); // dynamics_eq(x15,x14,u14)
      val += w.SEG(2,57).dot(dynamics_eq::eval(param, x.SEG(2,32),x.SEG(2,30),x.SEG(1,65))); // dynamics_eq(x16,x15,u15)
      val += w.SEG(2,59).dot(dynamics_eq::eval(param, x.SEG(2,34),x.SEG(2,32),x.SEG(1,66))); // dynamics_eq(x17,x16,u16)
      val += w.SEG(2,61).dot(dynamics_eq::eval(param, x.SEG(2,36),x.SEG(2,34),x.SEG(1,67))); // dynamics_eq(x18,x17,u17)
      val += w.SEG(2,63).dot(dynamics_eq::eval(param, x.SEG(2,38),x.SEG(2,36),x.SEG(1,68))); // dynamics_eq(x19,x18,u18)
      val += w.SEG(2,65).dot(dynamics_eq::eval(param, x.SEG(2,40),x.SEG(2,38),x.SEG(1,69))); // dynamics_eq(x20,x19,u19)
      val += w.SEG(2,67).dot(dynamics_eq::eval(param, x.SEG(2,42),x.SEG(2,40),x.SEG(1,70))); // dynamics_eq(x21,x20,u20)
      val += w.SEG(2,69).dot(dynamics_eq::eval(param, x.SEG(2,44),x.SEG(2,42),x.SEG(1,71))); // dynamics_eq(x22,x21,u21)
      val += w.SEG(2,71).dot(dynamics_eq::eval(param, x.SEG(2,46),x.SEG(2,44),x.SEG(1,72))); // dynamics_eq(x23,x22,u22)
      val += w.SEG(2,73).dot(dynamics_eq::eval(param, x.SEG(2,48),x.SEG(2,46),x.SEG(1,73))); // dynamics_eq(x24,x23,u23)
      val += w.SEG(2,75).dot(dynamics_ss::eval(param, x.SEG(2,74),x.SEG(1,76))); // dynamics_ss(xss,uss)
      return val;
    };

    /**
     * Compute the gradient of the weighted sum
     */
    static constexpr seqinfo grad_seq[173] = {
      {4,2},{51,1},
      {48,2},{74,2},
      {0,2},{50,1},{74,2},{76,1},
      {2,2},{51,1},{74,2},{76,1},
      {4,2},{52,1},{74,2},{76,1},
      {6,2},{53,1},{74,2},{76,1},
      {8,2},{54,1},{74,2},{76,1},
      {10,2},{55,1},{74,2},{76,1},
      {12,2},{56,1},{74,2},{76,1},
      {14,2},{57,1},{74,2},{76,1},
      {16,2},{58,1},{74,2},{76,1},
      {18,2},{59,1},{74,2},{76,1},
      {20,2},{60,1},{74,2},{76,1},
      {22,2},{61,1},{74,2},{76,1},
      {24,2},{62,1},{74,2},{76,1},
      {26,2},{63,1},{74,2},{76,1},
      {28,2},{64,1},{74,2},{76,1},
      {30,2},{65,1},{74,2},{76,1},
      {32,2},{66,1},{74,2},{76,1},
      {34,2},{67,1},{74,2},{76,1},
      {36,2},{68,1},{74,2},{76,1},
      {38,2},{69,1},{74,2},{76,1},
      {40,2},{70,1},{74,2},{76,1},
      {42,2},{71,1},{74,2},{76,1},
      {44,2},{72,1},{74,2},{76,1},
      {46,2},{73,1},{74,2},{76,1},
      {0,2},{50,1},
      {4,2},{2,2},{51,1},
      {6,2},{4,2},{52,1},
      {8,2},{6,2},{53,1},
      {10,2},{8,2},{54,1},
      {12,2},{10,2},{55,1},
      {14,2},{12,2},{56,1},
      {16,2},{14,2},{57,1},
      {18,2},{16,2},{58,1},
      {20,2},{18,2},{59,1},
      {22,2},{20,2},{60,1},
      {24,2},{22,2},{61,1},
      {26,2},{24,2},{62,1},
      {28,2},{26,2},{63,1},
      {30,2},{28,2},{64,1},
      {32,2},{30,2},{65,1},
      {34,2},{32,2},{66,1},
      {36,2},{34,2},{67,1},
      {38,2},{36,2},{68,1},
      {40,2},{38,2},{69,1},
      {42,2},{40,2},{70,1},
      {44,2},{42,2},{71,1},
      {46,2},{44,2},{72,1},
      {48,2},{46,2},{73,1},
      {74,2},{76,1}};
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient)
    {
      gradient.array() = 0;
      scalar_t val = 0;
      accGrad(val, gradient, grad_seq+0, 2, w.SEG(2,0), dynamics_0::jac(param, x.SEG(2,4),x.SEG(1,51))); // {std(call)}
      accGrad(val, gradient, grad_seq+2, 2, w.SEG(1,2), terminal_cost::jac(param, x.SEG(2,48),x.SEG(2,74))); // {std(call)}
      accGrad(val, gradient, grad_seq+4, 4, w.SEG(1,3), stage_cost::jac(param, x.SEG(2,0),x.SEG(1,50),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+8, 4, w.SEG(1,4), stage_cost::jac(param, x.SEG(2,2),x.SEG(1,51),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+12, 4, w.SEG(1,5), stage_cost::jac(param, x.SEG(2,4),x.SEG(1,52),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+16, 4, w.SEG(1,6), stage_cost::jac(param, x.SEG(2,6),x.SEG(1,53),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+20, 4, w.SEG(1,7), stage_cost::jac(param, x.SEG(2,8),x.SEG(1,54),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+24, 4, w.SEG(1,8), stage_cost::jac(param, x.SEG(2,10),x.SEG(1,55),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+28, 4, w.SEG(1,9), stage_cost::jac(param, x.SEG(2,12),x.SEG(1,56),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+32, 4, w.SEG(1,10), stage_cost::jac(param, x.SEG(2,14),x.SEG(1,57),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+36, 4, w.SEG(1,11), stage_cost::jac(param, x.SEG(2,16),x.SEG(1,58),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+40, 4, w.SEG(1,12), stage_cost::jac(param, x.SEG(2,18),x.SEG(1,59),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+44, 4, w.SEG(1,13), stage_cost::jac(param, x.SEG(2,20),x.SEG(1,60),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+48, 4, w.SEG(1,14), stage_cost::jac(param, x.SEG(2,22),x.SEG(1,61),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+52, 4, w.SEG(1,15), stage_cost::jac(param, x.SEG(2,24),x.SEG(1,62),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+56, 4, w.SEG(1,16), stage_cost::jac(param, x.SEG(2,26),x.SEG(1,63),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+60, 4, w.SEG(1,17), stage_cost::jac(param, x.SEG(2,28),x.SEG(1,64),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+64, 4, w.SEG(1,18), stage_cost::jac(param, x.SEG(2,30),x.SEG(1,65),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+68, 4, w.SEG(1,19), stage_cost::jac(param, x.SEG(2,32),x.SEG(1,66),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+72, 4, w.SEG(1,20), stage_cost::jac(param, x.SEG(2,34),x.SEG(1,67),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+76, 4, w.SEG(1,21), stage_cost::jac(param, x.SEG(2,36),x.SEG(1,68),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+80, 4, w.SEG(1,22), stage_cost::jac(param, x.SEG(2,38),x.SEG(1,69),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+84, 4, w.SEG(1,23), stage_cost::jac(param, x.SEG(2,40),x.SEG(1,70),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+88, 4, w.SEG(1,24), stage_cost::jac(param, x.SEG(2,42),x.SEG(1,71),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+92, 4, w.SEG(1,25), stage_cost::jac(param, x.SEG(2,44),x.SEG(1,72),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+96, 4, w.SEG(1,26), stage_cost::jac(param, x.SEG(2,46),x.SEG(1,73),x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      accGrad(val, gradient, grad_seq+100, 2, w.SEG(2,27), dynamics_0::jac(param, x.SEG(2,0),x.SEG(1,50))); // {std(call)}
      accGrad(val, gradient, grad_seq+102, 3, w.SEG(2,29), dynamics_eq::jac(param, x.SEG(2,4),x.SEG(2,2),x.SEG(1,51))); // {std(call)}
      accGrad(val, gradient, grad_seq+105, 3, w.SEG(2,31), dynamics_eq::jac(param, x.SEG(2,6),x.SEG(2,4),x.SEG(1,52))); // {std(call)}
      accGrad(val, gradient, grad_seq+108, 3, w.SEG(2,33), dynamics_eq::jac(param, x.SEG(2,8),x.SEG(2,6),x.SEG(1,53))); // {std(call)}
      accGrad(val, gradient, grad_seq+111, 3, w.SEG(2,35), dynamics_eq::jac(param, x.SEG(2,10),x.SEG(2,8),x.SEG(1,54))); // {std(call)}
      accGrad(val, gradient, grad_seq+114, 3, w.SEG(2,37), dynamics_eq::jac(param, x.SEG(2,12),x.SEG(2,10),x.SEG(1,55))); // {std(call)}
      accGrad(val, gradient, grad_seq+117, 3, w.SEG(2,39), dynamics_eq::jac(param, x.SEG(2,14),x.SEG(2,12),x.SEG(1,56))); // {std(call)}
      accGrad(val, gradient, grad_seq+120, 3, w.SEG(2,41), dynamics_eq::jac(param, x.SEG(2,16),x.SEG(2,14),x.SEG(1,57))); // {std(call)}
      accGrad(val, gradient, grad_seq+123, 3, w.SEG(2,43), dynamics_eq::jac(param, x.SEG(2,18),x.SEG(2,16),x.SEG(1,58))); // {std(call)}
      accGrad(val, gradient, grad_seq+126, 3, w.SEG(2,45), dynamics_eq::jac(param, x.SEG(2,20),x.SEG(2,18),x.SEG(1,59))); // {std(call)}
      accGrad(val, gradient, grad_seq+129, 3, w.SEG(2,47), dynamics_eq::jac(param, x.SEG(2,22),x.SEG(2,20),x.SEG(1,60))); // {std(call)}
      accGrad(val, gradient, grad_seq+132, 3, w.SEG(2,49), dynamics_eq::jac(param, x.SEG(2,24),x.SEG(2,22),x.SEG(1,61))); // {std(call)}
      accGrad(val, gradient, grad_seq+135, 3, w.SEG(2,51), dynamics_eq::jac(param, x.SEG(2,26),x.SEG(2,24),x.SEG(1,62))); // {std(call)}
      accGrad(val, gradient, grad_seq+138, 3, w.SEG(2,53), dynamics_eq::jac(param, x.SEG(2,28),x.SEG(2,26),x.SEG(1,63))); // {std(call)}
      accGrad(val, gradient, grad_seq+141, 3, w.SEG(2,55), dynamics_eq::jac(param, x.SEG(2,30),x.SEG(2,28),x.SEG(1,64))); // {std(call)}
      accGrad(val, gradient, grad_seq+144, 3, w.SEG(2,57), dynamics_eq::jac(param, x.SEG(2,32),x.SEG(2,30),x.SEG(1,65))); // {std(call)}
      accGrad(val, gradient, grad_seq+147, 3, w.SEG(2,59), dynamics_eq::jac(param, x.SEG(2,34),x.SEG(2,32),x.SEG(1,66))); // {std(call)}
      accGrad(val, gradient, grad_seq+150, 3, w.SEG(2,61), dynamics_eq::jac(param, x.SEG(2,36),x.SEG(2,34),x.SEG(1,67))); // {std(call)}
      accGrad(val, gradient, grad_seq+153, 3, w.SEG(2,63), dynamics_eq::jac(param, x.SEG(2,38),x.SEG(2,36),x.SEG(1,68))); // {std(call)}
      accGrad(val, gradient, grad_seq+156, 3, w.SEG(2,65), dynamics_eq::jac(param, x.SEG(2,40),x.SEG(2,38),x.SEG(1,69))); // {std(call)}
      accGrad(val, gradient, grad_seq+159, 3, w.SEG(2,67), dynamics_eq::jac(param, x.SEG(2,42),x.SEG(2,40),x.SEG(1,70))); // {std(call)}
      accGrad(val, gradient, grad_seq+162, 3, w.SEG(2,69), dynamics_eq::jac(param, x.SEG(2,44),x.SEG(2,42),x.SEG(1,71))); // {std(call)}
      accGrad(val, gradient, grad_seq+165, 3, w.SEG(2,71), dynamics_eq::jac(param, x.SEG(2,46),x.SEG(2,44),x.SEG(1,72))); // {std(call)}
      accGrad(val, gradient, grad_seq+168, 3, w.SEG(2,73), dynamics_eq::jac(param, x.SEG(2,48),x.SEG(2,46),x.SEG(1,73))); // {std(call)}
      accGrad(val, gradient, grad_seq+171, 2, w.SEG(2,75), dynamics_ss::jac(param, x.SEG(2,74),x.SEG(1,76))); // {std(call)}
      return val;
    };

    /**
     * Initialize the hessian of the function
     */
    static void initialize_hessian(Eigen::SparseMatrix<scalar_t> &H)
    {
      H.resize(77,77);
      H.reserve(945);
      typedef Eigen::Triplet<scalar_t> T;
      std::array<T,945> tripletList = {T{0,1,1},{0,50,1},{0,74,1},{0,75,1},{0,76,1},{1,0,1},{1,1,1},{1,50,1},{1,74,1},{1,75,1},{1,76,1},{2,2,1},{2,3,1},{2,4,1},{2,5,1},{2,51,1},{2,74,1},{2,75,1},{2,76,1},{3,2,1},{3,3,1},{3,4,1},{3,5,1},{3,51,1},{3,74,1},{3,75,1},{3,76,1},{4,2,1},{4,3,1},{4,4,1},{4,5,1},{4,6,1},{4,7,1},{4,51,1},{4,52,1},{4,74,1},{4,75,1},{4,76,1},{5,2,1},{5,3,1},{5,4,1},{5,5,1},{5,6,1},{5,7,1},{5,51,1},{5,52,1},{5,74,1},{5,75,1},{5,76,1},{6,4,1},{6,5,1},{6,6,1},{6,7,1},{6,8,1},{6,9,1},{6,52,1},{6,53,1},{6,74,1},{6,75,1},{6,76,1},{7,4,1},{7,5,1},{7,6,1},{7,7,1},{7,8,1},{7,9,1},{7,52,1},{7,53,1},{7,74,1},{7,75,1},{7,76,1},{8,6,1},{8,7,1},{8,8,1},{8,9,1},{8,10,1},{8,11,1},{8,53,1},{8,54,1},{8,74,1},{8,75,1},{8,76,1},{9,6,1},{9,7,1},{9,8,1},{9,9,1},{9,10,1},{9,11,1},{9,53,1},{9,54,1},{9,74,1},{9,75,1},{9,76,1},{10,8,1},{10,9,1},{10,10,1},{10,11,1},{10,12,1},{10,13,1},{10,54,1},{10,55,1},{10,74,1},{10,75,1},{10,76,1},{11,8,1},{11,9,1},{11,10,1},{11,11,1},{11,12,1},{11,13,1},{11,54,1},{11,55,1},{11,74,1},{11,75,1},{11,76,1},{12,10,1},{12,11,1},{12,12,1},{12,13,1},{12,14,1},{12,15,1},{12,55,1},{12,56,1},{12,74,1},{12,75,1},{12,76,1},{13,10,1},{13,11,1},{13,12,1},{13,13,1},{13,14,1},{13,15,1},{13,55,1},{13,56,1},{13,74,1},{13,75,1},{13,76,1},{14,12,1},{14,13,1},{14,14,1},{14,15,1},{14,16,1},{14,17,1},{14,56,1},{14,57,1},{14,74,1},{14,75,1},{14,76,1},{15,12,1},{15,13,1},{15,14,1},{15,15,1},{15,16,1},{15,17,1},{15,56,1},{15,57,1},{15,74,1},{15,75,1},{15,76,1},{16,14,1},{16,15,1},{16,16,1},{16,17,1},{16,18,1},{16,19,1},{16,57,1},{16,58,1},{16,74,1},{16,75,1},{16,76,1},{17,14,1},{17,15,1},{17,16,1},{17,17,1},{17,18,1},{17,19,1},{17,57,1},{17,58,1},{17,74,1},{17,75,1},{17,76,1},{18,16,1},{18,17,1},{18,18,1},{18,19,1},{18,20,1},{18,21,1},{18,58,1},{18,59,1},{18,74,1},{18,75,1},{18,76,1},{19,16,1},{19,17,1},{19,18,1},{19,19,1},{19,20,1},{19,21,1},{19,58,1},{19,59,1},{19,74,1},{19,75,1},{19,76,1},{20,18,1},{20,19,1},{20,20,1},{20,21,1},{20,22,1},{20,23,1},{20,59,1},{20,60,1},{20,74,1},{20,75,1},{20,76,1},{21,18,1},{21,19,1},{21,20,1},{21,21,1},{21,22,1},{21,23,1},{21,59,1},{21,60,1},{21,74,1},{21,75,1},{21,76,1},{22,20,1},{22,21,1},{22,22,1},{22,23,1},{22,24,1},{22,25,1},{22,60,1},{22,61,1},{22,74,1},{22,75,1},{22,76,1},{23,20,1},{23,21,1},{23,22,1},{23,23,1},{23,24,1},{23,25,1},{23,60,1},{23,61,1},{23,74,1},{23,75,1},{23,76,1},{24,22,1},{24,23,1},{24,24,1},{24,25,1},{24,26,1},{24,27,1},{24,61,1},{24,62,1},{24,74,1},{24,75,1},{24,76,1},{25,22,1},{25,23,1},{25,24,1},{25,25,1},{25,26,1},{25,27,1},{25,61,1},{25,62,1},{25,74,1},{25,75,1},{25,76,1},{26,24,1},{26,25,1},{26,26,1},{26,27,1},{26,28,1},{26,29,1},{26,62,1},{26,63,1},{26,74,1},{26,75,1},{26,76,1},{27,24,1},{27,25,1},{27,26,1},{27,27,1},{27,28,1},{27,29,1},{27,62,1},{27,63,1},{27,74,1},{27,75,1},{27,76,1},{28,26,1},{28,27,1},{28,28,1},{28,29,1},{28,30,1},{28,31,1},{28,63,1},{28,64,1},{28,74,1},{28,75,1},{28,76,1},{29,26,1},{29,27,1},{29,28,1},{29,29,1},{29,30,1},{29,31,1},{29,63,1},{29,64,1},{29,74,1},{29,75,1},{29,76,1},{30,28,1},{30,29,1},{30,30,1},{30,31,1},{30,32,1},{30,33,1},{30,64,1},{30,65,1},{30,74,1},{30,75,1},{30,76,1},{31,28,1},{31,29,1},{31,30,1},{31,31,1},{31,32,1},{31,33,1},{31,64,1},{31,65,1},{31,74,1},{31,75,1},{31,76,1},{32,30,1},{32,31,1},{32,32,1},{32,33,1},{32,34,1},{32,35,1},{32,65,1},{32,66,1},{32,74,1},{32,75,1},{32,76,1},{33,30,1},{33,31,1},{33,32,1},{33,33,1},{33,34,1},{33,35,1},{33,65,1},{33,66,1},{33,74,1},{33,75,1},{33,76,1},{34,32,1},{34,33,1},{34,34,1},{34,35,1},{34,36,1},{34,37,1},{34,66,1},{34,67,1},{34,74,1},{34,75,1},{34,76,1},{35,32,1},{35,33,1},{35,34,1},{35,35,1},{35,36,1},{35,37,1},{35,66,1},{35,67,1},{35,74,1},{35,75,1},{35,76,1},{36,34,1},{36,35,1},{36,36,1},{36,37,1},{36,38,1},{36,39,1},{36,67,1},{36,68,1},{36,74,1},{36,75,1},{36,76,1},{37,34,1},{37,35,1},{37,36,1},{37,37,1},{37,38,1},{37,39,1},{37,67,1},{37,68,1},{37,74,1},{37,75,1},{37,76,1},{38,36,1},{38,37,1},{38,38,1},{38,39,1},{38,40,1},{38,41,1},{38,68,1},{38,69,1},{38,74,1},{38,75,1},{38,76,1},{39,36,1},{39,37,1},{39,38,1},{39,39,1},{39,40,1},{39,41,1},{39,68,1},{39,69,1},{39,74,1},{39,75,1},{39,76,1},{40,38,1},{40,39,1},{40,40,1},{40,41,1},{40,42,1},{40,43,1},{40,69,1},{40,70,1},{40,74,1},{40,75,1},{40,76,1},{41,38,1},{41,39,1},{41,40,1},{41,41,1},{41,42,1},{41,43,1},{41,69,1},{41,70,1},{41,74,1},{41,75,1},{41,76,1},{42,40,1},{42,41,1},{42,42,1},{42,43,1},{42,44,1},{42,45,1},{42,70,1},{42,71,1},{42,74,1},{42,75,1},{42,76,1},{43,40,1},{43,41,1},{43,42,1},{43,43,1},{43,44,1},{43,45,1},{43,70,1},{43,71,1},{43,74,1},{43,75,1},{43,76,1},{44,42,1},{44,43,1},{44,44,1},{44,45,1},{44,46,1},{44,47,1},{44,71,1},{44,72,1},{44,74,1},{44,75,1},{44,76,1},{45,42,1},{45,43,1},{45,44,1},{45,45,1},{45,46,1},{45,47,1},{45,71,1},{45,72,1},{45,74,1},{45,75,1},{45,76,1},{46,44,1},{46,45,1},{46,46,1},{46,47,1},{46,48,1},{46,49,1},{46,72,1},{46,73,1},{46,74,1},{46,75,1},{46,76,1},{47,44,1},{47,45,1},{47,46,1},{47,47,1},{47,48,1},{47,49,1},{47,72,1},{47,73,1},{47,74,1},{47,75,1},{47,76,1},{48,46,1},{48,47,1},{48,48,1},{48,49,1},{48,73,1},{48,74,1},{48,75,1},{49,46,1},{49,47,1},{49,48,1},{49,49,1},{49,73,1},{49,74,1},{49,75,1},{50,0,1},{50,1,1},{50,50,1},{50,74,1},{50,75,1},{50,76,1},{51,2,1},{51,3,1},{51,4,1},{51,5,1},{51,51,1},{51,74,1},{51,75,1},{51,76,1},{52,4,1},{52,5,1},{52,6,1},{52,7,1},{52,52,1},{52,74,1},{52,75,1},{52,76,1},{53,6,1},{53,7,1},{53,8,1},{53,9,1},{53,53,1},{53,74,1},{53,75,1},{53,76,1},{54,8,1},{54,9,1},{54,10,1},{54,11,1},{54,54,1},{54,74,1},{54,75,1},{54,76,1},{55,10,1},{55,11,1},{55,12,1},{55,13,1},{55,55,1},{55,74,1},{55,75,1},{55,76,1},{56,12,1},{56,13,1},{56,14,1},{56,15,1},{56,56,1},{56,74,1},{56,75,1},{56,76,1},{57,14,1},{57,15,1},{57,16,1},{57,17,1},{57,57,1},{57,74,1},{57,75,1},{57,76,1},{58,16,1},{58,17,1},{58,18,1},{58,19,1},{58,58,1},{58,74,1},{58,75,1},{58,76,1},{59,18,1},{59,19,1},{59,20,1},{59,21,1},{59,59,1},{59,74,1},{59,75,1},{59,76,1},{60,20,1},{60,21,1},{60,22,1},{60,23,1},{60,60,1},{60,74,1},{60,75,1},{60,76,1},{61,22,1},{61,23,1},{61,24,1},{61,25,1},{61,61,1},{61,74,1},{61,75,1},{61,76,1},{62,24,1},{62,25,1},{62,26,1},{62,27,1},{62,62,1},{62,74,1},{62,75,1},{62,76,1},{63,26,1},{63,27,1},{63,28,1},{63,29,1},{63,63,1},{63,74,1},{63,75,1},{63,76,1},{64,28,1},{64,29,1},{64,30,1},{64,31,1},{64,64,1},{64,74,1},{64,75,1},{64,76,1},{65,30,1},{65,31,1},{65,32,1},{65,33,1},{65,65,1},{65,74,1},{65,75,1},{65,76,1},{66,32,1},{66,33,1},{66,34,1},{66,35,1},{66,66,1},{66,74,1},{66,75,1},{66,76,1},{67,34,1},{67,35,1},{67,36,1},{67,37,1},{67,67,1},{67,74,1},{67,75,1},{67,76,1},{68,36,1},{68,37,1},{68,38,1},{68,39,1},{68,68,1},{68,74,1},{68,75,1},{68,76,1},{69,38,1},{69,39,1},{69,40,1},{69,41,1},{69,69,1},{69,74,1},{69,75,1},{69,76,1},{70,40,1},{70,41,1},{70,42,1},{70,43,1},{70,70,1},{70,74,1},{70,75,1},{70,76,1},{71,42,1},{71,43,1},{71,44,1},{71,45,1},{71,71,1},{71,74,1},{71,75,1},{71,76,1},{72,44,1},{72,45,1},{72,46,1},{72,47,1},{72,72,1},{72,74,1},{72,75,1},{72,76,1},{73,46,1},{73,47,1},{73,48,1},{73,49,1},{73,73,1},{73,74,1},{73,75,1},{73,76,1},{74,0,1},{74,1,1},{74,2,1},{74,3,1},{74,4,1},{74,5,1},{74,6,1},{74,7,1},{74,8,1},{74,9,1},{74,10,1},{74,11,1},{74,12,1},{74,13,1},{74,14,1},{74,15,1},{74,16,1},{74,17,1},{74,18,1},{74,19,1},{74,20,1},{74,21,1},{74,22,1},{74,23,1},{74,24,1},{74,25,1},{74,26,1},{74,27,1},{74,28,1},{74,29,1},{74,30,1},{74,31,1},{74,32,1},{74,33,1},{74,34,1},{74,35,1},{74,36,1},{74,37,1},{74,38,1},{74,39,1},{74,40,1},{74,41,1},{74,42,1},{74,43,1},{74,44,1},{74,45,1},{74,46,1},{74,47,1},{74,48,1},{74,49,1},{74,50,1},{74,51,1},{74,52,1},{74,53,1},{74,54,1},{74,55,1},{74,56,1},{74,57,1},{74,58,1},{74,59,1},{74,60,1},{74,61,1},{74,62,1},{74,63,1},{74,64,1},{74,65,1},{74,66,1},{74,67,1},{74,68,1},{74,69,1},{74,70,1},{74,71,1},{74,72,1},{74,73,1},{74,74,1},{74,75,1},{74,76,1},{75,0,1},{75,1,1},{75,2,1},{75,3,1},{75,4,1},{75,5,1},{75,6,1},{75,7,1},{75,8,1},{75,9,1},{75,10,1},{75,11,1},{75,12,1},{75,13,1},{75,14,1},{75,15,1},{75,16,1},{75,17,1},{75,18,1},{75,19,1},{75,20,1},{75,21,1},{75,22,1},{75,23,1},{75,24,1},{75,25,1},{75,26,1},{75,27,1},{75,28,1},{75,29,1},{75,30,1},{75,31,1},{75,32,1},{75,33,1},{75,34,1},{75,35,1},{75,36,1},{75,37,1},{75,38,1},{75,39,1},{75,40,1},{75,41,1},{75,42,1},{75,43,1},{75,44,1},{75,45,1},{75,46,1},{75,47,1},{75,48,1},{75,49,1},{75,50,1},{75,51,1},{75,52,1},{75,53,1},{75,54,1},{75,55,1},{75,56,1},{75,57,1},{75,58,1},{75,59,1},{75,60,1},{75,61,1},{75,62,1},{75,63,1},{75,64,1},{75,65,1},{75,66,1},{75,67,1},{75,68,1},{75,69,1},{75,70,1},{75,71,1},{75,72,1},{75,73,1},{75,74,1},{75,75,1},{75,76,1},{76,0,1},{76,1,1},{76,2,1},{76,3,1},{76,4,1},{76,5,1},{76,6,1},{76,7,1},{76,8,1},{76,9,1},{76,10,1},{76,11,1},{76,12,1},{76,13,1},{76,14,1},{76,15,1},{76,16,1},{76,17,1},{76,18,1},{76,19,1},{76,20,1},{76,21,1},{76,22,1},{76,23,1},{76,24,1},{76,25,1},{76,26,1},{76,27,1},{76,28,1},{76,29,1},{76,30,1},{76,31,1},{76,32,1},{76,33,1},{76,34,1},{76,35,1},{76,36,1},{76,37,1},{76,38,1},{76,39,1},{76,40,1},{76,41,1},{76,42,1},{76,43,1},{76,44,1},{76,45,1},{76,46,1},{76,47,1},{76,50,1},{76,51,1},{76,52,1},{76,53,1},{76,54,1},{76,55,1},{76,56,1},{76,57,1},{76,58,1},{76,59,1},{76,60,1},{76,61,1},{76,62,1},{76,63,1},{76,64,1},{76,65,1},{76,66,1},{76,67,1},{76,68,1},{76,69,1},{76,70,1},{76,71,1},{76,72,1},{76,73,1},{76,74,1},{76,75,1},{76,76,1}};
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

    static constexpr seqinfo hessian_seq[1070] = {
      {30,2},{34,1},{41,2},{45,1},{534,3},{30,2},{34,1},{41,2},{45,1},{534,3},
      {514,2},{517,2},{521,2},{524,2},{764,2},{790,2},{841,2},{867,2},
      {0,12},{526,6},{716,2},{766,1},{790,5},{843,1},{867,5},{918,1},{942,3},
      {12,2},{16,6},{24,4},{532,2},{536,4},{718,2},{767,1},{790,3},{795,2},{844,1},{867,3},{872,2},{919,1},{942,3},
      {30,2},{35,4},{41,2},{46,4},{540,2},{544,4},{720,2},{768,1},{790,3},{797,2},{845,1},{867,3},{874,2},{920,1},{942,3},
      {52,2},{57,4},{63,2},{68,4},{548,2},{552,4},{722,2},{769,1},{790,3},{799,2},{846,1},{867,3},{876,2},{921,1},{942,3},
      {74,2},{79,4},{85,2},{90,4},{556,2},{560,4},{724,2},{770,1},{790,3},{801,2},{847,1},{867,3},{878,2},{922,1},{942,3},
      {96,2},{101,4},{107,2},{112,4},{564,2},{568,4},{726,2},{771,1},{790,3},{803,2},{848,1},{867,3},{880,2},{923,1},{942,3},
      {118,2},{123,4},{129,2},{134,4},{572,2},{576,4},{728,2},{772,1},{790,3},{805,2},{849,1},{867,3},{882,2},{924,1},{942,3},
      {140,2},{145,4},{151,2},{156,4},{580,2},{584,4},{730,2},{773,1},{790,3},{807,2},{850,1},{867,3},{884,2},{925,1},{942,3},
      {162,2},{167,4},{173,2},{178,4},{588,2},{592,4},{732,2},{774,1},{790,3},{809,2},{851,1},{867,3},{886,2},{926,1},{942,3},
      {184,2},{189,4},{195,2},{200,4},{596,2},{600,4},{734,2},{775,1},{790,3},{811,2},{852,1},{867,3},{888,2},{927,1},{942,3},
      {206,2},{211,4},{217,2},{222,4},{604,2},{608,4},{736,2},{776,1},{790,3},{813,2},{853,1},{867,3},{890,2},{928,1},{942,3},
      {228,2},{233,4},{239,2},{244,4},{612,2},{616,4},{738,2},{777,1},{790,3},{815,2},{854,1},{867,3},{892,2},{929,1},{942,3},
      {250,2},{255,4},{261,2},{266,4},{620,2},{624,4},{740,2},{778,1},{790,3},{817,2},{855,1},{867,3},{894,2},{930,1},{942,3},
      {272,2},{277,4},{283,2},{288,4},{628,2},{632,4},{742,2},{779,1},{790,3},{819,2},{856,1},{867,3},{896,2},{931,1},{942,3},
      {294,2},{299,4},{305,2},{310,4},{636,2},{640,4},{744,2},{780,1},{790,3},{821,2},{857,1},{867,3},{898,2},{932,1},{942,3},
      {316,2},{321,4},{327,2},{332,4},{644,2},{648,4},{746,2},{781,1},{790,3},{823,2},{858,1},{867,3},{900,2},{933,1},{942,3},
      {338,2},{343,4},{349,2},{354,4},{652,2},{656,4},{748,2},{782,1},{790,3},{825,2},{859,1},{867,3},{902,2},{934,1},{942,3},
      {360,2},{365,4},{371,2},{376,4},{660,2},{664,4},{750,2},{783,1},{790,3},{827,2},{860,1},{867,3},{904,2},{935,1},{942,3},
      {382,2},{387,4},{393,2},{398,4},{668,2},{672,4},{752,2},{784,1},{790,3},{829,2},{861,1},{867,3},{906,2},{936,1},{942,3},
      {404,2},{409,4},{415,2},{420,4},{676,2},{680,4},{754,2},{785,1},{790,3},{831,2},{862,1},{867,3},{908,2},{937,1},{942,3},
      {426,2},{431,4},{437,2},{442,4},{684,2},{688,4},{756,2},{786,1},{790,3},{833,2},{863,1},{867,3},{910,2},{938,1},{942,3},
      {448,2},{453,4},{459,2},{464,4},{692,2},{696,4},{758,2},{787,1},{790,3},{835,2},{864,1},{867,3},{912,2},{939,1},{942,3},
      {470,2},{475,4},{481,2},{486,4},{700,2},{704,4},{760,2},{788,1},{790,3},{837,2},{865,1},{867,3},{914,2},{940,1},{942,3},
      {492,2},{497,4},{503,2},{508,4},{708,2},{712,4},{762,2},{789,4},{839,2},{866,4},{916,2},{941,4},
      {0,3},{6,3},{526,3},{0,3},{6,3},{526,3},
      {30,2},{28,2},{34,1},{41,2},{39,2},{45,1},{14,2},{12,2},{16,1},{22,2},{20,2},{24,1},{534,2},{532,2},{536,1},{30,2},{28,2},{34,1},{41,2},{39,2},{45,1},{14,2},{12,2},{16,1},{22,2},{20,2},{24,1},{534,2},{532,2},{536,1},
      {52,2},{50,2},{56,1},{63,2},{61,2},{67,1},{32,2},{30,2},{35,1},{43,2},{41,2},{46,1},{542,2},{540,2},{544,1},{52,2},{50,2},{56,1},{63,2},{61,2},{67,1},{32,2},{30,2},{35,1},{43,2},{41,2},{46,1},{542,2},{540,2},{544,1},
      {74,2},{72,2},{78,1},{85,2},{83,2},{89,1},{54,2},{52,2},{57,1},{65,2},{63,2},{68,1},{550,2},{548,2},{552,1},{74,2},{72,2},{78,1},{85,2},{83,2},{89,1},{54,2},{52,2},{57,1},{65,2},{63,2},{68,1},{550,2},{548,2},{552,1},
      {96,2},{94,2},{100,1},{107,2},{105,2},{111,1},{76,2},{74,2},{79,1},{87,2},{85,2},{90,1},{558,2},{556,2},{560,1},{96,2},{94,2},{100,1},{107,2},{105,2},{111,1},{76,2},{74,2},{79,1},{87,2},{85,2},{90,1},{558,2},{556,2},{560,1},
      {118,2},{116,2},{122,1},{129,2},{127,2},{133,1},{98,2},{96,2},{101,1},{109,2},{107,2},{112,1},{566,2},{564,2},{568,1},{118,2},{116,2},{122,1},{129,2},{127,2},{133,1},{98,2},{96,2},{101,1},{109,2},{107,2},{112,1},{566,2},{564,2},{568,1},
      {140,2},{138,2},{144,1},{151,2},{149,2},{155,1},{120,2},{118,2},{123,1},{131,2},{129,2},{134,1},{574,2},{572,2},{576,1},{140,2},{138,2},{144,1},{151,2},{149,2},{155,1},{120,2},{118,2},{123,1},{131,2},{129,2},{134,1},{574,2},{572,2},{576,1},
      {162,2},{160,2},{166,1},{173,2},{171,2},{177,1},{142,2},{140,2},{145,1},{153,2},{151,2},{156,1},{582,2},{580,2},{584,1},{162,2},{160,2},{166,1},{173,2},{171,2},{177,1},{142,2},{140,2},{145,1},{153,2},{151,2},{156,1},{582,2},{580,2},{584,1},
      {184,2},{182,2},{188,1},{195,2},{193,2},{199,1},{164,2},{162,2},{167,1},{175,2},{173,2},{178,1},{590,2},{588,2},{592,1},{184,2},{182,2},{188,1},{195,2},{193,2},{199,1},{164,2},{162,2},{167,1},{175,2},{173,2},{178,1},{590,2},{588,2},{592,1},
      {206,2},{204,2},{210,1},{217,2},{215,2},{221,1},{186,2},{184,2},{189,1},{197,2},{195,2},{200,1},{598,2},{596,2},{600,1},{206,2},{204,2},{210,1},{217,2},{215,2},{221,1},{186,2},{184,2},{189,1},{197,2},{195,2},{200,1},{598,2},{596,2},{600,1},
      {228,2},{226,2},{232,1},{239,2},{237,2},{243,1},{208,2},{206,2},{211,1},{219,2},{217,2},{222,1},{606,2},{604,2},{608,1},{228,2},{226,2},{232,1},{239,2},{237,2},{243,1},{208,2},{206,2},{211,1},{219,2},{217,2},{222,1},{606,2},{604,2},{608,1},
      {250,2},{248,2},{254,1},{261,2},{259,2},{265,1},{230,2},{228,2},{233,1},{241,2},{239,2},{244,1},{614,2},{612,2},{616,1},{250,2},{248,2},{254,1},{261,2},{259,2},{265,1},{230,2},{228,2},{233,1},{241,2},{239,2},{244,1},{614,2},{612,2},{616,1},
      {272,2},{270,2},{276,1},{283,2},{281,2},{287,1},{252,2},{250,2},{255,1},{263,2},{261,2},{266,1},{622,2},{620,2},{624,1},{272,2},{270,2},{276,1},{283,2},{281,2},{287,1},{252,2},{250,2},{255,1},{263,2},{261,2},{266,1},{622,2},{620,2},{624,1},
      {294,2},{292,2},{298,1},{305,2},{303,2},{309,1},{274,2},{272,2},{277,1},{285,2},{283,2},{288,1},{630,2},{628,2},{632,1},{294,2},{292,2},{298,1},{305,2},{303,2},{309,1},{274,2},{272,2},{277,1},{285,2},{283,2},{288,1},{630,2},{628,2},{632,1},
      {316,2},{314,2},{320,1},{327,2},{325,2},{331,1},{296,2},{294,2},{299,1},{307,2},{305,2},{310,1},{638,2},{636,2},{640,1},{316,2},{314,2},{320,1},{327,2},{325,2},{331,1},{296,2},{294,2},{299,1},{307,2},{305,2},{310,1},{638,2},{636,2},{640,1},
      {338,2},{336,2},{342,1},{349,2},{347,2},{353,1},{318,2},{316,2},{321,1},{329,2},{327,2},{332,1},{646,2},{644,2},{648,1},{338,2},{336,2},{342,1},{349,2},{347,2},{353,1},{318,2},{316,2},{321,1},{329,2},{327,2},{332,1},{646,2},{644,2},{648,1},
      {360,2},{358,2},{364,1},{371,2},{369,2},{375,1},{340,2},{338,2},{343,1},{351,2},{349,2},{354,1},{654,2},{652,2},{656,1},{360,2},{358,2},{364,1},{371,2},{369,2},{375,1},{340,2},{338,2},{343,1},{351,2},{349,2},{354,1},{654,2},{652,2},{656,1},
      {382,2},{380,2},{386,1},{393,2},{391,2},{397,1},{362,2},{360,2},{365,1},{373,2},{371,2},{376,1},{662,2},{660,2},{664,1},{382,2},{380,2},{386,1},{393,2},{391,2},{397,1},{362,2},{360,2},{365,1},{373,2},{371,2},{376,1},{662,2},{660,2},{664,1},
      {404,2},{402,2},{408,1},{415,2},{413,2},{419,1},{384,2},{382,2},{387,1},{395,2},{393,2},{398,1},{670,2},{668,2},{672,1},{404,2},{402,2},{408,1},{415,2},{413,2},{419,1},{384,2},{382,2},{387,1},{395,2},{393,2},{398,1},{670,2},{668,2},{672,1},
      {426,2},{424,2},{430,1},{437,2},{435,2},{441,1},{406,2},{404,2},{409,1},{417,2},{415,2},{420,1},{678,2},{676,2},{680,1},{426,2},{424,2},{430,1},{437,2},{435,2},{441,1},{406,2},{404,2},{409,1},{417,2},{415,2},{420,1},{678,2},{676,2},{680,1},
      {448,2},{446,2},{452,1},{459,2},{457,2},{463,1},{428,2},{426,2},{431,1},{439,2},{437,2},{442,1},{686,2},{684,2},{688,1},{448,2},{446,2},{452,1},{459,2},{457,2},{463,1},{428,2},{426,2},{431,1},{439,2},{437,2},{442,1},{686,2},{684,2},{688,1},
      {470,2},{468,2},{474,1},{481,2},{479,2},{485,1},{450,2},{448,2},{453,1},{461,2},{459,2},{464,1},{694,2},{692,2},{696,1},{470,2},{468,2},{474,1},{481,2},{479,2},{485,1},{450,2},{448,2},{453,1},{461,2},{459,2},{464,1},{694,2},{692,2},{696,1},
      {492,2},{490,2},{496,1},{503,2},{501,2},{507,1},{472,2},{470,2},{475,1},{483,2},{481,2},{486,1},{702,2},{700,2},{704,1},{492,2},{490,2},{496,1},{503,2},{501,2},{507,1},{472,2},{470,2},{475,1},{483,2},{481,2},{486,1},{702,2},{700,2},{704,1},
      {514,2},{512,2},{516,1},{521,2},{519,2},{523,1},{494,2},{492,2},{497,1},{505,2},{503,2},{508,1},{710,2},{708,2},{712,1},{514,2},{512,2},{516,1},{521,2},{519,2},{523,1},{494,2},{492,2},{497,1},{505,2},{503,2},{508,1},{710,2},{708,2},{712,1},
      {790,3},{867,3},{942,3},{790,3},{867,3},{942,3}};
    static constexpr int hessian_seq_len[77] = {5,5,8,9,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,12,3,3,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,3,3};
    static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient, Eigen::Ref<hessian_t> hessian)
    {
      gradient.array() = 0;
      scalar_t val = 0;
      auto ptr = hessian.valuePtr();
      for(int i=0; i<hessian.nonZeros(); i++) ptr[i] = 0;

      accHessian(val, gradient, hessian, grad_seq+0, 2, hessian_seq+0, hessian_seq_len+0, w.SEG(2,0), dynamics_0::hessian(param, x.SEG(2,4),x.SEG(1,51)));
      accHessian(val, gradient, hessian, grad_seq+2, 2, hessian_seq+10, hessian_seq_len+2, w.SEG(1,2), terminal_cost::hessian(param, x.SEG(2,48),x.SEG(2,74)));
      accHessian(val, gradient, hessian, grad_seq+4, 4, hessian_seq+18, hessian_seq_len+3, w.SEG(1,3), stage_cost::hessian(param, x.SEG(2,0),x.SEG(1,50),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+8, 4, hessian_seq+27, hessian_seq_len+4, w.SEG(1,4), stage_cost::hessian(param, x.SEG(2,2),x.SEG(1,51),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+12, 4, hessian_seq+41, hessian_seq_len+5, w.SEG(1,5), stage_cost::hessian(param, x.SEG(2,4),x.SEG(1,52),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+16, 4, hessian_seq+56, hessian_seq_len+6, w.SEG(1,6), stage_cost::hessian(param, x.SEG(2,6),x.SEG(1,53),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+20, 4, hessian_seq+71, hessian_seq_len+7, w.SEG(1,7), stage_cost::hessian(param, x.SEG(2,8),x.SEG(1,54),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+24, 4, hessian_seq+86, hessian_seq_len+8, w.SEG(1,8), stage_cost::hessian(param, x.SEG(2,10),x.SEG(1,55),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+28, 4, hessian_seq+101, hessian_seq_len+9, w.SEG(1,9), stage_cost::hessian(param, x.SEG(2,12),x.SEG(1,56),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+32, 4, hessian_seq+116, hessian_seq_len+10, w.SEG(1,10), stage_cost::hessian(param, x.SEG(2,14),x.SEG(1,57),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+36, 4, hessian_seq+131, hessian_seq_len+11, w.SEG(1,11), stage_cost::hessian(param, x.SEG(2,16),x.SEG(1,58),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+40, 4, hessian_seq+146, hessian_seq_len+12, w.SEG(1,12), stage_cost::hessian(param, x.SEG(2,18),x.SEG(1,59),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+44, 4, hessian_seq+161, hessian_seq_len+13, w.SEG(1,13), stage_cost::hessian(param, x.SEG(2,20),x.SEG(1,60),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+48, 4, hessian_seq+176, hessian_seq_len+14, w.SEG(1,14), stage_cost::hessian(param, x.SEG(2,22),x.SEG(1,61),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+52, 4, hessian_seq+191, hessian_seq_len+15, w.SEG(1,15), stage_cost::hessian(param, x.SEG(2,24),x.SEG(1,62),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+56, 4, hessian_seq+206, hessian_seq_len+16, w.SEG(1,16), stage_cost::hessian(param, x.SEG(2,26),x.SEG(1,63),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+60, 4, hessian_seq+221, hessian_seq_len+17, w.SEG(1,17), stage_cost::hessian(param, x.SEG(2,28),x.SEG(1,64),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+64, 4, hessian_seq+236, hessian_seq_len+18, w.SEG(1,18), stage_cost::hessian(param, x.SEG(2,30),x.SEG(1,65),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+68, 4, hessian_seq+251, hessian_seq_len+19, w.SEG(1,19), stage_cost::hessian(param, x.SEG(2,32),x.SEG(1,66),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+72, 4, hessian_seq+266, hessian_seq_len+20, w.SEG(1,20), stage_cost::hessian(param, x.SEG(2,34),x.SEG(1,67),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+76, 4, hessian_seq+281, hessian_seq_len+21, w.SEG(1,21), stage_cost::hessian(param, x.SEG(2,36),x.SEG(1,68),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+80, 4, hessian_seq+296, hessian_seq_len+22, w.SEG(1,22), stage_cost::hessian(param, x.SEG(2,38),x.SEG(1,69),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+84, 4, hessian_seq+311, hessian_seq_len+23, w.SEG(1,23), stage_cost::hessian(param, x.SEG(2,40),x.SEG(1,70),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+88, 4, hessian_seq+326, hessian_seq_len+24, w.SEG(1,24), stage_cost::hessian(param, x.SEG(2,42),x.SEG(1,71),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+92, 4, hessian_seq+341, hessian_seq_len+25, w.SEG(1,25), stage_cost::hessian(param, x.SEG(2,44),x.SEG(1,72),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+96, 4, hessian_seq+356, hessian_seq_len+26, w.SEG(1,26), stage_cost::hessian(param, x.SEG(2,46),x.SEG(1,73),x.SEG(2,74),x.SEG(1,76)));
      accHessian(val, gradient, hessian, grad_seq+100, 2, hessian_seq+368, hessian_seq_len+27, w.SEG(2,27), dynamics_0::hessian(param, x.SEG(2,0),x.SEG(1,50)));
      accHessian(val, gradient, hessian, grad_seq+102, 3, hessian_seq+374, hessian_seq_len+29, w.SEG(2,29), dynamics_eq::hessian(param, x.SEG(2,4),x.SEG(2,2),x.SEG(1,51)));
      accHessian(val, gradient, hessian, grad_seq+105, 3, hessian_seq+404, hessian_seq_len+31, w.SEG(2,31), dynamics_eq::hessian(param, x.SEG(2,6),x.SEG(2,4),x.SEG(1,52)));
      accHessian(val, gradient, hessian, grad_seq+108, 3, hessian_seq+434, hessian_seq_len+33, w.SEG(2,33), dynamics_eq::hessian(param, x.SEG(2,8),x.SEG(2,6),x.SEG(1,53)));
      accHessian(val, gradient, hessian, grad_seq+111, 3, hessian_seq+464, hessian_seq_len+35, w.SEG(2,35), dynamics_eq::hessian(param, x.SEG(2,10),x.SEG(2,8),x.SEG(1,54)));
      accHessian(val, gradient, hessian, grad_seq+114, 3, hessian_seq+494, hessian_seq_len+37, w.SEG(2,37), dynamics_eq::hessian(param, x.SEG(2,12),x.SEG(2,10),x.SEG(1,55)));
      accHessian(val, gradient, hessian, grad_seq+117, 3, hessian_seq+524, hessian_seq_len+39, w.SEG(2,39), dynamics_eq::hessian(param, x.SEG(2,14),x.SEG(2,12),x.SEG(1,56)));
      accHessian(val, gradient, hessian, grad_seq+120, 3, hessian_seq+554, hessian_seq_len+41, w.SEG(2,41), dynamics_eq::hessian(param, x.SEG(2,16),x.SEG(2,14),x.SEG(1,57)));
      accHessian(val, gradient, hessian, grad_seq+123, 3, hessian_seq+584, hessian_seq_len+43, w.SEG(2,43), dynamics_eq::hessian(param, x.SEG(2,18),x.SEG(2,16),x.SEG(1,58)));
      accHessian(val, gradient, hessian, grad_seq+126, 3, hessian_seq+614, hessian_seq_len+45, w.SEG(2,45), dynamics_eq::hessian(param, x.SEG(2,20),x.SEG(2,18),x.SEG(1,59)));
      accHessian(val, gradient, hessian, grad_seq+129, 3, hessian_seq+644, hessian_seq_len+47, w.SEG(2,47), dynamics_eq::hessian(param, x.SEG(2,22),x.SEG(2,20),x.SEG(1,60)));
      accHessian(val, gradient, hessian, grad_seq+132, 3, hessian_seq+674, hessian_seq_len+49, w.SEG(2,49), dynamics_eq::hessian(param, x.SEG(2,24),x.SEG(2,22),x.SEG(1,61)));
      accHessian(val, gradient, hessian, grad_seq+135, 3, hessian_seq+704, hessian_seq_len+51, w.SEG(2,51), dynamics_eq::hessian(param, x.SEG(2,26),x.SEG(2,24),x.SEG(1,62)));
      accHessian(val, gradient, hessian, grad_seq+138, 3, hessian_seq+734, hessian_seq_len+53, w.SEG(2,53), dynamics_eq::hessian(param, x.SEG(2,28),x.SEG(2,26),x.SEG(1,63)));
      accHessian(val, gradient, hessian, grad_seq+141, 3, hessian_seq+764, hessian_seq_len+55, w.SEG(2,55), dynamics_eq::hessian(param, x.SEG(2,30),x.SEG(2,28),x.SEG(1,64)));
      accHessian(val, gradient, hessian, grad_seq+144, 3, hessian_seq+794, hessian_seq_len+57, w.SEG(2,57), dynamics_eq::hessian(param, x.SEG(2,32),x.SEG(2,30),x.SEG(1,65)));
      accHessian(val, gradient, hessian, grad_seq+147, 3, hessian_seq+824, hessian_seq_len+59, w.SEG(2,59), dynamics_eq::hessian(param, x.SEG(2,34),x.SEG(2,32),x.SEG(1,66)));
      accHessian(val, gradient, hessian, grad_seq+150, 3, hessian_seq+854, hessian_seq_len+61, w.SEG(2,61), dynamics_eq::hessian(param, x.SEG(2,36),x.SEG(2,34),x.SEG(1,67)));
      accHessian(val, gradient, hessian, grad_seq+153, 3, hessian_seq+884, hessian_seq_len+63, w.SEG(2,63), dynamics_eq::hessian(param, x.SEG(2,38),x.SEG(2,36),x.SEG(1,68)));
      accHessian(val, gradient, hessian, grad_seq+156, 3, hessian_seq+914, hessian_seq_len+65, w.SEG(2,65), dynamics_eq::hessian(param, x.SEG(2,40),x.SEG(2,38),x.SEG(1,69)));
      accHessian(val, gradient, hessian, grad_seq+159, 3, hessian_seq+944, hessian_seq_len+67, w.SEG(2,67), dynamics_eq::hessian(param, x.SEG(2,42),x.SEG(2,40),x.SEG(1,70)));
      accHessian(val, gradient, hessian, grad_seq+162, 3, hessian_seq+974, hessian_seq_len+69, w.SEG(2,69), dynamics_eq::hessian(param, x.SEG(2,44),x.SEG(2,42),x.SEG(1,71)));
      accHessian(val, gradient, hessian, grad_seq+165, 3, hessian_seq+1004, hessian_seq_len+71, w.SEG(2,71), dynamics_eq::hessian(param, x.SEG(2,46),x.SEG(2,44),x.SEG(1,72)));
      accHessian(val, gradient, hessian, grad_seq+168, 3, hessian_seq+1034, hessian_seq_len+73, w.SEG(2,73), dynamics_eq::hessian(param, x.SEG(2,48),x.SEG(2,46),x.SEG(1,73)));
      accHessian(val, gradient, hessian, grad_seq+171, 2, hessian_seq+1064, hessian_seq_len+75, w.SEG(2,75), dynamics_ss::hessian(param, x.SEG(2,74),x.SEG(1,76)));
      return val;
    }
  };

  static void variable_bounds(param_t &param, Eigen::Ref<variable_t> lb, Eigen::Ref<variable_t> ub)
  {
    constexpr scalar_t inf = std::numeric_limits<double>::infinity();
    lb << -inf,-inf,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-5.0,-12.0,-12.0,-20.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-inf,-inf,-inf;
    ub << inf,inf,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,8.3,12.0,12.0,30.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,inf,inf,inf;
  }
};
constexpr seqinfo QP::constraints::jac_seq[];
constexpr seqinfo QP::objective::grad_seq[];
constexpr seqinfo QP::objective::hessian_seq[];
constexpr int QP::objective::hessian_seq_len[];
constexpr seqinfo QP::lagrangian::grad_seq[];
constexpr seqinfo QP::lagrangian::hessian_seq[];
constexpr int QP::lagrangian::hessian_seq_len[];
#endif