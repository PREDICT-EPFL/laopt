import cogapp as cog

class Variable:
    def __init__(self, name, offset, rows, cols=1, col=None):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.offset = offset

        self.col = col

    def __getitem__(self, key):
        assert type(key) in (int, Index), "Index must be an integer or an Index object"

        # Gets a column
        assert(self.cols > 1) # Can't get a column of a single vector

        newvar = copy.copy(self)
        newvar.col = key
        return newvar

    def __str__(self):
        if self.cols > 1:
            if self.col is None: # Return the whole matrix as a vector
                return f"{self.name}Mat()"
            else:
                return f"{self.name}({self.col})" # Return the requested column
        else:
            return f"{self.name}()"

    # def gen_define(self):
    #     # Generate DECLARE_VAR macro

    #     cog.out(f"DECLARE_VAR({self.name}, {self.offset}, {self.rows}")
    #     if self.cols > 1:
    #         cog.out(f", {self.cols}")
    #     cog.outl(");")

    # def gen_define(self, size = False, offset = False, func = False):
    #     # Set to true the element to generate
    #     # Generate var(i) const op
    #     if size:
    #         cog.outl(f"static constexpr auto s{self.name} = {self.rows};")

    #         if self.cols > 1:
    #             cog.outl(f"static constexpr auto s{self.name}Mat = {self.rows * self.cols};")


    #     if self.cols == 1:
    #         if offset:
    #             cog.outl(f"constexpr auto o{self.name}() {{return {self.offset};}};")
    #         if func:
    #             cog.outl(f"constexpr auto  {self.name}() {{return x.template segment<s{self.name}>(o{self.name}());}};")
    #     else:
    #         # Accessors for a column of the variable
    #         if offset:
    #             cog.outl(f"constexpr auto o{self.name}(int col) {{return {self.offset}+{self.rows}*col;}};")
    #         if func:
    #             cog.outl(f"constexpr auto  {self.name}(int col) {{return x.template segment<s{self.name}>(o{self.name}(col));}};")

    #         # Accessors for the whole matrix
    #         if offset:
    #             cog.outl(f"constexpr auto o{self.name}Mat() {{return {self.offset};}};")
    #         if func:
    #             cog.outl(f"constexpr auto  {self.name}Mat() {{return x.template segment<s{self.name}Mat>(o{self.name}Mat());}};")

    # @property
    # def offset_func(self):
    #     """Return the op to compute the offset"""
    #     if self.cols == 1:
    #         return f"o{self.name}()"
    #     if self.col is None:
    #         return f"o{self.name}Mat()"
    #     return f"o{self.name}({self.col})"

    # @property
    # def size_func(self):
    #     """Return the constant computing the size of the variable"""
    #     if self.cols == 1:
    #         return f"s{self.name}"    # There's only one column
    #     if self.col is None:
    #         return f"s{self.name}Mat" # Return the whole matrix
    #     return f"s{self.name}"        # Size of one column

    # @property
    # def offset(self):
    #     if self.col is None:
    #         return self.__offset
    #     else:
    #         return self.__offset + self.rows * self.col

    # @offset.setter
    # def offset(self, offset):
    #     self.__offset = offset

    # @property
    # def seg(self):
    #     # return a x.SEG(size, offset) form
    #     return f"x.SEG({self.rows}, {self.offset})"


class fConstant(float):
  def __new__(cls, name, num):
      return super(Constant, cls).__new__(cls, num)

  def __init__(self, name, num):
      self.name = name


class iConstant(int):
  def __new__(cls, name, num):
      return super(Constant, cls).__new__(cls, num)

  def __init__(self, name, num):
      self.name = name


class Function:
  def __init__(self, name, size_output, *input_types):
    self.name = name
    self.size_output = size_output # If None, then this is a scalar-output op
    self.input_types = input_types # Tuples of (varname, size)



class NLP:
  def __init__(self):
    self.base_name = "base_name"
    self.nx = 5
    self.ne = 6
    self.ni = 7
    self.scalar = "scalar"
  
    self.constants = []
    self.vars = []
  
  def const(self, name, value, number_type=None):
    if number_type is None:
      number_type = type(value)
    if number_type == float:
      c = fConstant(name, value)
    else if number_type == int:
      c = iConstant(name, value)
    else
      raise Exception("Unknown number type")
    self.constants.append(c)
    return c
  
  def var(self, name, rows, cols=1):
    v = Variable(name, rows, cols)
    self.vars.append(v)
    return v


class Index:
  def __init__(self, rng, op = 'i'):
      self.op = op
      self.rng = rng
  
  @property
  def num_iterations(self):
      # Compute the number of iterations that this index represents 
      # i.e., max(i) - min(i)
      return self.rng.stop - self.rng.start
  
  def __str__(self):
      return self.op
  
  def makeop(self, other, op):
      if op in ('+', '-'):
          return Index(self.rng, f"({str(self)}{op}{str(other)})")
      else:
          return Index(self.rng, f"{str(self)}{op}{str(other)}")
  
  def __add__(self, other):
      return self.makeop(other, '+')
  
  def __sub__(self, other):
      return self.makeop(other, '-')
  
  def __mul__(self, other):
      return self.makeop(other, '*')
  
  def __div__(self, other):
      return self.makeop(other, '/')
  
  def __radd__(self, other):
      return self.makeop(other, '+')
  
  def __rsub__(self, other):
      return self.makeop(other, '-')
  
  def __rmul__(self, other):
      return self.makeop(other, '*')
  
  def __rdiv__(self, other):
      return self.makeop(other, '/')


class Generator:
  def generate(self):
    # Main generation op
  
    # Declare problem sizes
    self.gen_forward_declaration()
  
    # Class header
    cog.outl(f"class {self.base_name} : public ProblemBase<{self.base_name}>")
    cog.outl("{")
  
    self.gen_cost()
    self.gen_cost()
    self.gen_cost_gradient()
    self.gen_cost_gradient_hessian()
    self.gen_equalities()
    self.gen_equalities_linearised()
    self.gen_inequalities()
    self.gen_inequalities_linearised()
  
    cog.outl("};")
  
  
  def gen_forward_declaration(self):
    cog.outl("POLYMPC_FORWARD_NLP_DECLARATION(")
    cog.outl(str(self.base_name) + ", // Name")
    cog.outl(str(self.nx) + ", // Number of primal variables")
    cog.outl(str(self.ne) + ", // Number of equalities")
    cog.outl(str(self.ni) + ", // Number of inequalities")
    cog.outl(str(0) + ", // Number of parameters (not used)")
    cog.outl(str(self.scalar) + " // Type);")
  
  def gen_cost(self):
    cog.outl("""
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, 
                                  scalar_t &cost) noexcept
    {
    """, dedent=True)
  
    cog.outl("}")
  
  def gen_cost_gradient(self):
    cog.outl("EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, ")
    cog.outl("                                       scalar_t &_cost, ")
    cog.outl("                                       Eigen::Ref<nlp_variable_t> cost_gradient) noexcept")
    cog.outl("{")
    cog.outl("}")
  
  def gen_cost_gradient_hessian(self):
    cog.outl("EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, ")
    cog.outl("                                               const Eigen::Ref<const static_parameter_t>& p,")
    cog.outl("                                               scalar_t &_cost, ")
    cog.outl("                                               Eigen::Ref<nlp_variable_t> _cost_gradient, ")
    cog.outl("                                               Eigen::Ref<nlp_hessian_t> hessian) noexcept")
    cog.outl("{")
    cog.outl("}")
  
  def gen_equalities(self):
    cog.outl("EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, ")
    cog.outl("                                    Eigen::Ref<nlp_constraints_t> _equalities) const noexcept")
    cog.outl("{")
    cog.outl("}")
  
  def gen_equalities_linearised(self):
    cog.outl("EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
    cog.outl("                                               Eigen::Ref<nlp_constraints_t> equalities,")
    cog.outl("                                               Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
    cog.outl("{")
    cog.outl("}")
  
  def gen_inequalities(self):
    cog.outl("EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, ")
    cog.outl("                                      Eigen::Ref<nlp_constraints_t> _equalities) const noexcept")
    cog.outl("{")
    cog.outl("}")
  
  def gen_inequalities_linearised(self):
    cog.outl("EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
    cog.outl("                                                 Eigen::Ref<nlp_constraints_t> equalities,")
    cog.outl("                                                 Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
    cog.outl("{")
    cog.outl("}")
