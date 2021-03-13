from polypy.poly import VariableSet

class Generator:
    def __init__(self, nlp, filename=None, number_type="double"):
        assert nlp.objective is not None, "Must set the objective before generation"

        self.nlp = nlp
        self.basename = nlp.name + "Base"
        self.number_type = number_type
        if filename is not None:
            with open(filename, 'w+') as f:
                with redirect_stdout(f):
                    self.generate()
        else:
            self.generate()

    def generate(self):
        print('#include "polygen_helper.hpp"\n\n')

        # Declare problem sizes
        self.gen_forward_declaration()

        # Class header
        print("template <typename Derived>")
        print(f"struct {self.basename} : public ProblemBase<{self.basename}<Derived>>")
        print("{")

        # Import types from ProblemBase
        p = PrePrint("")
        with p:
            p(f"using Base = ProblemBase<{self.basename}<Derived>>;")
            p("using typename Base::scalar_t;")
            p("using typename Base::nlp_variable_t;")
            p("using typename Base::nlp_constraints_t;")
            p("using typename Base::nlp_eq_jacobian_t;")
            p("")

            p("/** problem dimensions */")
            p("using Base::VAR_SIZE;")
            p("using Base::NUM_EQ;")
            p("using Base::NUM_INEQ;")
            p("using Base::NUM_BOX;")
            p("using Base::DUAL_SIZE;")
            p("")

        self.gen_constants()
        self.gen_vartypes()
        self.gen_variables()
        self.gen_equality_constraints()
        self.gen_functions()

        # for f in self.nlp.functions:
        #     f.gen_jacobian()

        # self.gen_cost()
        # self.gen_cost_gradient()
        # self.gen_cost_gradient_hessian()
        self.gen_equalities()
        self.gen_equalities_linearised()
        # self.gen_inequalities()
        # self.gen_inequalities_linearised()
        print("};")

    def gen_forward_declaration(self):
        nlp = self.nlp

        p = PrePrint("")
        p("// Define traits class")
        p("template<typename Derived>")
        p(f"struct {self.basename};")
        p("")
        p("template<typename Derived>")
        p(f"struct nlp_traits<{self.basename}<Derived>>")
        p("{")
        with p:
            p(f"using scalar_t = {self.number_type};")
            p(f"enum {{ NX = {nlp.nx}, NE = {nlp.ne}, NI = {nlp.ni}, NP = 0}};")
        p("};\n")

    def gen_constants(self):
        nlp = self.nlp
        p = PrePrint("\t")
        p("enum {")
        with p:
            for c in nlp.constants:
                p(f"{c.name} = {c},")
        p("};")
        p("")

    def gen_vartypes(self):
        # Declare the sizes of all variables being used
        nlp = self.nlp
        p = PrePrint("\t")
        for v in nlp.var_types:
            p(f"DECLARE_VAR_TYPE({v}, {v.len});")
        p("")

    def gen_variables(self):
        nlp = self.nlp
        offset = 0
        for var in nlp.vars:
            if isinstance(var, VariableSet):
                print(f"\tDECLARE_VAR({var}, {offset}, {var.var_type}_size, {var.num_vars});")
            else:
                print(f"\tDECLARE_VAR({var}, {offset}, {var.var_type}_size);")
            offset += var.var_type.len * var.num_vars
        print("")

    def gen_equality_constraints(self):
        # Produce macros to access the equality constraints / duals
        p = PrePrint("")
        with p:
            nlp = self.nlp
            offset = 0
            for con in nlp.equalities:
                p(f"DECLARE_CONSTRAINT(eq_{con.name}, {offset}, {con.size_output}, {con.num_iterations});")

                # Increase the offset
                offset += con.num_iterations * con.size_output
            p("")

    def gen_functions(self):
        # Produce macros to declare functions
        nlp = self.nlp
        p = PrePrint("")
        with p:
            for f in nlp.functions:
                input_sizes = ", ".join(str(t.len) for t in f.input_types)
                p(f"DECLARE_FUNCTION({f.name}, {f.size_output}, {input_sizes});")
            p("")

            # Generate the constructor
            args = ",\n\t\t".join(f.name + '(this)' for f in nlp.functions)
            p(f"{nlp.name}Base() :\n\t\t{args} {{}}")
        p("")

    def gen_cost(self):
        p = PrePrint("")
        p("EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, ")
        p("                              scalar_t &cost) noexcept")
        p("{")
        p("}")

    def gen_cost_gradient(self):
        p = PrePrint("")
        p("EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, ")
        p("                                       scalar_t &_cost, ")
        p("                                       Eigen::Ref<nlp_variable_t> cost_gradient) noexcept")
        p("{")
        p("}")

    def gen_cost_gradient_hessian(self):
        p = PrePrint("")
        p("EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, ")
        p("                                               const Eigen::Ref<const static_parameter_t>& p,")
        p("                                               scalar_t &_cost, ")
        p("                                               Eigen::Ref<nlp_variable_t> _cost_gradient, ")
        p("                                               Eigen::Ref<nlp_hessian_t> hessian) noexcept")
        p("{")
        p("}")

    def gen_equalities(self):
        p = PrePrint("")
        with p:
            p("EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, ")
            p("                                    Eigen::Ref<nlp_constraints_t> _equalities) noexcept")
            p("{")
            with p:
                for con in nlp.equalities:
                    # out = f"eq1(_equalities, i+1)"
                    i = 0
                    output = f"eq_{con.name}({'_equalities'}, 0)"
                    if list(con.indices):
                        i = list(con.indices)[0]
                        p(f"for (int {i}={i.rng.start}, _con_ind=0; {i}<{i.rng.stop}; {i}+={i.rng.step}, _con_ind++)")
                        output = f"eq_{con.name}({'_equalities'}, _con_ind)"
                    p(f"{con.expression.op.name}({con.generate_args('var')}, {output});")
            p("}")
            p("")

    def gen_equalities_linearised(self):
        p = PrePrint("")
        with p:
            p("EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
            p("                                               Eigen::Ref<nlp_constraints_t> equalities,")
            p("                                               Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
            p("{")
            with p:
                for con in nlp.equalities:
                    func = con.expression.op
                    i = 0
                    if list(con.indices):
                        i = list(con.indices)[0]
                        num_iterations = len(list(i.rng))
                        p(f"for (int {i}={i.rng.start}, _con_index=0; {i}<{i.rng.stop}; {i}+={i.rng.step}, _con_index++)")
                    p(f"{con.expression.op.name}({con.generate_args('var')}, // Inputs")
                    with p:
                        con_index = "_con_index" if list(con.indices) else "0"
                        # p(f"{con.generate_args('var')}, // Inputs")
                        output = f"eq_{con.name}({'equalities'}, {con_index})"
                        p(f"{output}, // Output")
                        j_args = []
                        for var, typ in zip(con.expression.args, func.input_types):
                            var_index = str(var.ind) if var.ind is not None else ""
                            offset = f"eq_{con.name}_offset({con_index}), {var.name}_offset({var_index})"
                            arg = f"jacobian.template block<eq_{con.name}_size, {typ}_size>({offset})"
                            j_args.append(arg)
                        args = ",\n\t\t\t".join(j_args)
                        p(args + ");")
                    p("")
            p("}")
        p("")

    def gen_inequalities(self):
        p = PrePrint("")
        p("EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, ")
        p("                                      Eigen::Ref<nlp_constraints_t> _equalities) const noexcept")
        p("{")
        p("}")

    def gen_inequalities_linearised(self):
        p("EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
        p("                                                 Eigen::Ref<nlp_constraints_t> equalities,")
        p("                                                 Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
        p("{")
        p("}")
