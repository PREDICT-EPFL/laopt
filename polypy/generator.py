from enum import Enum, auto
import polypy
from polypy.poly import VariableSet
import io
from string import ascii_letters, digits
from collections import defaultdict
from copy import copy
from contextlib import contextmanager


class PrePrint:

    # List of things that we want to generate
    class GenTypes(Enum):
        Function = auto() 
        ConstantMatrix = auto()
        ConstantScalar = auto()
        VariableMatrix = auto()
        VariableScalar = auto()


    """Overload print by adding a prefix to every line"""
    def __init__(self, pre):
        self.pre = pre
        self.output = io.StringIO()
        self.dependencies = dict()
        self.evaluations = []

        # self.number_type = "scalar_t"  # Used to generate eigen matrices. Are they templated types, or scalar_t?
        self._options = dict()  # Options that can be queried during generation
        self._options['number_type'] = 'scalar_t'

    def __enter__(self):
        self.pre = self.pre + "\t"
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.pre is not None:
            self.pre = self.pre[1:]
        # return True

    def option(self, option_name):
        """Return the value of the option, or None"""
        return self._options.get(option_name, None)

    def set_option(self, name, value):
        """Set the given option

        Note: In the current implementation, this option will be unset 
        when leaving the current "with" block
        """
        self._options[name] = value

    @contextmanager
    def set_options(self, **kwargs):
        old_options = copy(self._options)
        for key, value in kwargs.items():
            self._options[key] = value
        try:
            with self:
                yield self
        finally:
            pass
            self._options = old_options


    @contextmanager
    def function(self, **kwargs):
        self.__call__("{")
        try:
            with self.set_options(**kwargs):
                yield self
        finally:
            self.__call__("}" + kwargs.get("post_string", ""))

    def __call__(self, *args, **kwargs):
        """My custom print() op."""
        buf = io.StringIO()
        print(*args, **kwargs, file=buf)
        return print("\n".join([self.pre + ln for ln in buf.getvalue().splitlines()]), file=self.output)

    def __str__(self):
        # Dump our current output to a string
        return self.output.getvalue()

    def add_dependency(self, dependency, dependencyType):
        # Register the item as needing generation, 
        # or return the generated item if its already been registered
        self.dependencies[dependency.name] = dependency

    # add and get are there for the reuse of function calls. If in a single expression a call to the same
    # function with the same arguments is made multiple times, then get will just return the solution of 
    # the previous call.
    # Note that if the same PrePrint object is used to generate multiple functions and the previous call 
    # goes out of scope, we may have a problem here...
    def add(self, expr, name):
        # Add an expression 
        if self.evaluations:
            evaled, _ = zip(*(self.evaluations))
            assert expr not in evaled, ValueError("Adding an evaluation that has already been evaluated... shouldn't happen")
        self.evaluations.append((expr, name))

    def get(self, expr):
        # Return name of expression if it's been previously eavluated, else None
        if self.evaluations:
            name = next((name for e, name in self.evaluations if expr == e), None)
            return name
        return None


def preprint(pre=""):
    """Factory to generate a PrePrint object"""
    return PrePrint(pre)


def validate_name(name):
    # Validate name as valid C++ name
    if set(name).difference(ascii_letters + digits + "_") \
       or name[0] not in ascii_letters + "_":
        raise ValueError(f"""Invalid name: {name}
            Only letters, numbers and underscores allowed.
            First character must be a letter or underscore.""")


def print_buffer(op):
    """Use the internal print buffer if one is not provided"""

    def newOp(self, *args, **kwargs):
        buffer = kwargs.pop("buffer", None)
        if buffer is None:
            buffer = self.p
        kwargs['p'] = buffer
        return op(self, *args, **kwargs)
    return newOp


class Generator:
    # Generate a collection of functions in a class

    def __init__(self, filename="gen.hpp"):
        self._functions = []
        self.data = []  # Constants (matrices and/or scalars)
        self.variables = []  # Parameters (matrices and/or scalars)

        self.filename = filename

        self.p = preprint()  # String buffer with indentation support
        self.generate_preamble()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # Write out the function
        self.write_file(filename=self.filename)

    def add_function(self, function):
        self._functions.append(function)

    @property
    def functions(self):
        return set(self._functions).union(*[f.functions for f in self._functions])

    @property
    def parameters(self):
        return set().union(*[f.parameters for f in self.functions])

    # @property
    # def constants(self):
    #     return set().union(*[f.constants for f in self.functions])

    @property
    def constantMatrices(self):
        # These are eigen matrices that need to be initialized
        return set().union(*[f.constantMatrices for f in self.functions])

    @print_buffer
    def generate_preamble(self, p=None):
        p('#include <math.h>')
        p('#include "Eigen/Dense"')
        p('#include <Eigen/Sparse>')
        p('#include "unsupported/Eigen/AutoDiff"')
        p('')
        p('#include "polygen_helper.hpp"')  # Generates Jacobians
        p('')
        p('using namespace Eigen;')
        p('')

    # @print_buffer
    # def function_body(self, p=None, post_string=""):

    #     class Body:
    #         def __enter__(self):
    #             p('{')
    #             p.__enter__()
    #             return self

    #         def __exit__(self, exc_type, exc_value, tb):
    #             p.__exit__(exc_type, exc_value, tb)
    #             p('}' + post_string)
    #             p("")

    #     return Body()

    @contextmanager
    def generate_class(self, classname="LOpt"):
        self.classname = classname
        self.p('template<typename scalar_t>')
        self.p(f'struct {self.classname}')
        self.p("{")
        p = PrePrint("")  # Create new buffer for this class
        try:
            with p:
                yield p  # Get user input on what they want in the class
        finally:
            with p:
                # Generate any dependencies that have come up for this class
                self.generate_dependencies(p)
            self.p(p)
            self.p("};")

        # return self.function_body(p=self.p, post_string=";")

        # class OpenClass:
        #     def __init__(self, p, classname="LOpt"):
        #         self.p = p
        #         self.classname = classname

        #     def __enter__(self):
        #         print("ENTER")
        #         p = self.p
        #         p('template<typename scalar_t>')
        #         p(f'struct {self.classname}')
        #         p('{')
        #         p.__enter__()
        #         return self

        #     def __exit__(self, exc_type, exc_value, tb):
        #         print("EXIT")
        #         p = self.p
        #         p.__exit__(exc_type, exc_value, tb)
        #         p('};')
        #         return True

        # return OpenClass(p, classname)



    # @print_buffer
    # def open_class(self, classname="LOpt", p=None):
    #     p('template<typename scalar_t>')
    #     p(f'struct {classname}')
    #     p('{')
    #     return p

    # @print_buffer
    # def close_class(self, p=None):
    #     p('};')
    #     return p

    # @print_buffer
    def generate_dependencies(self, p):
        """Recursively generate all dependencies that have been added via add_dependency"""

        p.set_option("cast_constants", True)
        generated = dict()  # Set of things that have been generated
        jacobians_to_initialize = set()
        while True:
            to_generate = {k: p.dependencies[k] for k in set(p.dependencies) - set(generated)}
            if not to_generate:
                break
            for dep in to_generate.values():
                p.evaluations = []  # Reuse function evaluations locally
                dep.generate_declaration(p)
                try:
                    dep.generate_jacobian(self.classname, p)
                    jacobians_to_initialize.add(dep)
                except AttributeError:
                    pass
                try:
                    dep.generate_hessian(self.classname, p)
                    jacobians_to_initialize.add(dep)
                except AttributeError:
                    pass

                p("")
                generated[dep.name] = dep


    # if self.parameters:
        #     p("// Parameters")
        #     for param in self.parameters:
        #         p(param.generate_declaration())
        #     p("")

        # # print(f"CONSTANT MATRICES = {self.constantMatrices}")
        # if self.constantMatrices:
        #     p("// Constant matrices")
        #     for const in self.constantMatrices:
        #         p(const.generate_declaration())
        #     p("")

        # for func in self.functions:
        #     func.generate(p)
        #     func.generate_jacobian(self.classname, p)
        #     p("")

        # Build the constructor
        p(f"{self.classname}() :")
        with p:
            # Initialize jacobian objects
            p(",\n".join(f"{func.name}(this)" for func in jacobians_to_initialize))
        with p.function(): 
            for param in generated.values():
                try:
                    param.generate_initialization(p)
                except AttributeError:
                    pass

        # return p

    def write_file(self, filename="gen.hpp"):
        print(f"Writing to file {filename}")
        with open(filename, "w+") as f:
            f.write(str(self.p))


# class Generator:
#     def __init__(self, nlp, filename=None, number_type="double"):
#         assert nlp.objective is not None, "Must set the objective before generation"

#         self.nlp = nlp
#         self.basename = nlp.name + "Base"
#         self.number_type = number_type
#         if filename is not None:
#             with open(filename, 'w+') as f:
#                 with redirect_stdout(f):
#                     self.generate()
#         else:
#             self.generate()

#     def generate(self):
#         print('#include "polygen_helper.hpp"\n\n')

#         # Declare problem sizes
#         self.gen_forward_declaration()

#         # Class header
#         print("template <typename Derived>")
#         print(f"struct {self.basename} : public ProblemBase<{self.basename}<Derived>>")
#         print("{")

#         # Import types from ProblemBase
#         p = PrePrint("")
#         with p:
#             p(f"using Base = ProblemBase<{self.basename}<Derived>>;")
#             p("using typename Base::scalar_t;")
#             p("using typename Base::nlp_variable_t;")
#             p("using typename Base::nlp_constraints_t;")
#             p("using typename Base::nlp_eq_jacobian_t;")
#             p("")

#             p("/** problem dimensions */")
#             p("using Base::VAR_SIZE;")
#             p("using Base::NUM_EQ;")
#             p("using Base::NUM_INEQ;")
#             p("using Base::NUM_BOX;")
#             p("using Base::DUAL_SIZE;")
#             p("")

#         self.gen_constants()
#         self.gen_vartypes()
#         self.gen_variables()
#         self.gen_equality_constraints()
#         self.gen_functions()

#         # for f in self.nlp.functions:
#         #     f.gen_jacobian()

#         # self.gen_cost()
#         # self.gen_cost_gradient()
#         # self.gen_cost_gradient_hessian()
#         self.gen_equalities()
#         self.gen_equalities_linearised()
#         # self.gen_inequalities()
#         # self.gen_inequalities_linearised()
#         print("};")

#     def gen_forward_declaration(self):
#         nlp = self.nlp

#         p = PrePrint("")
#         p("// Define traits class")
#         p("template<typename Derived>")
#         p(f"struct {self.basename};")
#         p("")
#         p("template<typename Derived>")
#         p(f"struct nlp_traits<{self.basename}<Derived>>")
#         p("{")
#         with p:
#             p(f"using scalar_t = {self.number_type};")
#             p(f"enum {{ NX = {nlp.nx}, NE = {nlp.ne}, NI = {nlp.ni}, NP = 0}};")
#         p("};\n")

#     def gen_constants(self):
#         nlp = self.nlp
#         p = PrePrint("\t")
#         p("enum {")
#         with p:
#             for c in nlp.constants:
#                 p(f"{c.name} = {c},")
#         p("};")
#         p("")

#     def gen_vartypes(self):
#         # Declare the sizes of all variables being used
#         nlp = self.nlp
#         p = PrePrint("\t")
#         for v in nlp.var_types:
#             p(f"DECLARE_VAR_TYPE({v}, {v.len});")
#         p("")

#     def gen_variables(self):
#         nlp = self.nlp
#         offset = 0
#         for var in nlp.vars:
#             if isinstance(var, VariableSet):
#                 print(f"\tDECLARE_VAR({var}, {offset}, {var.var_type}_size, {var.num_vars});")
#             else:
#                 print(f"\tDECLARE_VAR({var}, {offset}, {var.var_type}_size);")
#             offset += var.var_type.len * var.num_vars
#         print("")

#     def gen_equality_constraints(self):
#         # Produce macros to access the equality constraints / duals
#         p = PrePrint("")
#         with p:
#             nlp = self.nlp
#             offset = 0
#             for con in nlp.equalities:
#                 p(f"DECLARE_CONSTRAINT(eq_{con.name}, {offset}, {con.size_output}, {con.num_iterations});")

#                 # Increase the offset
#                 offset += con.num_iterations * con.size_output
#             p("")

#     def gen_functions(self):
#         # Produce macros to declare functions
#         nlp = self.nlp
#         p = PrePrint("")
#         with p:
#             for f in nlp.functions:
#                 input_sizes = ", ".join(str(t.len) for t in f.input_types)
#                 p(f"DECLARE_FUNCTION({f.name}, {f.size_output}, {input_sizes});")
#             p("")

#             # Generate the constructor
#             args = ",\n\t\t".join(f.name + '(this)' for f in nlp.functions)
#             p(f"{nlp.name}Base() :\n\t\t{args} {{}}")
#         p("")

#     def gen_cost(self):
#         p = PrePrint("")
#         p("EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, ")
#         p("                              scalar_t &cost) noexcept")
#         p("{")
#         p("}")

#     def gen_cost_gradient(self):
#         p = PrePrint("")
#         p("EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, ")
#         p("                                       scalar_t &_cost, ")
#         p("                                       Eigen::Ref<nlp_variable_t> cost_gradient) noexcept")
#         p("{")
#         p("}")

#     def gen_cost_gradient_hessian(self):
#         p = PrePrint("")
#         p("EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, ")
#         p("                                               const Eigen::Ref<const static_parameter_t>& p,")
#         p("                                               scalar_t &_cost, ")
#         p("                                               Eigen::Ref<nlp_variable_t> _cost_gradient, ")
#         p("                                               Eigen::Ref<nlp_hessian_t> hessian) noexcept")
#         p("{")
#         p("}")

#     def gen_equalities(self):
#         p = PrePrint("")
#         with p:
#             p("EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, ")
#             p("                                    Eigen::Ref<nlp_constraints_t> _equalities) noexcept")
#             p("{")
#             with p:
#                 for con in nlp.equalities:
#                     # out = f"eq1(_equalities, i+1)"
#                     i = 0
#                     output = f"eq_{con.name}({'_equalities'}, 0)"
#                     if list(con.indices):
#                         i = list(con.indices)[0]
#                         p(f"for (int {i}={i.rng.start}, _con_ind=0; {i}<{i.rng.stop}; {i}+={i.rng.step}, _con_ind++)")
#                         output = f"eq_{con.name}({'_equalities'}, _con_ind)"
#                     p(f"{con.expression.op.name}({con.generate_args('var')}, {output});")
#             p("}")
#             p("")

#     def gen_equalities_linearised(self):
#         p = PrePrint("")
#         with p:
#             p("EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
#             p("                                               Eigen::Ref<nlp_constraints_t> equalities,")
#             p("                                               Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
#             p("{")
#             with p:
#                 for con in nlp.equalities:
#                     func = con.expression.op
#                     i = 0
#                     if list(con.indices):
#                         i = list(con.indices)[0]
#                         num_iterations = len(list(i.rng))
#                         p(f"for (int {i}={i.rng.start}, _con_index=0; {i}<{i.rng.stop}; {i}+={i.rng.step}, _con_index++)")
#                     p(f"{con.expression.op.name}({con.generate_args('var')}, // Inputs")
#                     with p:
#                         con_index = "_con_index" if list(con.indices) else "0"
#                         # p(f"{con.generate_args('var')}, // Inputs")
#                         output = f"eq_{con.name}({'equalities'}, {con_index})"
#                         p(f"{output}, // Output")
#                         j_args = []
#                         for var, typ in zip(con.expression.args, func.input_types):
#                             var_index = str(var.ind) if var.ind is not None else ""
#                             offset = f"eq_{con.name}_offset({con_index}), {var.name}_offset({var_index})"
#                             arg = f"jacobian.template block<eq_{con.name}_size, {typ}_size>({offset})"
#                             j_args.append(arg)
#                         args = ",\n\t\t\t".join(j_args)
#                         p(args + ");")
#                     p("")
#             p("}")
#         p("")

#     def gen_inequalities(self):
#         p = PrePrint("")
#         p("EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, ")
#         p("                                      Eigen::Ref<nlp_constraints_t> _equalities) const noexcept")
#         p("{")
#         p("}")

#     def gen_inequalities_linearised(self):
#         p("EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
#         p("                                                 Eigen::Ref<nlp_constraints_t> equalities,")
#         p("                                                 Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
#         p("{")
#         p("}")
