# Generation model for Eigen C++

import polypy as pp


class Eigen(pp.generator.PrePrint):
    """PrePrint class overload that provides language-specific code access"""

    comment_prepend = "//"  # Text to prepend to every line of a comment

    def get_nlp_generator(self, nlp):
        return EigenNLP(nlp)

    #####################################################################
    # Evaluate functions return strings to evaluate the given expression
    # in the local language
    #####################################################################

    def evaluate_functionExpression(self, expr, *args):
        # Evaluate the function into a temporary matrix, and return the temp
        p = self
        out = pp._get_unique_name()
        p(f"Matrix<{p.option('number_type')}, {len(expr.function)}, 1> {out};")
        p(f"{expr.function.wrapped_name}<{p.option('number_type')}>({', '.join(args)}, {out});")
        return out

    def evaluate_hstack(self, expr, args, arg_shapes):
        # Stack the arguments horizontally
        p = self
        shape = expr.shape
        tmp = pp._get_unique_name()
        p(f"Matrix<{p.option('number_type')}, {shape[0]}, {shape[1]}> {tmp};")
        offset = 0
        for shape, strArg in zip(arg_shapes, args):
            p(f"{tmp}.template block<{shape[0]}, {shape[1]}>(0, {offset}) = {strArg};")
            offset += shape[1]
        return tmp

    def evaluate_vstack(self, expr, args, arg_shapes):
        # Stack the arguments vertically
        p = self
        tmp = pp._get_unique_name()
        shape = expr.shape
        p(f"Matrix<{p.option('number_type')}, {shape[0]}, {shape[1]}> {tmp};")
        offset = 0
        for shape, strArg in zip(arg_shapes, args):
            p(f"{tmp}.template block<{shape[0]}, {shape[1]}>({offset}, 0) = {strArg};")
            offset += shape[0]
        return tmp

    def evaluate_unaryexpression(self, expr, op, arg):
        # Evaluate a unary expression

        # Eigen builtin elementwise operations
        if op == "abs2":  # Squared absolute value
            return f"({arg}).cwiseAbs2()"
        if op == "abs":
            return f"({arg}).cwiseAbs()"
        if op == "sqrt":
            return f"({arg}).cwiseSqrt()"
        if op == "inverse":
            return f"({arg}).cwiseInverse()"

        # Convert to array, evaluate elementwise, convert back to matrix
        if op in set("log", "log10", "exp", "square", "cube", "inverse", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "arg"):
            return f"(({arg}).array().{op}()).matrix()"

        # Evaluate whatever function is given elementwise
        return f"({arg}).unaryExpr([](scalar_t x){{return {op}(x);}})"

    def evaluate_elementwise_expression(self, op, arg):
        """Return a string to compute a unary expression elementwise"""
        return f"{op}(({arg}).array()).matrix()"


    def get_var_info(self, var):
        """Return string representing object containing information about the variable"""
        if var.var_set:
            return f"{str(var.var_set.name)}.info({var.ind})"
        else:
            return f"{str(var)}.info()"

    def get_var_offset(self, var):
        """Return string computing the offset of the variable into the global var"""
        if var.var_set:
            return f"{str(var.var_set.name)}({var.ind})"
        else:
            return f"{str(var)}()"


    def eval_constraint(self, con, input_var, output_vec, output_jacobian):
        """Print a string to self that evaluates the constraint"""

        # col_offset = []
        # for var in con.args:
        #     iVar = self.variables.index(var)
        #     iCon = self.constraints.index(con)
        #     col_offset.append(sum([len(blk.con) for blk in blocks[:iCon, iVar] if blk]))

        # We're using the call structure:
        #   function({offset1, offset2, ..., offsetN}, eq_offset, variable, out, jacobian)

        vars = ', '.join([arg.basename + f"({arg.ind_str})" for arg in con.args])

        p(f"{con.function.name}({{{vars}}}, {{{', '.join(str(i) for i in col_offset)}}}, {con.name}, var, constraints, jacobian);")



class EigenGenerator(pp.generator.Generator):
    """Implements generating code in Eigen"""

    def __init__(self, filename="gen.hpp"):
        super().__init__(filename)

    def generate_preamble(self, p):
        p('#include <math.h>')
        p('#include <map>')
        p('#include "Eigen/Dense"')
        p('#include <Eigen/Sparse>')
        p('#include "unsupported/Eigen/AutoDiff"')
        p('')
        p('#include "polygen_helper.hpp"')  # Generates Jacobians
        p('')
        p('using namespace Eigen;')
        p('')

    # def class_preamble(self, p):
    #     """Generate 


class EigenNLP:
    """Eigen-specific generation code for NLP objects"""

    def __init__(self, nlp):
        self.nlp = nlp  # Generic NLP object

    def declare_variables(self, p, vars):
        """Print generated code into p to define the variable accessors
        """
        prev_var = None
        for var in vars:
            try:
                num_vars = var.num_vars
            except AttributeError:
                num_vars = 1

            if prev_var is None:
                prev = ""
            else:
                prev = f", {prev_var}_t"

            p(f"using {var}_t = var_t<{len(var)}, {num_vars}{prev}>;")
            prev_var = var
        p("")
        for var in vars:
            p(f"static constexpr auto {var} = {var}_t();")

        # var_list = ", ".join(f'{{{var}, "{var}"}}' for var in vars)
        # p("")
        # p.comment("Convenient user-accessors")
        # p(f"std::array<var_slow_t, {len(vars)}> variable_list = {{var_slow_t{var_list}}};")


    def declare_constraints(self, p, constraints):
        """Print generated code into p to declare constraints"""
        prev_con = None
        offset = 0
        for con in constraints:
            setattr(con, 'offset', offset)
            num_con = con.shape[1]

            if prev_con is None:
                prev = ""
            else:
                prev = f", {prev_con}_t"

            p(f"using {con}_t = con_t<{len(con)}, {num_con}, {con.nnz}{prev}>;")

            offset += len(con) * num_con
            prev_con = con

        p("")
        for con in constraints:
            p(f"static constexpr auto {con} = {con}_t();")

        p("")
        con_list = ", ".join(f'{{{con}, "{con}"}}' for con in constraints)
        p.comment("Convenient user-accessors")
        p(f"std::array<con_slow_t, {len(constraints)}> constraint_list = {{con_slow_t{con_list}}};")


    def declare_nlp_sizes(self, p):
        nlp = self.nlp
        p("enum")
        with p.function(post_string=";"):
            p(f"NUM_VARS = {nlp.compressed_vars[-1]}.next,")
            p(f"NUM_CON   = {nlp.constraints[-1]}.next,")
            p(f"nnz_constraints_jacobian = {' + '.join(f'{con}.nnz' for con in nlp.constraints)}")
        p("")

