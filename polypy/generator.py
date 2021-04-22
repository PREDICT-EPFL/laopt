from enum import Enum, auto
import polypy as pp
# from polypy.poly import VariableSet
import io
from string import ascii_letters, digits
from collections import defaultdict
from copy import copy
from contextlib import contextmanager


class PrePrint:
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
        return print("\n".join([self.pre + ln for ln in buf.getvalue().splitlines()]), file=self.output, **kwargs)

    def __str__(self):
        # Dump our current output to a string
        return self.output.getvalue()

    def add_dependency(self, dependency, dependencyType):
        # Register the item as needing generation
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


# def print_buffer(op):
#     """Use the internal print buffer if one is not provided"""

#     def newOp(self, *args, **kwargs):
#         buffer = kwargs.pop("buffer", None)
#         if buffer is None:
#             buffer = self.p
#         kwargs['p'] = buffer
#         return op(self, *args, **kwargs)
#     return newOp


class Generator:
    """Abstract class to manage a generated file"""

    def __init__(self, filename):
        self.filename = filename

        self.dependencies_p = defaultdict(lambda: PrePrint(""))  # String buffer for each type of dependency
        # self.p = PrePrint("")  # Global string buffer
        self.p = pp.generator_eigen.Eigen("")

        try:
            self.generate_preamble(self.p)
        except AttributeError:
            print("WARNING: No preamble generation function found for this class type")

    def __enter__(self):
        return self  # <- Return a Generator_Printer

    def __exit__(self, exc_type, exc_value, tb):
        # Write out the function
        self.write_file(filename=self.filename)

    def add_function(self, function):
        self._functions.append(function)

    # @property
    # def functions(self):
    #     return set(self._functions).union(*[f.functions for f in self._functions])

    # @property
    # def parameters(self):
    #     return set().union(*[f.parameters for f in self.functions])

    # @property
    # def constantMatrices(self):
    #     # These are eigen matrices that need to be initialized
    #     return set().union(*[f.constantMatrices for f in self.functions])

    def __call__(self, obj, **kwargs):
        """Generate the given object"""
        obj.generate_declaration(self.p, **kwargs)
        try:
            obj.generate_declaration(self.p, **kwargs)
        except AttributeError:
            print(f"ERROR: Could not generate code for {obj}")

    @contextmanager
    def generate_class(self, classname="LOpt"):
        self.classname = classname
        # with self.new_class(classname) as p:
        self.p('template<typename scalar_t>')
        self.p(f'struct {self.classname}')
        self.p("{")
        # p = PrePrint("")
        p = pp.generator_eigen.Eigen("")
        try:
            with p:
                yield p  # Get user input on what they want in the class
        finally:
            with p:
                # Generate any dependencies that have come up for this class
                self.generate_dependencies(p)
            self.p(p)
            self.p("};")

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

    def write_file(self, filename="gen.hpp"):
        print(f"Writing to file {filename}")
        with open(filename, "w+") as f:
            f.write(str(self.p))

