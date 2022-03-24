# import lacompiler as la
import copy
import textwrap as tw
from collections import OrderedDict
from ctypes import cdll
from functools import partial

import numpy as np
import yaml
from colorama import Back, Fore, Style, init
from scipy.sparse import csc_matrix, lil_matrix

init(autoreset=True)
np.set_printoptions(linewidth=200)


class fullprint:
	"""context manager for printing full numpy arrays

    a = np.arange(1001)

    with fullprint():
        print(a)

    print(a)

    with fullprint(threshold=None, edgeitems=10):
        print(a)
	"""

	def __init__(self, **kwargs):
		kwargs.setdefault('threshold', np.inf)
		self.opt = kwargs

	def __enter__(self):
		self._opt = np.get_printoptions()
		np.set_printoptions(**self.opt)

	def __exit__(self, type, value, traceback):
		np.set_printoptions(**self._opt)


class Indenter:
	"""Class that can indent / outdent as you write"""
	def __init__(self):
		self._stack = [""]

	def write(self, s):
		"""Append the line s to the output, indenting every line"""
		self._stack[-1] += s

	def writeln(self, s):
		"""Append the line s to the output, indenting every line"""
		self._stack[-1] += s + "\n"

	def __iadd__(self, s):
		self.writeln(s)
		return self

	def indent(self):
		self._stack.append("")

	def dedent(self):
		if len(self._stack) == 1:
			return
		self.write(tw.indent(self._stack.pop(), '  '))

	def _flatten(self):
		"""Return a flattened copy"""
		s = copy.deepcopy(self._stack)
		while len(s) > 1:
			p = s.pop()
			s[-1] += tw.indent(p, '  ')
		return s[0]

	def __str__(self):
		return self._flatten().rstrip()

	def __repr__(self):
		return str(self)


class Variable:
	def __init__(self, name, length, offset):
		self.name = name
		self.len = length
		self.offset = offset

		self.lb = np.full(length, -np.inf)
		self.ub = np.full(length, np.inf)

	@property
	def position(self):
		"""Return the position (offset, len) of this variable in the problem
		variable as a whole"""
		return (self.offset, self.len)

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __len__(self):
		return self.len

	def __le__(self, other):
		self.ub[:] = other
		return self

	def __ge__(self, other):
		self.lb[:] = other
		return self


class VariableSet:
	def __init__(self, name, variables):
		self.name = name
		self.variables = variables

	def __str__(self):
		if len(self.variables) == 1:
			return self.name
		return f"{self.name}[{len(self.variables)}]"

	def __repr__(self):
		return str(self)

	def __len__(self):
		return len(self.variables)


class Callable:
	"""A callable function"""
	def __init__(self, 
				 signature, 
				 name, 
				 num_args, 
				 num_outputs, 
				 input_sizes, 
				 jacobianStructure, 
				 hessianStructure):
		self.signature = signature
		self.name = name
		self.num_args = num_args
		self.num_outputs = num_outputs
		self.input_sizes = input_sizes
		self.jacobianStructure = jacobianStructure
		self.hessianStructure = hessianStructure

	def __call__(self, *args):
		"""Call the callable with the given arguments.
		Returns a Call."""

		# Verify correct arguments
		assert len(self.input_sizes) == len(args), f"Number of arguments must be {self.num_args}"
		for i, a in enumerate(args):
			assert type(a) == Variable, "Arguments must be variables"
			assert len(a) == self.input_sizes[i], f"{i}'th argument must be length {self.input_sizes[i]}, but {a} is of length {len(a)}"
		return Call("con", self, args)

	def __len__(self):
		"""Number of outputs"""
		return self.num_outputs

	@property
	def shape(self):
		"""(Number of outputs, number of inputs)"""
		return (self.num_outputs, sum(self.input_sizes))

	def __str__(self):
		return f"{self.name} : ({','.join(str(i) for i in self.input_sizes)}) ↦ {self.num_outputs}"

	def __repr__(self):
		return str(self)


class Infix:
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(partial(self.func, other))
    def __call__(self, v1, v2):
        return self.func(v1, v2)


@Infix
def name(str, con):
	con.name = str
	return con


class Call:
	"""A Callable after its been called"""
	def __init__(self, name, callable, args):
		self.name = name
		self.callable = callable
		self.args = args
		self.lb = np.full(callable.num_outputs, -np.inf)
		self.ub = np.full(callable.num_outputs, np.inf)

	def __str__(self):
		return f"{self.callable.name}({','.join(str(a) for a in self.args)})"

	def __len__(self):
		"""Number of outputs"""
		return len(self.callable)

	@property
	def shape(self):
		return self.callable.shape

	def __le__(self, other):
		self.ub[:] = other
		return self

	def __ge__(self, other):
		self.lb[:] = other
		return self

	def __eq__(self, other):
		self.lb[:] = other
		self.ub[:] = other
		return self

	def __or__(self, other):
		self.name = other
		return self


class VectorFunction:
	"""A concantenation of calls into a vector-valued function"""
	def __init__(self, name, compiler):
		self.name = name
		self.calls = []
		self.compiler = compiler

		self.o_postfix = Indenter()  # Used to write material after the function struct

	def __lshift__(self, call):
		"""Add a call to the function"""
		self.calls.append(call)
		return self

	def __str__(self):
		s = Fore.RED + self.name + "\n" + Fore.RESET
		name_len = max(len(c.name) for c in self.calls)
		lb_len = max(len(str(c.lb)) for c in self.calls)
		ub_len = max(len(str(c.ub)) for c in self.calls)
		call_len = max(len(str(c)) for c in self.calls)

		for call in self.calls:
			if np.linalg.norm(call.lb - call.ub) == 0:
				lb = f"{'': >{lb_len}}"
				lb_sym = "  "
				ub_sym = "=="
			else:
				lb = f"{str(call.lb): >{lb_len}}"
				lb_sym = "<="
				ub_sym = "<="
			s += f"  {call.name: <{name_len}} | {lb} {lb_sym} {str(call): <{call_len}} {ub_sym} {str(call.ub): <{ub_len}}\n"
		return s

	def __or__(self, name):
		"""Name the last entered call"""
		self.calls[-1].name = name
		return self

	def __len__(self):
		"""Number of outputs"""
		return sum(len(c) for c in self.calls)

	@property
	def shape(self):
		"""Return (num_outputs, num_variables)"""
		return (len(self), self.compiler.num_variables)

	@property
	def jacobianStructure(self):
		"""Build the jacobian sparsity structure for this function

		Returns a sparse matrix.
		"""
		S = lil_matrix(self.shape)

		# Write out sparsity structure by call
		row = 0
		for call in self.calls:
			J = call.callable.jacobianStructure
			column = 0
			for arg in call.args:
				VectorFunction.copy_block(S, J[:, column:column+len(arg)], row, arg.offset)
				column += len(arg)
			row += J.shape[0]

		S = csc_matrix(S)
		S.sort_indices()
		S.data = np.arange(stop=len(S.data))

		return S

	@property
	def hessianStructure(self):
		"""Build the hessian sparsity structure for this function

		Sets the valuePtr = 0..nnz"""
		H = lil_matrix((self.shape[1], self.shape[1]))

		# Write out sparsity structure by call
		for call in self.calls:
			args = call.args
			arg_offsets = np.insert(np.cumsum([len(x) for x in args]), 0, 0)

			for ind_output in range(len(call)):
				h = call.callable.hessianStructure[ind_output]

				# Iterate though each pair of args (x,y) and copy the
				# sub-block of their hessian to the right place
				for ix, x in enumerate(args):
					for iy, y in enumerate(args):
						xo, yo = arg_offsets[ix], arg_offsets[iy]
						blk = h[xo:xo+len(x), yo:yo+len(y)]
						VectorFunction.copy_block(H, blk, x.offset, y.offset)

		H = csc_matrix(H)
		H.sort_indices()
		H.data = np.arange(stop=len(H.data))
		return H	

	def copy_block(M, blk, row, column):
		"""Copy the blk matrix into the M matrix at location row, column"""
		for r, c in zip(*blk.nonzero()):
			M[row + r, column + c] = blk[r, c]

	def build_copy_sequence(self, target, source, rows, cols):
		"""Copy source into target
		The source is partitioned into blocks and copied into target according to
		- rows = {{target_row, len}, ...}
		- cols = {{target_col, len}, ...}
		Returns a vector of specifying the copies to be done on the source data in 
		data-contiguous order to achieve the requested sparse block-copy.

		Returns ((index, len), ...)
			index : offset into the target data vector
			len   : number of elements to copy
		sum(len) == len(source.data)

		Copy source to target by iterating through return sequence:
			offset = 0
				copy source.data(offset, len) -> target(index, len)
				offset += len
		"""

		# Add ordering to the coeffs of target
		def order_coeff(M):
			M = csc_matrix(M)
			M.sort_indices()
			M.data = np.arange(stop=len(M.data))
			return M
		source = order_coeff(source)
		target = order_coeff(target)

		def get_target_location(loc, partition):
			"""Return the location in the target where loc = (row, col) in the source should be copied

			partition = {{index_i,len_i}, ...}
			Partitions a vector of length sum len_i into segements starting at the index_i's"""

			for offset, length in partition:
				if loc >= length:
					loc -= length
				else:
					return offset + loc

			print(f"loc {loc}, partition {partition}")
			assert False, "Should not be here!"


		# We iterate over the source in data-continuous order, defining the copy sequence to the target
		seq = [] # The copying sequence

		for c in range(source.shape[1]):
			tcol = get_target_location(c, cols)

			# Iterate over the nonzeros in the column
			for r in source.indices[range(source.indptr[c], source.indptr[c+1])]:
				trow = get_target_location(r, rows)
				seq.append(target[trow, tcol]) # Index into the data at the target location

		# We now have a sequence of locations where the source in data-contiguous order
		# should be copied to. 
		# Partition this sequence into contiguous segments
		def consecutive(data, stepsize=1):
		    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
		seq = consecutive(seq)

		# And now we compress it into the desired return format
		return [(s[0], len(s)) for s in seq]

	def generate_sequence(self, name, blocks):
		"""Generate an array of sparseblock_info from the given block sequence

		blocks = (((offset,length),...), ...)
		"""
		flat_list = [item for sublist in blocks for item in sublist]
		# length = sum(length for offset,length in flat_list)
		return self.generate_array(name, blocks, "seqinfo", 
					array_length=len(flat_list),
					to_str=lambda x: ",".join(f"{{{offset},{length}}}" for offset,length in x),
					multiline=True)

	def generate_array(self, name, array, element_type, array_length=None, to_str=str, multiline=False):
		"""Generate a static constexpr array from array

			to_str: function mapping an element of array to a string
			element_type: C++ type of array elements
		"""
		array_length = array_length if array_length != None else len(array)

		o = Indenter()

		o.write(f"static constexpr {element_type} {name}[{array_length}] = {{")
		joiner = ","
		if multiline:
			o += ""
			o.indent()
			joiner = ",\n"
		o += joiner.join(map(to_str, array)) + "};"

		self.o_postfix += f"constexpr {element_type} {self.compiler.name}::{self.name}::{name}[];"
		return str(o)

	@property
	def bounds(self):
		"""Return the bounds lb,ub for the function"""
		lb = np.hstack([call.lb for call in self.calls])
		ub = np.hstack([call.ub for call in self.calls])
		return lb, ub

	def generate(self):
		"""Generate C++ code to evaluate this function"""
		o = Indenter()
		o += f"struct {self.name} : public function_util_t<scalar_t, Eigen::Vector<scalar_t, {len(self)}>, Eigen::SparseMatrix<scalar_t>>"
		o += "{"
		o.indent()
		o += tw.dedent(f"""\
				static constexpr std::size_t output_size = {len(self)};
				static constexpr std::size_t nnz_jacobian = {self.jacobianStructure.getnnz()};
				using out_t = Eigen::Vector<scalar_t, output_size>;
				using jacobian_t = Eigen::SparseMatrix<scalar_t>;
				using hessian_t = Eigen::SparseMatrix<scalar_t>;
				""")
		o += self.generate_eval() + "\n"
		o += self.generate_jacobian() + "\n"

		o += "static void bounds(param_t &param, Eigen::Ref<out_t> lb, Eigen::Ref<out_t> ub)"
		o += "{"
		o.indent()
		o += "constexpr scalar_t inf = std::numeric_limits<double>::infinity();"
		lb, ub = self.bounds
		o += "lb << " + ",".join(map(str,lb)) + ";"
		o += "ub << " + ",".join(map(str,ub)) + ";"
		o.dedent()
		o += "};"

		o.dedent()
		o += "};\n";
		return str(o)

	def generate_arglist(self, args):
		"""Generate a comma seperated list of arguments from the variables args"""
		make_arg = lambda var : f"x.SEG({len(var)},{var.offset})"
		return ','.join(map(make_arg,args))

	def generate_eval(self):
		"""Generate a C++ function to evaluate this vector function"""
		o = Indenter()
		o += tw.dedent(f"""\
				/**
				 * Evalute the function for the parameter param and return the result in out
				 */
				static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out)
				{{""")
		o.indent()
		offset = 0 # Output offset
		for call in self.calls:
			o += f"out.SEG({len(call)},{offset}) = {call.callable.name}::eval(param, {self.generate_arglist(call.args)}); // {str(call)}"
			offset += len(call)

		# # Search for sequences that can be compressed
		# ind = 0
		# comp = [(self.calls[0].name, self.calls[0].args)]
		# while ind < len(self.calls):
		# 	# Test if we're a fixed step past the last call
		# 	comp = self.calls[ind].name



		o.dedent()
		o += "};"
		return str(o)

	def generate_sparse_init(J, funcName, matrixName="J"):
		"""Generate a function that will initialize a sparse matrix to the given structure"""
		o = Indenter()

		# First we generate a function that will initialize the jacobian
		o += f"static void {funcName}(Eigen::SparseMatrix<scalar_t> &{matrixName})"
		o += "{"
		o.indent()
		o.write(tw.dedent(f"""\
				{matrixName}.resize({J.shape[0]},{J.shape[1]});
				{matrixName}.reserve({J.getnnz()});
				typedef Eigen::Triplet<scalar_t> T;
				std::array<T,{J.getnnz()}> tripletList = {{T"""))
		o.write(",".join(f"{{{row},{col},1}}" for row, col in zip(*J.nonzero())))
		o += "};"

		o += f"{matrixName}.setFromTriplets(tripletList.begin(), tripletList.end());"
		o.dedent()
		o += "}"
		return str(o)

	def generate_jacobian(self):
		"""Produce code to evaluate the jacobian of this function

		The call produced will have the form

		eval(param_t param, variable_t x, out_t &out, jacobian_t &jac)"""

		o = Indenter()

		J = self.jacobianStructure;
		o += VectorFunction.generate_sparse_init(J, "initialize_jacobian", "J")
		o += ""

		o.write(tw.dedent("""\
			/**
			 * Compute the jacobian of the overall function
			 */
			 """))

		# Store the sequence of copies to fill in the jacobian
		sequence = []
		row = 0
		for call in self.calls:
			col_partition = [arg.position for arg in call.args]
			row_partition = [(row, call.shape[1]),]

			seq = self.build_copy_sequence(J, call.callable.jacobianStructure, row_partition, col_partition)
			sequence.append(seq)
			row += call.shape[0]

		o += self.generate_sequence("jac_seq", sequence)

		o += "static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out, Eigen::Ref<jacobian_t> jacobian)"
		o += "{"
		o.indent()

		offset = 0 # Output offset
		seq_offset = 0 # Offset into the jac_seq
		for call, seq in zip(self.calls, sequence):
			o.write(f"setJ(out, jacobian, {offset}, jac_seq+{seq_offset}, {len(seq)}, ")
			o.write(f"{call.callable.name}::jac(param, {self.generate_arglist(call.args)}));")
			o.writeln(" // " + str(call))

			offset += call.shape[0]
			seq_offset += len(seq)

		o.dedent()
		o += "};"
		return str(o)


class WeightedSum(VectorFunction):
	"""A weighted sum of VectorFunctions

	f(x) = sum <wi, fi(x)> 
	where fi are VectorFunctions"""

	def __init__(self, name, compiler):
		super().__init__(name, compiler)

	def __iadd__(self, other):
		"""Add a VectorFunction to the sum"""
		self << other
		return self

	def __str__(self):
		o = Indenter()
		o += Fore.RED + self.name + Fore.RESET
		o.indent()
		o += "\n  + ".join(f"<w[{len(call)}], {str(call)}>" for call in self.calls)
		return str(o)

	def generate(self):
		"""Produce C++ code"""
		o = Indenter()

		o += f"struct {self.name} : public weightedsum_util_t<scalar_t, Eigen::Vector<scalar_t, num_variables>, Eigen::Vector<scalar_t, {len(self)}>>"
		o += "{"
		o.indent()
		o += tw.dedent(f"""\
			static constexpr std::size_t num_weights = {len(self)};
			using weight_t = Eigen::Vector<scalar_t, num_weights>;
			using gradient_t = Eigen::Vector<scalar_t, num_variables>;
			static constexpr std::size_t hessian_nnz = {self.hessianStructure.getnnz()};
			using hessian_t = Eigen::SparseMatrix<scalar_t>;
			""")
		o += ""
		o += self.generate_eval()
		o += ""
		o += self.generate_gradient()
		o += ""
		o += self.generate_hessian()
		o.dedent()
		o += "};"

		return str(o)

	def generate_eval(self):
		"""Produce code to evaluate this function

		The call produced will have the form

		val = eval(param_t param, weight_t weight, variable_t x)"""
		o = Indenter()
		o += tw.dedent(f"""\
			/**
			 * Evalute the function for the parameter param and return the result in out
			 */
			static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> &w, const Eigen::Ref<const variable_t> &x)
			{{""")
		o.indent()

		o += "scalar_t val = 0;"
		offset = 0 # Output offset
		for call in self.calls:
			# Write out the C++ format
			o += f"val += w.SEG({len(call)},{offset}).dot({call.callable.name}::eval(param, {self.generate_arglist(call.args)})); // {str(call)}"
			offset += len(call)
		o += "return val;"
		o.dedent()
		o += "};"
		return str(o)

	def generate_gradient(self):
		"""Produce code to evaluate the gradient of this weighted sum

		val = eval(param_t param, weight_t w, variable_t x, gradient_t &jac)"""
		o = Indenter()

		o += tw.dedent("""\
			/**
			 * Compute the gradient of the weighted sum
			 */""")

		oo = Indenter()
		oo += "static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient)"
		oo += "{"
		oo.indent()
		oo += "gradient.array() = 0;"
		oo += "scalar_t val = 0;"

		offset = 0 # Output offset
		sequence_offset = 0
		sequence = []
		for call in self.calls:
			oo += f"accGrad(val, gradient, grad_seq+{sequence_offset}, {call.callable.num_args}, "\
				 	f"w.SEG({len(call)},{offset}), {call.callable.name}::jac(param, {self.generate_arglist(call.args)})); "\
				 	"// {std(call)}"

			sequence.append([(arg.offset, len(arg)) for arg in call.args])
			sequence_offset += len(call.args)
			offset += len(call)

		oo += "return val;"
		oo.dedent()
		oo += "};"

		o += self.generate_sequence("grad_seq", sequence)
		o += str(oo)
		return str(o)

	def generate_hessian(self):
		"""Produce code to evaluate the hessian of this function

		The call produced will have the form

		eval(param_t param, variable_t x, out_t &out, jacobian_t &J, hessian_t &H)"""

		o = Indenter()

		H = self.hessianStructure

		o += tw.dedent(f"""\
			/**
			 * Initialize the hessian of the function
			 */""")
		o += VectorFunction.generate_sparse_init(H, "initialize_hessian", "H")
		o += tw.dedent(f"""\
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
		   """)

		oo = Indenter()
		oo += "static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient, Eigen::Ref<hessian_t> hessian)"
		oo += "{"
		oo.indent()
		oo += tw.dedent(f"""\
		   		gradient.array() = 0;
		   		scalar_t val = 0;
		   		auto ptr = hessian.valuePtr();
		   		for(int i=0; i<hessian.nonZeros(); i++) ptr[i] = 0;
		   		""")

		# Iterate over each call computing the copy sequence
		sequence = []
		offset = 0
		gradient_offset = 0
		hessian_offset = 0
		output_num_copies = []  # Number of copies in sequence for given output
		# seq_lengths = []
		for call in self.calls:
			# Compute the hessian for each output in turn
			hessian_call_sequence = [] # Sequence for the entire set of hessians
			for i in range(len(call)):
				partition = [arg.position for arg in call.args]
				seq = self.build_copy_sequence(H, call.callable.hessianStructure[i], partition, partition)
				hessian_call_sequence.extend(seq)
				output_num_copies.append(len(seq))
			sequence.append(hessian_call_sequence)

			oo += f"accHessian(val, gradient, hessian, grad_seq+{gradient_offset}, {call.callable.num_args}, "\
				  f"hessian_seq+{hessian_offset}, hessian_seq_len+{offset}, "\
				  f"w.SEG({call.callable.num_outputs},{offset}), "\
				  f"{call.callable.name}::hessian(param, {self.generate_arglist(call.args)}));"

			gradient_offset += call.callable.num_args
			hessian_offset += len(hessian_call_sequence)
			offset += call.callable.num_outputs

		oo += "return val;" 
		oo.dedent()
		oo += "}"

		o += self.generate_sequence("hessian_seq", sequence)
		o += self.generate_array("hessian_seq_len", output_num_copies, "int")
		o += str(oo)

		return str(o)


class Compiler:
	"""Collection of differentiable functions and weighted sums

		function : f(x) = [f1(x);...;fN(x)]
		weighted sum : f(x) = sum <wi, fi(x)>		
	"""
	def __init__(self, name):
		self.name = name
		self.variables = []
		self.variable_sets = []
		self.callables = {}
		self.functions = []
		self.weighted_sums = []

		self.param_t = "param_t"  # Name of parameter struct
		self.scalar_t = "scalar_t"  # Name of scalar type

		self.o_postfix = Indenter()  # Material to add at the end of the generation

	def _variable(self, name, length):
		variable = Variable(name, length, self.num_variables)
		self.variables.append(variable)
		return variable

	def variable(self, name, length, number=1):
		if number == 1:
			variables = [self._variable(name, length), ]
		else:
			variables = [self._variable(name + str(i), length) for i in range(number)]

		self.variable_sets.append(VariableSet(name, variables))
		if len(variables) > 1:
			return variables
		else:
			return variables[0]

	def function(self, name):
		f = VectorFunction(name, self)
		self.functions.append(f)
		return f

	def weighted_sum(self, name):
		f = WeightedSum(name, self)
		self.weighted_sums.append(f)
		return f

	@property
	def num_variables(self):
		"""Returns to total number of variables in the problem"""
		return sum(map(len, self.variables))

	@property
	def variable_bounds(self):
		"""Return the bounds lb,ub for the variables"""
		lb = np.hstack([variable.lb for variable in self.variables])
		ub = np.hstack([variable.ub for variable in self.variables])
		return lb, ub

	def _yaml_sparsity(yaml, shape):
		dat = list(zip(*yaml))
		o = np.ones_like(dat[0])
		return csc_matrix((o, dat), shape)

	def load_function_info(self, filename):
		"""Read C++ function information from given YAML file"""

		with open(filename, 'r') as file:
			callables = yaml.safe_load(file)
			for name, y in callables.items():
				num_inputs = sum(y['input_sizes'])

				J = Compiler._yaml_sparsity(y['jacobianStructure'], (y['num_outputs'], num_inputs))
				H = [Compiler._yaml_sparsity(y['hessianStructure'][i], (num_inputs, num_inputs)) for i in range(y['num_outputs'])]

				self.callables[name] = Callable(
							y['signature'],
							y['name'],
							y['num_input_vars'],
							y['num_outputs'],
							y['input_sizes'],
							J,
							H)

	def generate(self):
		"""Generate code for all the functions in this problem"""
		o = Indenter()

		o += tw.dedent(f"""\
			#ifndef __{self.name}_HPP
			#define __{self.name}_HPP

			struct {self.name}
			{{""")
		o.indent()
		o += tw.dedent(f"""\
			static constexpr int num_variables = {self.num_variables};
			using param_t = {self.param_t};
			using scalar_t = {self.scalar_t};
			using variable_t = Eigen::Vector<scalar_t, num_variables>;
			""")

		# Generate accessors for all the variables
		o += "// Variable accessors";
		for varset in self.variable_sets:
			var = varset.variables[0]
			if len(varset) == 1:
				o += f"static Eigen::Ref<Eigen::Vector<scalar_t, {len(var)}>> {var.name}(Eigen::Ref<variable_t> var) {{return var.template segment<{len(var)}>({var.offset});}};"
			else:
				o += f"static Eigen::Ref<Eigen::Vector<scalar_t, {len(var)}>> {varset.name}(Eigen::Ref<variable_t> var, int ind) {{return var.template segment<{len(var)}>({var.offset}+{len(var)}*ind);}};"
				mattype = f"Eigen::Matrix<scalar_t, {len(var)}, {len(varset)}>"
				o += f"static Eigen::Ref<{mattype}> {varset.name}(Eigen::Ref<variable_t> var) {{return Eigen::Map<{mattype}>(var.template segment<{len(var) * len(varset)}>({var.offset}).data());}};"
		o += ""

		# Define the signatures of all the callables
		o += "// Define convenience names for all differentiable functions"
		for name, callable in self.callables.items():
			o += f"using {name} = {callable.signature};"
		o += ""

		for func in self.functions:
			o += func.generate()
			self.o_postfix += str(func.o_postfix)
			func.o_postfix = Indenter()
		o += ""

		for wsum in self.weighted_sums:
			o += wsum.generate()
			self.o_postfix += str(wsum.o_postfix)
			wsum.o_postfix = Indenter()

		o += ""
		o += "static void variable_bounds(param_t &param, Eigen::Ref<variable_t> lb, Eigen::Ref<variable_t> ub)"
		o += "{"
		o.indent()
		o += "constexpr scalar_t inf = std::numeric_limits<double>::infinity();"
		lb, ub = self.variable_bounds
		o += "lb << " + ",".join(map(str,lb)) + ";"
		o += "ub << " + ",".join(map(str,ub)) + ";"
		o.dedent()
		o += "}"

		o.dedent()
		o += "};";
		o += str(self.o_postfix)
		self.o_postfix = Indenter()
		o += "#endif"

		return str(o)


class OptimizationProblem:
	"""An optimization problem of the form

	min_x objective(x)
	s.t.   lb <= g(x) <= ub
	     x_lb <= x    <= x_ub

	This is just a light wrapper on Compiler where one 
	function called "constraints" and one weighted sum
	called "objective" is defined.
	"""
	def __init__(self, name):
		self._compiler = Compiler(name)
		self._constraints = self._compiler.function("constraints")
		self._objective = self._compiler.weighted_sum("objective")

	def variable(self, name, length, number=1):
		return self._compiler.variable(name,length,number)

	@property
	def constraints(self):
		return self._constraints

	@property
	def objective(self):
		return self._objective

	@objective.setter
	def objective(self, value):
		# Do nothing... 
		# we just need this so that += gets passed through to the objective
		pass

	@property
	def scalar_t(self):
		return self._compiler.scalar_t

	@scalar_t.setter
	def scalar_t(self, value):
		self._compiler.scalar_t = value

	@property
	def param_t(self):
		return self._compiler.param_t

	@param_t.setter
	def param_t(self, value):
		self._compiler.param_t = value

	@property
	def callables(self):
		return self._compiler.callables

	def _add_lagrangian(self):
		"""Return a copy of the compiler with the lagrangian added"""
		compiler = copy.deepcopy(self._compiler)
		lag = compiler.weighted_sum("lagrangian")

		for f in self.objective.calls:
			lag += f
		for f in self.constraints.calls:
			lag += f

		## TODO: Add variable constarints to the lagrangian, but not the hessian...
		## NOTE: This will compute the correct hessian, but the gradient and value will be wrong

		return compiler

	
	def generate(self):
		return self._add_lagrangian().generate()

		# Generate code to compute the lagrangian
	    # L = obj + lam_ineq' * ineq + lam_eq' * eq + lam_var' * var

    	# The Hessian matrix that %Ipopt uses is
    	# \sigma_f \nabla^2 f(x_k) + \sum_{i=1}^m\lambda_i\nabla^2 g_i(x_k) \f]
    	# for the given values for \f$x\f$, \f$\sigma_f\f$, and \f$\lambda\f$.
    	# See \ref TRIPLET for a discussion of the sparse matrix format used in this method.

		# code = self.compiler.generate()
		# return code

	def load_function_info(self, filename):
		"""Read C++ function information from given YAML file"""
		self._compiler.load_function_info(filename)

	def __str__(self):
		o = Indenter()
		o += Fore.RED + f"Optimization problem {self._compiler.name}" + Fore.RESET
		o.indent()
		o += tw.dedent("""\
				min_x objective(x)
				s.t.   lb <= g(x) <= ub
				     x_lb <= x    <= x_ub
				""")
		o.dedent()

		o += Fore.RED + "variables" + Fore.RESET
		o.indent()
		o += str(self._compiler.variable_sets)
		o.dedent()
		o += ""

		o += str(self.constraints)
		o += str(self.objective)

		return str(o)




if __name__ == '__main__': 
	prob = OptimizationProblem("ipopt_nlp_test")
	x1 = prob.variable("x1", 1)
	x2 = prob.variable("x2", 1)
	x3 = prob.variable("x3", 1)
	x4 = prob.variable("x4", 1)

	prob.load_function_info("ipopt_functions.yml")
	print(prob.callables)
	ineq = prob.callables['ineq']
	eq = prob.callables['eq']
	obj = prob.callables['obj']

	prob.constraints << (ineq(x1,x2,x3,x4) >= 25) | "inequality"
	prob.constraints << (eq(x1,x2,x3,x4) == 40) | "equality"
	1 <= x1 <= 5
	1 <= x2 <= 5
	1 <= x3 <= 5
	1 <= x4 <= 5

	prob.objective += obj(x1,x2,x3,x4)

	print(prob)

	prob.scalar_t = "double"
	prob.param_t = "MyFunctions<double>::param_t"
	with open('../examples/ipopt_test.compiled.hpp', 'w') as f:
		f.write(prob.generate())


	# prob = OptimizationProblem("QP")
	# N = 25

	# x = prob.variable("x", 2, N)
	# u = prob.variable("u", 1, N-1)
	# xss = prob.variable("xss", 2)
	# uss = prob.variable("uss", 1)

	# prob.load_function_info("qp_functions.yml")

	# dynamics = prob.callables['dynamics_0']
	# dynamics_ss = prob.callables['dynamics_ss']
	# dynamics_eq = prob.callables['dynamics_eq']
	# stage_cost = prob.callables['stage_cost']
	# terminal_cost = prob.callables['terminal_cost']

	# prob.constraints << (dynamics(x[0],u[0]) == 0)
	# ub = np.array([1, 2])
	# -20 <= u[0] <= 30
	# for i in range(1,N-1):
	# 	prob.constraints << (dynamics_eq(x[i+1],x[i],u[i]) == 0) | f"dynamics{i}"
	# 	-2 <= u[i] <= 3
	# 	-5 <= x[i] <= 8.3
	# -12 <= x[N-1] <= 12
	# prob.constraints << (0 <= dynamics_ss(xss,uss) <= 0) | "steady_state"

	# obj = prob.objective
	# obj += dynamics(x[2],u[1])
	# obj += terminal_cost(x[N-1], xss)
	# for i in range(N-1):
	# 	obj += stage_cost(x[i], u[i], xss, uss)

	# prob.scalar_t = "double"
	# prob.param_t = "MyFunctions<double>::param_t"

	# print(prob)

	# with open('../examples/qp.compiled.hpp', 'w') as f:
	# 	f.write(prob.generate())
