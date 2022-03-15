/**
 * Compile an LA Problem into data structures that can be incorporated into a solver.
 * 
 */


#include <string>
#include <sstream>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <chrono>
#include <ctime> 
#include <numeric>
#include <fstream>
#include <tuple>


#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <experimental/iterator>
#include <limits>

#include "lampc.hpp"
#include "la_compiler.hpp"
#include "IndentStream.hpp"

callable_info::callable_info(variable_p var)
{
	int size = var->len;
	jacobianStructure = Eigen::MatrixX<int>::Identity(size, size).sparseView();
	for(int i=0; i<size; i++)
		hessianStructure.push_back(Eigen::SparseMatrix<int>(size, size));

	input_sizes.push_back(size);
	num_outputs = size;
	num_args = 1;
	name = "id";
	signature = "TODO id";
}


// A group of similarly-named variables
struct variableset_t
{
	std::string name;
	std::vector<variable_p> variables;

	variableset_t(std::string name) : name(name)	{};

	std::string str()
	{
		if(variables.size() == 1) return name;
		else                      return fmt::format("{}[{}]", name, variables.size());
	}
};



int call_t::num_outputs() const 
	{ return callable->num_outputs; }


/**
 * Return the location in the target where ind in the source should be copied
 * 
 * partition = {{index_i,len_i}, ...}
 * Partitions a vector of length sum len_i into segements starting at the index_i's
 */
int get_target_location(int ind, std::vector<seqinfo> &partition)
{
	std::stringstream o;
	// for(auto &p : partition) o << fmt::format("({},{}), ", p.index, p.length);
	// std::cout << fmt::format("Requesting index {} in partition {} -> ", ind, o.str());
	for(auto &segment: partition)
	{
		if(ind >= segment.length) ind -= segment.length;
		else 
		{
			// std::cout << segment.index + ind << "\n";
			return segment.index + ind;
		}
	}
	throw std::runtime_error("Invalid index passed to get_target_location");
	assert(0 && "Invalid index passed to get_target_location");
	return -1;
}


/**
 * Copy source into target
 * 
 * The source is partitioned into blocks and copied into target according to
 * - rows = {{target_row, len}, ...}
 * - cols = {{target_col, len}, ...}
 * 
 * Returns a vector of seqinfo specifying the copies to be done on the source data in 
 * data-contiguous order to achieve the requested sparse block-copy.
 */
std::vector<seqinfo> build_copy_sequence(
						Eigen::SparseMatrix<int> &_target,
						Eigen::SparseMatrix<int> &source,
						std::vector<seqinfo> &rows,
						std::vector<seqinfo> &cols)
{
	if(!_target.isCompressed()) std::runtime_error( "_target matrix must be in compressed format" );
	if(!source.isCompressed()) std::runtime_error( "source matrix must be in compressed format" );

	// Add ordering to the coeffs of target
	Eigen::SparseMatrix<int> target(_target);
	for(int i=0; i<target.nonZeros(); i++)
		target.valuePtr()[i] = i;

	// We iterate over the source in data-continuous order, defining the copy sequence to the target
	std::vector<seqinfo> seq; // The copying sequence

	for (int c=0; c<source.outerSize(); ++c)
	{
		int tcol = get_target_location(c, cols);
		for (SparseMatrix<int>::InnerIterator it(source,c); it; ++it)
		{
			int trow = get_target_location(it.row(), rows);
			int tindex = target.coeffRef(trow,tcol); // Index into the data at the target location

			// std::cout << fmt::format("source ({:3d},{:3d}) -> target ({:3d},{:3d}) [{:3d}]", it.row(), c, trow, tcol, tindex);

			// The next target index if the data is contiguous
			if(seq.size() == 0 || tindex != seq.back().index + seq.back().length)
			{
				// std::cout << "  NON-CONTIGUOUS\n";
				seq.push_back({.index = tindex, .length=1});
			}
			else
			{
				// std::cout << "  CONTIGUOUS\n";
				seq.back().length++;
			}
		}		
	}

	return seq;
}

/**
 * Take a set of variable arguments and build a partition of the source matrix
 */
std::vector<seqinfo> args_to_partion(std::vector<variable_p> &args)
{
	std::vector<seqinfo> partition;	
	for(auto& arg: args) 
		partition.push_back(seqinfo{.index=arg->offset, .length=arg->len});
	return partition;
}




callable_p LACompiler::callable(callable_info info)
{
	auto callable = std::make_shared<callable_t>(info, this);
	callables.push_back(callable);
	return callable;
}

function_p LACompiler::function(std::string name)
{
	auto function = std::make_shared<function_t>(name, *this);
	functions.push_back(function);
	return function;
}

weightedsum_p LACompiler::weighted_sum(std::string name)
{
	auto wsum = std::make_shared<weightedsum_t>(name, *this);
	weightedsums.push_back(wsum);
	return wsum;
}

int LACompiler::num_variables()
{
	if(variables.size() == 0) 
		return 0;
	
	return variables.back()->offset + variables.back()->len;
}


/**
 * Create a variable
 */
variable_p LACompiler::variable_impl(std::string name, int len)
{
	auto var = std::make_shared<variable_t>(name, len, num_variables());
	variables.push_back(var);
	return var;
}

variable_p LACompiler::variable(std::string name, int len)
{
	auto var = variable_impl(name, len);
	auto varset = std::make_shared<variableset_t>(name);
	varset->variables.push_back(var);
	variablesets.push_back(varset);
	return var;
}

/**
 * Create an array of variables of size number
 */
std::vector<variable_p> LACompiler::variable(std::string name, int len, int number)
{
	auto varset = std::make_shared<variableset_t>(name);
	std::vector<variable_p> vars;
	for(int i=0; i<number; i++)
	{
		auto var = variable_impl(name + std::to_string(i), len);
		vars.push_back(var);
		varset->variables.push_back(var);
	}
	variablesets.push_back(varset);
	return vars;
}



void function_t::call(callable_p callable, std::vector<variable_p> args)
{
	// Check that we've got the right number of arguments
	if(args.size() != callable->num_args)
	{
    std::stringstream err;
		err << fmt::format("Expected {} arguments for function {}, but {} were provided",
						callable->num_args, callable->name, args.size());
		throw std::runtime_error(err.str());
	}

	// Check that each argument is the right size
	for(int i=0; i<args.size(); i++)
	{
		if(args[i]->len != callable->input_sizes[i])
		{
		    std::stringstream err;
			err << fmt::format("Error calling {}: Argument {} should have length {}, but {} is of length {}",
							callable->name, i, callable->input_sizes[i], args[i]->name, args[i]->len);
			throw std::runtime_error(err.str());
		}
	}

	calls.push_back(std::make_shared<call_t>(callable, args));
}

int function_t::num_outputs()
{
	auto fold = [](int acc, call_p call){return acc + call->num_outputs();};
    return std::accumulate(calls.begin(), calls.end(), 0, fold);
}

/**
 * Return true if coeff(row,col) == null and false otherwise
 */
template<typename T>
bool isNull(const Eigen::SparseMatrix<T>& mat, int row, int col)
{
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, col); it; ++it) {
        if (it.row() == row) return false;
    }
    return true;
}

/**
 * Copy the blk matrix into the M matrix at location row, column
 */
void copy_block(Eigen::SparseMatrix<int> &M, const Eigen::SparseMatrix<int> &blk, int row, int column)
{
	for(int r=0; r<blk.rows(); r++)
		for(int c=0; c<blk.cols(); c++)
		{
			if(!isNull(blk, r, c) && isNull(M, row+r, column+c))
				M.insert(row + r, column + c) = blk.coeff(r, c);
		}
}

/**
 * Build the jacobian sparsity structure for this function
 * 
 * Sets the valuePtr = 0..nnz
 */
Eigen::SparseMatrix<int> function_t::jacobianStructure()
{
	Eigen::SparseMatrix<int> S(num_outputs(), compiler.num_variables());

	// Write out sparsity structure by call
	int row = 0;
	for(auto& call: calls)
	{
		auto J = call->callable->jacobianStructure;

		int column = 0;
		for(auto& arg: call->args)
		{
			copy_block(S, J.middleCols(column, arg->len), row, arg->offset);
			column += arg->len;
		}
		row += J.rows();
	}

	S.makeCompressed();
	for(int i=0; i<S.nonZeros(); i++)
		S.valuePtr()[i] = i;

	return S;
}

/**
 * Build the hessian sparsity structure for this function
 * 
 * Sets the valuePtr = 0..nnz
 */
Eigen::SparseMatrix<int> function_t::hessianStructure()
{
	Eigen::SparseMatrix<int> H(compiler.num_variables(), compiler.num_variables());

	// Write out sparsity structure by call
	for(auto& call: calls)
	{
		for(int i=0; i<call->callable->num_outputs; i++)
		{
			auto h = call->callable->hessianStructure[i];

			int arg1_offset = 0;
			for(auto& arg1: call->args)
			{
				int arg2_offset = 0;
				for(auto& arg2: call->args)
				{
					copy_block(H, h.block(arg1_offset,arg2_offset,arg1->len,arg2->len), arg1->offset, arg2->offset);
					arg2_offset += arg2->len;
				}
				arg1_offset += arg1->len;
			}
		}
	}

	H.makeCompressed();
	for(int i=0; i<H.nonZeros(); i++)
		H.valuePtr()[i] = i;

	return H;
}




/**
 * Human-readable outputs
 */
std::ostream &operator<<(std::ostream &os, std::vector<variable_p> const &vars) 
{
	std::vector<std::string> varnames;
	transform(vars.begin(), vars.end(), back_inserter(varnames), 
		[](auto var){return var->name;});

    std::copy(std::begin(varnames),
              std::end(varnames),
              std::experimental::make_ostream_joiner(os, ", "));

    return os;
}
std::ostream &operator<<(std::ostream &os, variable_p const &var)
{
	os << var->name;
	return os;
}
std::ostream &operator<<(std::ostream &os, variableset_p const &var)
{
	os << var->str();
	return os;
}
std::ostream &operator<<(std::ostream &os, std::vector<variableset_p> const &vars)
{
	std::vector<std::string> varnames;
	transform(vars.begin(), vars.end(), back_inserter(varnames), 
		[](auto var){return var->str();});

    std::copy(std::begin(varnames),
              std::end(varnames),
              std::experimental::make_ostream_joiner(os, ", "));

	return os;
}

std::ostream &operator<<(std::ostream &os, callable_t const &callable)
{
	os << fmt::format("{} with {} args of size (", callable.name, callable.num_args);
    std::copy(std::begin(callable.input_sizes),
              std::end(callable.input_sizes),
              std::experimental::make_ostream_joiner(os, ", "));
    os << fmt::format(") and {} outputs", callable.num_outputs);
	return os;
}
std::ostream &operator<<(std::ostream &os, callable_p const callable)
{
	os << callable.get();
	return os;
}
std::ostream &operator<<(std::ostream &os, std::vector<callable_p> const &callables)
{
	IndentStream o;
	o << "Callables:\n" << IndentStream::indent;
	for(auto call: callables)
		o << call << std::endl;
	os << o.str();
	return os;
}

std::ostream &operator<<(std::ostream &os, call_t const &call)
{
	os << call.callable->name << "(";

	std::vector<std::string> varnames;
	transform(call.args.begin(), call.args.end(), back_inserter(varnames), 
		[](auto var){return var->name;});

    std::copy(std::begin(varnames),
              std::end(varnames),
              std::experimental::make_ostream_joiner(os, ", "));
    os << ")";

    return os;
}
std::ostream &operator<<(std::ostream &os, call_p const &call)
{
	os << *(call.get());
	return os;
}

std::ostream &operator<<(std::ostream &os, function_p const &f)
{
	IndentStream o;
	o << fmt::format("Function {} calls {} callables and has {} outputs:\n", f->name, f->calls.size(), f->num_outputs());
	o << IndentStream::indent;
	for(auto &call : f->calls)
		o << call << std::endl;
	o << IndentStream::outdent;
	os << o.str();
	return os;
}

std::ostream &operator<<(std::ostream &os, callable_info const &callable)
{
	IndentStream o;
	o << "callable_info : " << callable.name << std::endl;
	o << IndentStream::indent;
	o << "signature : " << callable.signature << std::endl;
	o << fmt::format("num_args = {} num_outputs = {}\n", callable.num_args, callable.num_outputs);
	o << "Input sizes:";
	for(auto sz: callable.input_sizes) o << sz << " ";
	o << std::endl;
	o << "Jacobian structure\n" << callable.jacobianStructure << std::endl;
	o << "Hessian structure\n";
	for(auto h: callable.hessianStructure) o << h << "\n";
	o << std::endl << IndentStream::outdent;

	os << o.str();
	return os;
}


/********************************************************
 * Generation code
 *********************************************************/

/**
 * Generate a function that will initialize a sparse matrix to the given structure
 */
std::string generate_sparse_init(Eigen::SparseMatrix<int> &J, std::string funcName, std::string matrixName="J")
{
	IndentStream o;

	J.makeCompressed();

	// First we generate a function that will initialize the jacobian
	o << fmt::format("static void {}(Eigen::SparseMatrix<scalar_t> &{})\n", funcName, matrixName)
	  << "{\n";

  o << IndentStream::indent;

	o << fmt::format("{}.resize({},{});\n", matrixName, J.rows(), J.cols());
	o << fmt::format("{}.reserve({});\n", matrixName, J.nonZeros());

	o << fmt::format("typedef Eigen::Triplet<scalar_t> T;\n");
	o << fmt::format("std::array<T,{}> tripletList = {{T", J.nonZeros());

	std::vector<std::string> args;
	for (int k=0; k < J.outerSize(); ++k)
    for (Eigen::SparseMatrix<int>::InnerIterator it(J,k); it; ++it)
    	args.push_back(fmt::format("{{{},{},1}}", it.row(), it.col()));

	std::copy(args.begin(), args.end(),
  	        std::experimental::make_ostream_joiner(o.get_stream(), ","));
	o << "};\n";
	o << fmt::format("{}.setFromTriplets(tripletList.begin(), tripletList.end());\n", matrixName);

	o << IndentStream::outdent << "}\n";
	return o.str();
}

/**
 * Call func (which returns a string) on each element in vec, and then join
 * the result in a comma seperated list
 */
template<typename F, typename T>
std::string make_csv(std::vector<T> &vec, F func)
{
	std::vector<std::string> args;
	for(auto &elem: vec) args.push_back(func(elem));

	std::stringstream o;
	std::copy(args.begin(), args.end(),
	          std::experimental::make_ostream_joiner(o, ","));
	return o.str();
}

/**
 * Generate an array named name of sparseblock_info
 */
std::string function_t::generate_sequence(std::string name, std::vector<std::vector<seqinfo>> &blocks)
{
	std::stringstream o;
	int len=0;
	for(auto &blk: blocks) len += blk.size();
	o << fmt::format("static constexpr seqinfo {}[{}] = {{\n  ", name, len);

	std::vector<std::string> args;
	for(auto &blk: blocks)
		args.push_back(make_csv(blk, [](seqinfo seq){return fmt::format("{{{},{}}}", seq.index, seq.length);}));
	
	std::copy(args.begin(), args.end(),
	          std::experimental::make_ostream_joiner(o, ",\n  "));

	o << "};\n";

	// Add the definition of this array to outside the class
	o_postfix << fmt::format("constexpr seqinfo {}::{}::{}[];\n", compiler.className, this->name, name);

	return o.str();
}
std::string function_t::generate_sequence(std::string name, std::vector<seqinfo> &blocks)
{
	std::vector<std::vector<seqinfo>> tmp = {blocks};
	return generate_sequence(name, tmp);
}


/**
 * Generate code for all the functions in this problem
 */
std::string LACompiler::generate()
{
	IndentStream o;

	o << fmt::format("#ifndef __{}_HPP\n#define __{}_HPP\n", className, className);
	o << "struct " << className << "\n{\n";
	o << IndentStream::indent;
	o << fmt::format("static constexpr int num_variables = {};\n", num_variables());
	o << fmt::format("using param_t = {};\n", param_t);
	o << fmt::format("using scalar_t = {};\n", scalar_t);
	o << "using variable_t = Eigen::Vector<scalar_t, num_variables>;\n";
	o << "\n";

	// Generate accessors for all the variables
	o << "// Variable accessors\n";
	for(auto &varset: variablesets)
	{
		if(varset->variables.size() == 1)
			o << fmt::format("static Eigen::Ref<Eigen::Vector<scalar_t, {}>> {}(Eigen::Ref<variable_t> var) {{return var.template segment<{}>({});}};\n",
				varset->variables[0]->len,
				varset->name,
				varset->variables[0]->len,
				varset->variables[0]->offset);
		else
		{
			o << fmt::format("static Eigen::Ref<Eigen::Vector<scalar_t, {}>> {}(Eigen::Ref<variable_t> var, int ind) {{return var.template segment<{}>({}+{}*ind);}};\n",
				varset->variables[0]->len,
				varset->name,
				varset->variables[0]->len,
				varset->variables[0]->offset,
				varset->variables[0]->len);
			std::string MatType = fmt::format("Eigen::Matrix<scalar_t, {}, {}>", varset->variables[0]->len, varset->variables.size());
			o << fmt::format("static Eigen::Ref<{}> {}(Eigen::Ref<variable_t> var) {{return Eigen::Map<{}>(var.template segment<{}>({}).data());}};\n",
				MatType,
				varset->name,
				MatType,
				varset->variables[0]->len * varset->variables.size(),
				varset->variables[0]->offset);
		}
	}

	// Define the signatures of all the callables
	o << "\n// Define convenience names for all differentiable functions\n";
	for(auto callable: callables)
		o << fmt::format("using {} = {};\n", callable->name, callable->signature);

	for(auto &func: functions) 
	{
		o << "\n" << func->generate() << "\n";
		o_postfix << func->o_postfix.str();
		func->o_postfix.str("");
		func->o_postfix.clear();
	}
	for(auto &wsum: weightedsums) 
	{
		o << "\n" << wsum->generate() << "\n";
		o_postfix << wsum->o_postfix.str();
		wsum->o_postfix.str("");
		wsum->o_postfix.clear();
	}
	o << o_hook.str();
	o_hook.str("");
	o_hook.clear();

	o << IndentStream::outdent;
	o << "};\n";
	o << o_postfix.str(); o_postfix.str(""); o_postfix.clear();
	o << "#endif\n";
	return o.str();
}



/**
 * Create a struct to evaluate this function
 */
std::string function_t::generate()
{
	IndentStream o;

	o << fmt::format("struct {} : public function_util_t<scalar_t, Eigen::Vector<scalar_t, {}>, Eigen::SparseMatrix<scalar_t>>\n",
				name, num_outputs());
	o << "\n{\n";
	o << IndentStream::indent;
	o << fmt::format("using out_t = Eigen::Vector<scalar_t, {}>;\n", num_outputs());
	o << "using jacobian_t = Eigen::SparseMatrix<scalar_t>;\n";
	o << "using hessian_t = Eigen::SparseMatrix<scalar_t>;\n";

	o << "\n";
	o << generate_eval() << "\n";
	o << generate_jacobian() << "\n";
	o << o_hook.str(); o_hook.str(""); o_hook.clear();
	o << IndentStream::outdent;

	o << "};\n";

	return o.str();
}


std::string arglist(std::vector<variable_p> args)
{
	std::vector<std::string> args_str;
  	std::vector<std::string> arg_names;
	for(auto &arg : args)
	{
		args_str.push_back(fmt::format("x.SEG({},{})", arg->len, arg->offset));
		arg_names.push_back(arg->name);
	}
	return fmt::format("{}", fmt::join(args_str, ", "));	
}

/**
 * Produce code to evaluate this function
 * 
 * The call produced will have the form
 * 
 * eval(param_t param, variable_t x, out_t &out)
 * 
 */
std::string function_t::generate_eval()
{
	IndentStream o;
	o << "/**\n"
		<< " * Evalute the function for the parameter param and return the result in out\n"
		<< " */\n";
	o << "static void eval(param_t &param, const Eigen::Ref<const variable_t> &x, Eigen::Ref<out_t> out)\n"
		<< "{\n" << IndentStream::indent;
	int offset = 0; // Output offset
	for(auto call : calls)
	{
		int size = call->callable->num_outputs;
		std::string name = call->callable->name;
		o << fmt::format("out.SEG({},{}) = {}::eval(param, {});", size, offset,	name, arglist(call->args));
		o << " // " << call << "\n";
		offset += size;
	}
	o << IndentStream::outdent << "};" << std::endl;
	return o.str();
}

/**
 * Produce code to evaluate the jacobian of this function
 * 
 * The call produced will have the form
 * 
 * eval(param_t param, variable_t x, out_t &out, jacobian_t &jac)
 * 
 */
std::string function_t::generate_jacobian()
{
	IndentStream o;

	// Get the jacobian structure
	auto J = jacobianStructure();


	o << generate_sparse_init(J, "initialize_jacobian");
	o << "\n";

	o << "/** \n"
	  << " * Compute the jacobian of the overall function\n"
		<< " */\n";

	// Store the sequence of copies to fill in the jacobian
	std::vector<std::vector<seqinfo>> sequence;
	int row = 0;
	for(auto& call: calls)
	{
		auto partition = args_to_partion(call->args);
		std::vector<seqinfo> row_partition = {{row,call->num_outputs()}};
		sequence.push_back(build_copy_sequence(J, call->callable->jacobianStructure, row_partition, partition));
		row += call->num_outputs();
	}			
	o << generate_sequence("jac_seq", sequence);

	o << "static void eval(param_t &param, variable_t x, out_t &out, jacobian_t &jacobian)\n"
	  << "{\n";

	o << IndentStream::indent;

	int offset = 0; // Output offset
	int call_index = 0;
	int seq_offset = 0; // Offset into the jac_seq
	for(int i=0; i<calls.size(); i++)
	{
		o << fmt::format("setJ(out, jacobian, {}, jac_seq+{}, {}, {}::jac(param, {})); // ",
							offset, seq_offset,
							sequence[i].size(),
							calls[i]->callable->name,
							arglist(calls[i]->args));
			o << calls[i] << "\n";

		offset += calls[i]->callable->num_outputs;
		seq_offset += sequence[i].size();
		call_index++;
	}

	o << IndentStream::outdent;
	o << "};" << std::endl;
	o << "\n";
	return o.str();
}


/**
 * Create a struct to evaluate this function
 */
std::string weightedsum_t::generate()
{
	IndentStream o;

	o << fmt::format("struct {} : public weightedsum_util_t<scalar_t, Eigen::Vector<scalar_t, num_variables>, Eigen::Vector<scalar_t, {}>>",
		name, num_outputs());
	o << "\n{\n";
	o << IndentStream::indent;
	o << fmt::format("using weight_t = Eigen::Vector<scalar_t, {}>;\n", num_outputs());
	o << "using gradient_t = Eigen::Vector<scalar_t, num_variables>;\n";
	o << "using hessian_t = Eigen::SparseMatrix<scalar_t>;\n";

	o << "\n";
	o << generate_eval() << "\n";
	o << generate_gradient() << "\n";
	o << generate_hessian() << "\n";
	o << IndentStream::outdent;

	o << "};\n";

	return o.str();
}


/**
 * Produce code to evaluate this function
 * 
 * The call produced will have the form
 * 
 * val = eval(param_t param, weight_t weight, variable_t x)
 * 
 */
std::string weightedsum_t::generate_eval()
{
	IndentStream o;
	o << "/**\n"
	  << " * Evalute the function for the parameter param and return the result in out\n"
	  << " */\n";
	o << "static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> &w, const Eigen::Ref<const variable_t> &x)\n"
	  << "{\n" << IndentStream::indent;

	o << "scalar_t val = 0;\n";
	int offset = 0; // Output offset
	for(auto call : calls)
	{
		// Write out the C++ format
		int size = call->callable->num_outputs;
		std::string name = call->callable->name;
		o << fmt::format("val += w.SEG({},{}).dot({}::eval(param, {})); // ", 
										size, offset, name, arglist(call->args));
		o << call << "\n";
		offset += size;
	}
	o << "return val;\n";
	o << IndentStream::outdent << "};" << std::endl;
	return o.str();
}


/**
 * Produce code to evaluate the gradient of this weighted sum
 * 
 * val = eval(param_t param, weight_t w, variable_t x, gradient_t &jac)
 * 
 */
std::string weightedsum_t::generate_gradient()
{
	IndentStream o;

	o << "/** \n"
	  << " * Compute the gradient of the weighted sum\n"
		<< " */\n";

	IndentStream oo;

	oo << "static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient)\n"
	  << "{\n" << IndentStream::indent
	  << "gradient.array() = 0;\n"
	  << "scalar_t val = 0;\n";

	int offset = 0; // Output offset
	int sequence_offset = 0;
	std::vector<std::vector<seqinfo>> sequence;
	for(auto call : calls)
	{
		int size = call->callable->num_outputs;
		std::string name = call->callable->name;

		oo << fmt::format("accGrad(val, gradient, grad_seq+{}, {}, w.SEG({},{}), {}::jac(param, {}));",
			sequence_offset,
			call->callable->num_args,
			size, offset, 
			name,
			arglist(call->args));
		oo << " // " << call << "\n";

		std::vector<seqinfo> blocks;
		for(auto arg: call->args)
			blocks.push_back(seqinfo{.index=arg->offset, .length=arg->len});
		sequence.push_back(blocks);
		sequence_offset += blocks.size();

		offset += size;
	}
	oo << "return val;\n";
	oo << IndentStream::outdent << "};" << std::endl;

  o << generate_sequence("grad_seq", sequence) << oo.str();

	o << "\n";
	return o.str();
}


/*
 * Produce code to evaluate the hessian of this function
 * 
 * The call produced will have the form
 * 
 * eval(param_t param, variable_t x, out_t &out, jacobian_t &J, hessian_t &H)
 */ 
std::string weightedsum_t::generate_hessian()
{
	IndentStream o;

	// auto J = jacobianStructure();
	auto H = hessianStructure();

	o << "/**\n"
	  << " * Initialize the hessian of the function\n"
	  << " */\n";
	o << generate_sparse_init(H, "initialize_hessian", "H");
	o << "\n";

	o << "/**\n"
	  << " * Copy the hessian of <w, f> into the right place\n"
	  << " * \n"
	  << " * Input:\n"
	  << " *   hessian_return_t (value, jacobian and hessian of the vector-valued function f)\n"
	  << " * \n"
	  << " * Output:\n"
	  << " *   gradient += w' * jacobian f(x) \n"
	  << " *   value += w' * f(x)\n"
	  << " *   hessian += sum wi * hessian fi(x)\n"
	  << " */\n";

	IndentStream oo;
	oo << "static scalar_t eval(param_t &param, const Eigen::Ref<const weight_t> w, const Eigen::Ref<const variable_t> x, Eigen::Ref<gradient_t> gradient, Eigen::Ref<hessian_t> hessian)\n"
	   << "{\n" << IndentStream::indent
	   << "gradient.array() = 0;\n"
	   << "scalar_t val = 0;\n"
	   << "auto ptr = hessian.valuePtr();\n"
	   << "for(int i=0; i<hessian.nonZeros(); i++) ptr[i] = 0;\n";

	// Iterate over each call computing the copy sequence
	std::vector<std::vector<seqinfo>> sequence;
	int offset = 0;
	int gradient_offset = 0;
	int hessian_offset = 0;
	std::vector<std::string> seq_lengths;
	for(auto& call: calls)
	{
		// Compute the hessian for each output in turn
		std::vector<seqinfo> hessian_call_sequence; // Sequence for the entire set of hessians
		for(int i=0; i<call->callable->num_outputs; i++)
		{
			auto h = call->callable->hessianStructure[i];
			auto partition = args_to_partion(call->args);
			auto seq = build_copy_sequence(H, h, partition, partition);
			hessian_call_sequence.insert(hessian_call_sequence.end(), seq.begin(), seq.end());
			seq_lengths.push_back(fmt::format("{}",seq.size()));
		}
		sequence.push_back(hessian_call_sequence);

		oo << fmt::format("accHessian(val, gradient, hessian, grad_seq+{}, {}, ", 
						gradient_offset, call->callable->num_args)
			 << fmt::format("hessian_seq+{}, hessian_seq_len+{}, ",	
			 			hessian_offset, offset)
			 << fmt::format("w.SEG({},{}), ",
			 			call->callable->num_outputs, offset)
			 << fmt::format("{}::hessian(param, {}));\n",
			 			call->callable->name, arglist(call->args));

 		gradient_offset += call->callable->num_args;
 		hessian_offset += hessian_call_sequence.size();
 		offset += call->callable->num_outputs;
	}
	oo << "return val;\n" << IndentStream::outdent << "}\n";
	o << generate_sequence("hessian_seq", sequence);
	o << "static constexpr int hessian_seq_len[" << seq_lengths.size() << "] = {";
  std::copy(std::begin(seq_lengths),
            std::end(seq_lengths),
            std::experimental::make_ostream_joiner(o.get_stream(), ","));
  o << "};\n";
  o_postfix << fmt::format("constexpr int {}::{}::hessian_seq_len[];\n", compiler.className, this->name);
	o << oo.str();

	return o.str();
}

