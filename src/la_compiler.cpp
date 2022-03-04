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


#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <experimental/iterator>

#include "lampc.hpp"
#include "la_compiler.hpp"


struct IndentStream
{
	std::vector<std::shared_ptr<std::stringstream>> streams;

	IndentStream()
	{
		streams.push_back(std::make_shared<std::stringstream>());
	}

	template <typename T>
	IndentStream& operator<<(const T& x)
	{
	    *(this->streams.back()) << x;
	    return *this;
	}


	// function that takes a custom stream, and returns it
	typedef IndentStream& (*IndentStreamManipulator)(IndentStream&);

	// take in a function with the custom signature
	IndentStream& operator<<(IndentStreamManipulator manip)
	{
	    // call the function, and return its value
	    return manip(*this);
	}

	// define the indent manipulator for this stream.
	static IndentStream& indent(IndentStream& stream)
	{
		stream.streams.push_back(std::make_shared<std::stringstream>());
	    return stream;
	}

	// define the outdent manipulator for this stream.
	static IndentStream& outdent(IndentStream& stream)
	{
		assert(stream.streams.size() > 1 && "Too many outdents");
		if(stream.streams.size() == 1) return stream;
		auto o = stream.streams.back();
		stream.streams.pop_back();
		stream << add_tab(o->str());
	    return stream;
	}

	static std::string add_tab(const std::string &i)
	{
		std::ostringstream o;
		std::stringstream s(i);
		for(std::string line; std::getline(s, line); ) 
			o << "  " << line << "\n";
		return o.str();
	}

	// this is the type of std::cout
	typedef std::basic_ostream<char, std::char_traits<char> > CoutType;

	// this is the function signature of std::endl
	typedef CoutType& (*StandardEndLine)(CoutType&);

	// define an operator<< to take in std::endl
	IndentStream& operator<<(StandardEndLine manip)
	{
	    // call the function, but we cannot return it's value
	    manip(*(this->streams.back()));
	    return *this;
	}

	/**
	 * Unroll the indents and write to the previous level
	 */
	std::string str()
	{
		// Flatten the string structure
		std::string s;
		for(int i=streams.size()-1; i>=0; i--)
			s = streams[i]->str() + add_tab(s);
		return s;
	}

	/**
	 * Return a reference to the current stream
	 */
	std::stringstream& get_stream()
	{
		return *(this->streams.back());
	}
};



/**
 * Copy column source_column of source into the target_column of target starting at row target_row
 * 
 * Assumption: The elements of the target matrix are enumerated from 0 -> nnz in the valuePtr array
 * 
 */
int build_copy_sequence(Eigen::SparseMatrix<int> &_target, Eigen::SparseMatrix<int> &source, 
						int target_row, int target_column, int source_column,
						std::vector<sparseblock_info<int>> &blocks)
{
	assert( _target.isCompressed() && "_target matrix must be in compressed format" );
	assert( source.isCompressed() && "source matrix must be in compressed format" );

	Eigen::SparseMatrix<int> target(_target);
	for(int i=0; i<target.nonZeros(); i++)
		target.valuePtr()[i] = i;

	// Number of nonzeros in this column
	int nnz = source.outerIndexPtr()[source_column+1] - source.outerIndexPtr()[source_column];
	int source_start = source.outerIndexPtr()[source_column];

	// std::cout << fmt::format("source_column = {}\n", source_column);
	// std::cout << fmt::format("source_start = {}\n", source_start);
	// std::cout << fmt::format("nnz = {}\n", nnz);

	int num_blocks = 0; // Number of blocks pushed to blocks
	sparseblock_info<int> blk;

	int number_written = 0;
	int target_ind = target.outerIndexPtr()[target_column];
	int source_ind = source.outerIndexPtr()[source_column];
	while( number_written < nnz )
	{
		// std::cout << fmt::format("target_ind = {} source_ind = {}\n", target_ind, source_ind);
		// std::cout << fmt::format("target.innerIndexPtr()[target_ind] = {} source.innerIndexPtr()[source_ind] + target_row = {}\n",
		// 	target.innerIndexPtr()[target_ind],
		// 	source.innerIndexPtr()[source_ind] + target_row);

		// Advance the target index until source and target are synchronized
		while( target.innerIndexPtr()[target_ind] < source.innerIndexPtr()[source_ind] + target_row )
		{
			target_ind++;
			assert( target_ind <= target.nonZeros() && "Error: target index ran past the end of the matrix" );
		};
		blk.target_index = target.valuePtr()[target_ind];
		blk.block_length = 0;

		// std::cout << fmt::format("target.innerIndexPtr()[target_ind] = {} source.innerIndexPtr()[source_ind] + target_row = {}\n",
		// 	target.innerIndexPtr()[target_ind],
		// 	source.innerIndexPtr()[source_ind] + target_row);
		// std::cout << fmt::format("blk.target_index = {}\n", blk.target_index);

		// Run down the block as long as the source and target are synched
		while( target.innerIndexPtr()[target_ind] == source.innerIndexPtr()[source_ind] + target_row 
				&& number_written < nnz )
		{
			blk.block_length++;
			target_ind++;
			source_ind++;
			number_written++;

			assert( target_ind <= target.nonZeros() && "Error: target index ran past the end of the matrix" );
			assert( source_ind <= source.nonZeros() && "Error: source index ran past the end of the matrix" );
		};

		// Either we're done, or the target has more nonzeros than the source
		blocks.push_back(blk);
	};

	return number_written;
}

struct variable_t
{
	std::string name;
	int len;
	int offset;

	// compiler_t &prob; // Reference to parent problem

	variable_t(std::string name_, int len_, int offset_)
		: name(name_), len(len_), offset(offset_)
		{}
};

// A group of similarly-named variables
struct variableset_t
{
	std::string name;
	std::vector<variable_p> variables;

	variableset_t(std::string name) : name(name)	{};
};



// List of callable functions
// We store these as strings so that we can do a comparison
// of static types without having to instantiate and compare
// function pointers, since these may be instantiated multiple
// times.
struct callable_t
{
	std::string signature;
	std::string name;
	int num_args;
	int num_outputs;
	std::vector<int> input_sizes;

	Eigen::SparseMatrix<int> jacobianStructure;
	std::vector<Eigen::SparseMatrix<int>> hessianStructure; // Structure per output

	callable_t(std::string signature_, std::string name_, 
			   int num_args_, int num_outputs_, std::vector<int> input_sizes)
		: signature(signature_), name(name_), 
		  num_args(num_args_), num_outputs(num_outputs_),
		  input_sizes(input_sizes)
	{}

	callable_t(callable_info &info)
		: signature(info.signature), name(info.name), 
		  num_args(info.num_args), num_outputs(info.num_outputs),
		  input_sizes(info.input_sizes)
	{
		jacobianStructure = info.jacobianStructure;

		for(int i=0; i<num_outputs; i++)
			hessianStructure.push_back(info.hessianStructure[i]);
	}

	bool operator==(const callable_t& rhs) const noexcept
	{
	    return (signature == rhs.signature) &&
	    	   (name == rhs.name) &&
	    	   (num_args == rhs.num_args) &&
	    	   (num_outputs == rhs.num_outputs);
	}
};


callable_p LACompiler::callable(callable_info info)
{
	auto callable = std::make_shared<callable_t>(info);
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
		auto var = variable_impl(name + "_" + std::to_string(i), len);
		vars.push_back(var);
		varset->variables.push_back(var);
	}
	variablesets.push_back(varset);
	return vars;
}


/**
 * A called callable. 
 * i.e., we call the function callable at a specific set of arguments (variables) args
 */
struct call_t
{
	callable_p callable;
	std::vector<variable_p> args;

	call_t(callable_p callable, std::vector<variable_p> args)
		: callable(callable), args(args) {};

	int num_outputs()	{ return callable->num_outputs; }

	friend std::ostream &operator<<(std::ostream &os, callable_p const &callable);
};


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

std::ostream &operator<<(std::ostream &os, callable_p const &callable)
{
	os << fmt::format("{} with {} args of size (", callable->name, callable->num_args);
    std::copy(std::begin(callable->input_sizes),
              std::end(callable->input_sizes),
              std::experimental::make_ostream_joiner(os, ", "));
    os << fmt::format(") and {} outputs", callable->num_outputs);
	return os;
}
std::ostream &operator<<(std::ostream &os, std::vector<callable_p> const &callables)
{
	IndentStream o;
	o << "Callables:\n" << IndentStream::indent;
	for(auto &call: callables)
		o << call << std::endl;
	os << o.str();
	return os;
}

std::ostream &operator<<(std::ostream &os, call_p const &call)
{
	os << call->callable->name << "(";

	std::vector<std::string> varnames;
	transform(call->args.begin(), call->args.end(), back_inserter(varnames), 
		[](auto var){return var->name;});

    std::copy(std::begin(varnames),
              std::end(varnames),
              std::experimental::make_ostream_joiner(os, ", "));
    os << ")";

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
	o << fmt::format("{}.reserve({});\n\n", matrixName, J.nonZeros());

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
 * Generate code for all the functions in this problem
 */
std::string LACompiler::generate()
{
	IndentStream o;

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

	for(auto &func: functions) o << "\n" << func->generate() << "\n";
	for(auto &wsum: weightedsums) o << "\n" << wsum->generate() << "\n";
	o << IndentStream::outdent;

	o << "};\n";
	return o.str();
}



/**
 * Create a struct to evaluate this function
 */
std::string function_t::generate()
{
	IndentStream o;

	o << "struct " << name << "\n{\n";
	o << IndentStream::indent;
	// o << "using variable_t = Eigen::Vector<scalar_t, num_variables>;\n"; 
	o << fmt::format("using out_t = Eigen::Vector<scalar_t, {}>;\n", num_outputs());
	o << "using jacobian_t = Eigen::SparseMatrix<scalar_t>;\n";
	o << "using hessian_t = Eigen::SparseMatrix<scalar_t>;\n";

	o << "\n";
	o << generate_eval() << "\n\n";
	o << generate_jacobian() << "\n\n";
	// o << generate_hessian() << "\n\n";
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
	std::vector<int> sequence_call_offset; // Offset into blocks for a given call
	std::vector<sparseblock_info<int>> blocks;
	int row = 0;
	for(auto& call: calls)
	{
		sequence_call_offset.push_back(blocks.size());

		auto source = call->callable->jacobianStructure;

		// Determine where each of the columns of the submatrix source should go
		// in the full jacobian matrix J
		int column = 0;
		for(auto& arg: call->args)
		{
		    for(int c=0; c<arg->len; c++)
		        build_copy_sequence(J, source, row, arg->offset+c, column+c, blocks);
			column += arg->len;
		}
		row += source.rows();
	}			
	sequence_call_offset.push_back(blocks.size());

	o << fmt::format("static constexpr sparseblock_info<int> jac_seq[{}] = {{", blocks.size());
	std::string join = "";
	for(auto &blk: blocks)
	{
		o << fmt::format("{}{{{},{}}}", join, blk.target_index, blk.block_length);
		join = ",";
	}
	o << "};\n"
	  << "template<int len, typename jacobian_output_t>\n"
	  << "static inline void setJ(out_t &out, jacobian_t &jacobian, // Values to write into\n"
	  << "         const int offset, // Offset into out for the evaluation\n"
      << "         const int sequence_offset, // Offset into jacobian copy sequence\n"
	  << "         const int num_blocks, \n"
	  << "         const jacobian_output_t &J) // Input\n"
	  << "{\n"
	  << "  out.template segment<len>(offset) = J.val;\n"
	  << "  copy_submatrix<scalar_t>(jacobian, J.jacobian, jac_seq + sequence_offset, num_blocks);\n"
	  << "}\n\n";

	o << "/**\n"
	  << " * Evaluate the function and its jacobian\n"
	  << " *\n"
	  << " * jacobian must have been initialized with the function initialize_jacobian\n"
	  << " */\n";
	o << "static void eval(param_t &param, variable_t x, out_t &out, jacobian_t &jacobian)\n"
	  << "{\n";

	o << IndentStream::indent;

	int offset = 0; // Output offset
	int call_index = 0;
	for(auto call : calls)
	{
		// Write out the C++ format
		int size = call->callable->num_outputs;
		std::string name = call->callable->name;

			o << fmt::format("setJ<{}>(out, jacobian, {}, {}, {}, {}::jac(param, {})); // ",
								size, offset, 
								sequence_call_offset[call_index],
								sequence_call_offset[call_index+1] - sequence_call_offset[call_index],
								name,
								arglist(call->args));
			o << call << "\n";

		offset += size;
		call_index++;
	}

	o << IndentStream::outdent;
	o << "};" << std::endl;
	o << "\n\n";
	return o.str();
}


/**
 * Create a struct to evaluate this function
 */
std::string weightedsum_t::generate()
{
	IndentStream o;

	o << "struct " << name << "\n{\n";
	o << IndentStream::indent;
	// o << "using variable_t = Eigen::Vector<scalar_t, num_variables>;\n"; 
	o << fmt::format("using weight_t = Eigen::Vector<scalar_t, {}>;\n", num_outputs());
	o << "using gradient_t = Eigen::Vector<scalar_t, num_variables>;\n";
	o << "using hessian_t = Eigen::SparseMatrix<scalar_t>;\n";

	o << "\n";
	o << generate_eval() << "\n\n";
	o << generate_gradient() << "\n\n";
	// o << generate_hessian() << "\n\n";
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
	int call_index = 0;
	std::vector<std::string> blocks;
	for(auto call : calls)
	{
		// Write out the C++ format
		int size = call->callable->num_outputs;
		std::string name = call->callable->name;

		oo << fmt::format("setGrad<{}>(val, gradient, {}, {}, w.SEG({},{}), {}::jac(param, {}));",
			size, blocks.size(),
			call->callable->num_args,
			size, offset, 
			name,
			arglist(call->args));
		oo << " // " << call << "\n";

		for(auto arg: call->args)
			blocks.push_back(fmt::format("{{{},{}}}", arg->offset, arg->len));

		offset += size;
		call_index++;
	}
	oo << "return val;\n";
	oo << IndentStream::outdent << "};" << std::endl;

	// Write out the gradient sequence
	// We need this to be copied directly into the generated file so that the function can see
	// the grad_seq. I think this is because its constexpr and we're using static functions.
	std::stringstream os;
	os << fmt::format("static constexpr sparseblock_info<int> grad_seq[{}] = {{", blocks.size());
  std::copy(std::begin(blocks),
            std::end(blocks),
            std::experimental::make_ostream_joiner(os, ","));
  os << "};\n";
	os << "template<int len, typename scalar_t, typename gradient_t, typename weight_t, typename jacobian_output_t>\n";
	os << "static inline void setGrad(scalar_t &val, gradient_t &grad, \n";
	os << "        int seq_offset, int num_vars, // Offsets of the vars into grad\n";
	os << "        const weight_t &w, \n";
	os << "        const jacobian_output_t &J)\n";
	os << "{\n";
	os << "  val += w.dot(J.val);\n";
	os << "  auto g = w.transpose() * J.jacobian;\n";
	os << "  int offset = 0;\n";
	os << "  int varlen = 0;\n";
	os << "  for(int i=0; i<num_vars; i++)\n";
	os << "  {\n";
	os << "   varlen = grad_seq[seq_offset+i].block_length;\n";
	os << "   grad.segment(grad_seq[seq_offset+i].target_index, varlen) += g.segment(offset, varlen);\n";
	os << "   offset += varlen;\n";
	os << "  }\n";
	os << "}\n";

  o << os.str() << oo.str();

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

	auto J = jacobianStructure();
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

	// Store the sequence of copies to fill in the hessian for each function
	std::vector<int> sequence_call_offset; // Offset into blocks for a given call
	std::vector<sparseblock_info<int>> blocks;
	for(auto& call: calls)
	{
		sequence_call_offset.push_back(blocks.size());

		for(int i=0; i<call->callable->num_outputs; i++)
		{
			auto h = call->callable->hessianStructure[i];

			int arg1_offset = 0;
			for(auto& arg1: call->args)
			{
				int arg2_offset = 0;
				for(auto& arg2: call->args)
				{
					Eigen::SparseMatrix<int> blk = Eigen::MatrixX<int>(h.block(arg1_offset,arg2_offset,arg1->len,arg2->len)).sparseView();
					for(int column=0; column<arg2->len; column++)
						build_copy_sequence(H, 
							blk, 
							arg1->offset, arg2->offset,
							column, blocks);
					arg2_offset += arg2->len;
				}
				arg1_offset += arg1->len;
			}
		}
	}
	sequence_call_offset.push_back(blocks.size());

	o << fmt::format("static constexpr sparseblock_info<int> hessian_seq[{}] = {{", blocks.size());
	std::string join = "";
	for(auto &blk: blocks)
	{
		o << fmt::format("{}{{{},{}}}", join, blk.target_index, blk.block_length);
		join = ",";
	}
	o << "};\n";

	o << "template<int len, typename hessian_output_t>\n"
	  << "static inline void setH(scalar_t &value, variable_t &gradient, hessian_t &hessian, // Values to write into\n"
	  << "         const Eigen::Ref<const Eigen::Vector<scalar_t, len>> w,\n" // Function is <w, fi(x)>
      << "         const int sequence_offset, // Offset into hessian copy sequence\n"
	  << "         const int num_blocks[len], \n"
	  << "         const hessian_output_t &H) // Input\n"
	  << "{\n"
	  << "  value += w.dot(H.val);\n"
	  << "  gradient += w.transpose() * H.jacobian;\n"
	  << "  for(int i=0; i<len; i++)\n"
	  << "  {\n"
	  << "    copy_submatrix<scalar_t>(hessian, H.hessian, hessian_seq + sequence_offset, num_blocks[i]);\n"
	  << "    sequence_offset += num_blocks[i];\n"
	  << "  }\n"
	  << "}\n\n";




	return o.str();
}

