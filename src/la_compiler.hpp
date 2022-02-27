/**
 * Compile an LA Problem into data structures that can be incorporated into a solver.
 * 
 */

#ifndef __LACOMPILER_HPP
#define __LACOMPILER_HPP

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

/**
 * Add a tab before each line in i and write the result to o
 */
std::string indent(const std::string &i)
{
	std::ostringstream o;
	std::stringstream s(i);
  for(std::string line; std::getline(s, line); ) 
  	o << "  " << line << "\n";
  return o.str();
}

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

template<typename Functions>
struct LACompiler
{
	std::string className; // Name of the class that will be generated

	using scalar_t = typename Functions::scalar_t;
	using param_t = typename Functions::param_t;

	struct variable_t
	{
		std::string name;
		int len;
		int offset;

		LACompiler &prob; // Reference to parent problem

		variable_t(std::string name_, int len_, LACompiler *ptr, int offset_)
			: name(name_), len(len_), prob(*ptr), offset(offset_)
			{}
	};
	using variables_t = std::vector<std::shared_ptr<variable_t>>;

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

		Eigen::SparseMatrix<int> jacobianStructure;

		callable_t(std::string signature_, std::string name_, int num_args_, int num_outputs_)
			: signature(signature_), name(name_), num_args(num_args_), num_outputs(num_outputs_)
		{}

		bool operator==(const callable_t& rhs) const noexcept
		{
		    return (signature == rhs.signature) &&
		    	   (name == rhs.name) &&
		    	   (num_args == rhs.num_args);
		}

		template<typename F>
		static callable_t create()
		{
			auto callable = callable_t(std::string(type_name<F>()), F::name, F::num_input_vars, F::num_outputs);
			callable.jacobianStructure = F::jacobianStructure();
			return callable;
		}
	};
	std::vector<callable_t> callables;

	/**
	 * Generate code for all the functions in this problem
	 */
	std::string generate()
	{
		std::stringstream o;

		o << "struct " << className << "\n{\n";
		o << indent(fmt::format("using variable_t = Eigen::Vector<{}, {}>;\n", 
									type_name<scalar_t>(), num_variables()));

		o << indent(fmt::format("using param_t = {};\n", std::string(type_name<param_t>())));
		o << indent(fmt::format("using scalar_t = {};\n", std::string(type_name<scalar_t>())));

		// Define the signatures of all the callables
		o << indent("\n// Define convenience names for all differentiable functions\n");
		for(auto &callable: callables)
			o << indent(fmt::format("using {} = {};\n", callable.name, callable.signature));

		for(auto &func: functions)
		{
			o << "\n" << indent(func->generate()) << "\n";
		}
		// o << indent(generate_eval()) << "\n\n";
		// o << indent(generate_jacobian()) << "\n\n";

		o << "};\n";
		return o.str();
	}


	/**
	 * A function of a list of callables
	 * 
	 * f = [f1(x,y); f2(x,z); f3(z,x); ...]
	 * 
	 */
	struct function_t
	{
		struct call_t
		{
			int callable; // Index into callables
			variables_t args;
		};

		// List of functions with the arguments
		std::vector<call_t> calls;
		std::string name;

		LACompiler &parent;

		function_t(std::string name_, LACompiler &parent_) : name(name_), parent(parent_) {}

		template<typename F>
		void add(F, std::initializer_list<std::shared_ptr<variable_t>> args)
		{
			// Check that we've got the right number of arguments
			if(args.size() != F::num_input_vars)
			{
			    std::stringstream err;
				err << "Expected " << F::num_input_vars << " arguments for function " 
					<< type_name<typename F::Func>() << ", but " << args.size() << " were provided.";
				throw std::runtime_error(err.str());
			}

			// Check that each argument is the right size
			for(int i=0; i<args.size(); i++)
			{
				// TODO: Pack the func::input_sizes into a tuple and compare against args
			}

			// Find the index of the callable, or add it to the list
			auto callable = callable_t::template create<F>();
		  auto it = find(parent.callables.begin(), parent.callables.end(), callable);
		 
		 	int index = -1;
		    if (it != parent.callables.end())
		        index = it - parent.callables.begin();
		    else {
		    	parent.callables.push_back(callable);
		    	index = parent.callables.size() - 1;
		    }

			// Add a call to the callable
			calls.push_back(call_t{.callable = index, .args = args});
		}

		/**
		 * Produce code to evaluate this function
		 * 
		 * The call produced will have the form
		 * 
		 * eval(param_t param, variable_t x, out_t &out)
		 * 
		 */
		std::string generate_eval()
		{
	    std::ostringstream o;
	    o << "/**\n"
	      << " * Evalute the function for the parameter param and return the result in out\n"
	      << " */\n";
			o << "static void eval(param_t param, variable_t x, out_t &out)\n"
			  << "{\n";
			  int offset = 0; // Output offset
				for(auto call : calls)
				{
					// Write out the C++ format
					int size = parent.callables[call.callable].num_outputs;
					std::string name = parent.callables[call.callable].name;
					o << fmt::format("  out.SEG({},{}) = {}::eval(param, ", 
													size, offset,	name);
					offset += size;

					// Build the argument list
					std::vector<std::string> args_str;
				  std::vector<std::string> arg_names;
					for(auto &arg : call.args)
					{
						args_str.push_back(fmt::format("x.SEG({},{})", arg->len, arg->offset));
						arg_names.push_back(arg->name);
					}
					o << fmt::format("{}", fmt::join(args_str, ", "));
					o << ");";

					// Write out the args in human-readable format
					o << " // " << fmt::format("{}", fmt::join(arg_names, ", ")) << "\n";
				}

			o << "};" << std::endl;
			return o.str();
		}

		int num_outputs()
		{
			  int num_outputs = 0;
				for(auto call : calls)
					num_outputs += parent.callables[call.callable].num_outputs;
				return num_outputs;
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
		void copy_block(Eigen::SparseMatrix<int> &M, 
										const Eigen::SparseMatrix<int> &blk, 
										int row, int column)
		{
			for(int r=0; r<blk.rows(); r++)
				for(int c=0; c<blk.cols(); c++)
				{
					if(!isNull(blk, r, c))
						M.insert(row + r, column + c) = blk.coeff(r, c);
				}
		}

		/**
		 * Build the jacobian sparsity structure for this function
		 * 
		 * Sets the valuePtr = 0..nnz
		 */
		Eigen::SparseMatrix<int> build_jacobian()
		{
			Eigen::SparseMatrix<int> S(num_outputs(), parent.num_variables());

			// Write out sparsity structure by call
			int row = 0;
			for(auto& call: calls)
			{
				auto &callable = parent.callables[call.callable];
				auto J = callable.jacobianStructure;

				// std::cout << "jacobian structure of " << callable.name << " = \n";
				// std::cout << J << std::endl;

				int column = 0;
				for(auto& arg: call.args)
				{
					// std::cout << fmt::format("arg {} len/offset {} / {}", arg->name, arg->len, arg->offset) << std::endl;
					// std::cout << fmt::format("column / arg->len = {} / {}", column, arg->len) << std::endl;
					// std::cout << J.middleCols(column, arg->len) << std::endl;
					copy_block(S, J.middleCols(column, arg->len), row, arg->offset);
					column += arg->len;
				}
				row += J.rows();
			}

			for(int i=0; i<S.nonZeros(); i++)
				S.valuePtr()[i] = i;

			return S;
		}

		/**
		 * Generate a function that will initialize a sparse matrix to the given structure
		 */
		std::string generate_sparse_init(Eigen::SparseMatrix<int> &J, std::string funcName)
		{
			std::stringstream o;

			J.makeCompressed();

			// First we generate a function that will initialize the jacobian
			o << fmt::format("static void {}(Eigen::SparseMatrix<scalar_t> &J)\n", funcName)
			  << "{\n";
			{
				std::stringstream oo;
				oo << fmt::format("J.resize({},{});\n", num_outputs(), parent.num_variables());
				oo << fmt::format("J.reserve({});\n\n", J.nonZeros());

				oo << fmt::format("typedef Eigen::Triplet<{}> T;\n", type_name<scalar_t>());
				oo << fmt::format("std::array<T,{}> tripletList = {{T", J.nonZeros());

				std::vector<std::string> args;
		    for (int k=0; k < J.outerSize(); ++k)
		        for (Eigen::SparseMatrix<int>::InnerIterator it(J,k); it; ++it)
		        	args.push_back(fmt::format("{{{},{},1}}", it.row(), it.col()));

		    std::copy(args.begin(), args.end(),
		              std::experimental::make_ostream_joiner(oo, ","));
		    oo << "};\n";
		    oo << "J.setFromTriplets(tripletList.begin(), tripletList.end());\n";

				o << indent(oo.str());
			}
			o << "}\n";
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
		std::string generate_jacobian()
		{
			std::stringstream o;

			// Get the jacobian structure
			auto J = build_jacobian();

			o << generate_sparse_init(J, "initialize_jacobian");
			o << "\n";

			o << "/** \n"
			  << " * Copy the LAMPC_Function output into the right place in the jacobian\n"
     		  << " */\n";

			// Store the sequence of copies to fill in the jacobian
			std::vector<int> sequence_call_offset; // Offset into blocks for a given call
			std::vector<sparseblock_info<int>> blocks;
			int row = 0;
			for(auto& call: calls)
			{
				sequence_call_offset.push_back(blocks.size());

				auto &callable = parent.callables[call.callable];
				auto source = callable.jacobianStructure;

				// Determine where each of the columns of the submatrix source should go
				// in the full jacobian matrix J
				int column = 0;
				for(auto& arg: call.args)
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
			o << "static void eval(param_t param, variable_t x, out_t &out, jacobian_t &jacobian)\n"
			  << "{\n";

			std::stringstream oo;

			int offset = 0; // Output offset
			int call_index = 0;
			for(auto call : calls)
			{
				// Write out the C++ format
				int size = parent.callables[call.callable].num_outputs;
				std::string name = parent.callables[call.callable].name;

      			oo << fmt::format("setJ<{}>(out, jacobian, {}, {}, {}, {}::jac(param, ",
      				size, offset, 
      				sequence_call_offset[call_index],
      				sequence_call_offset[call_index+1] - sequence_call_offset[call_index],
      				name);

				offset += size;

				// Build the argument list
				std::vector<std::string> args_str;
			    std::vector<std::string> arg_names;
				for(auto &arg : call.args)
				{
					args_str.push_back(fmt::format("x.SEG({},{})", arg->len, arg->offset));
					arg_names.push_back(arg->name);
				}
				oo << fmt::format("{}", fmt::join(args_str, ", "));
				oo << "));\n";

				call_index++;
			}

			o << indent(oo.str());
			o << "};" << std::endl;

			return o.str();
		}


		/**
		 * Write out the files required to evaluate this function to a struct in the file o
		 */
		std::string generate()
		{
			std::stringstream o;

			o << "struct " << name << "\n{\n";
			o << fmt::format("  using variable_t = Eigen::Vector<{}, {}>;\n", 
										type_name<scalar_t>(), parent.num_variables());
			o << fmt::format("  using out_t = Eigen::Vector<{}, {}>;\n", 
										type_name<scalar_t>(), num_outputs());
			o << fmt::format("  using jacobian_t = Eigen::SparseMatrix<{}>;\n", 
										type_name<scalar_t>());

			o << "\n";
			o << indent(generate_eval()) << "\n\n";
			o << indent(generate_jacobian()) << "\n\n";

			o << "};\n";

			return o.str();
		}
	};

	friend std::ostream &operator<<(std::ostream &os, LACompiler::variables_t const &variables);
	friend std::ostream &operator<<(std::ostream &os, std::shared_ptr<LACompiler::function_t> const &function);
	friend std::ostream &operator<<(std::ostream &os, LACompiler const &la);

	variables_t variables;
	std::vector<std::shared_ptr<function_t>> functions;

	LACompiler(std::string className_)
		: className(className_)
		{};

	/** Returns the total number of variables defined
	 */
	int num_variables()
	{
		if(variables.size() == 0) 
			return 0;
		
		return variables.back()->offset + variables.back()->len;
	}

	/**
	 * Create a variable
	 */
	std::shared_ptr<variable_t> variable(std::string name, int len)
	{
		auto var = std::make_shared<variable_t>(name, len, this, num_variables());
		variables.push_back(var);
		return var;
	}

	/**
	 * Create an array of variables of size number
	 */
	std::vector<std::shared_ptr<variable_t>> variable(std::string name, int len, int number)
	{
		std::vector<std::shared_ptr<variable_t>> vars;
		for(int i=0; i<number; i++)
			vars.push_back(variable(name + "_" + std::to_string(i), len));
		return vars;
	}

	/**
	 * Create a new function
	 */
	std::shared_ptr<function_t> function(std::string name)
	{
		auto function = std::make_shared<function_t>(name, *this);
		functions.push_back(function);
		return function;
	}

};

/**
 * List the variables of the problem in order
 */
template<typename Functions>
std::ostream &operator<<(std::ostream &os, typename LACompiler<Functions>::variables_t const &variables) 
{
	std::vector<std::string> varnames;
	transform(variables.begin(), variables.end(), back_inserter(varnames), 
		[](auto var){return var->name;});

    std::copy(std::begin(varnames),
              std::end(varnames),
              std::experimental::make_ostream_joiner(os, ", "));

    return os;
}

/**
 * Pretty-print a function call
 */
// template<typename Functions>
// std::ostream &operator<<(std::ostream &os, std::shared_ptr<typename LACompiler<Functions>::function_t> const &function_ptr) 
// { 
// 	LACompiler<Functions>::function_t &f = *function_ptr;
// 	os << f.name << " [" << f.calls.size() << " function calls]" << std::endl;
// 	for(auto const &call : f.calls)
// 	{
// 		os << "\t" << f.parent.callables[call.callable].name << "(" << call.args << ")" << std::endl;
// 	}
//     return os;
// }

// /**
//  * Display a summary of the problem
//  */
// template<typename Functions>
// std::ostream &operator<<(std::ostream &os, typename LACompiler<Functions> const &la) 
// { 
// 	int maxRowLen = 30; // Tunable parameter
// 	int len = (maxRowLen + 2 - 2 - la.className.length())/2;

// 	os << std::string(len, '=') << " " << la.className << " "
// 	   << std::string(maxRowLen - la.className.length() + 2 - 2 - len, '=') << "\n";

// 	os << "Variables:\n  ";
// 	int rowlen = 0;
// 	std::string s;
// 	std::stringstream vars;
// 	vars << la.variables; // Comma seperated list
// 	while(vars >> s) // Wrap list
// 	{
// 		if(s.length() + rowlen > maxRowLen)
// 		{
// 			os << "\n  ";
// 			rowlen = 0;
// 		}	
// 		os << s << " ";
// 		rowlen += s.length();
// 	}
// 	os << "\n\n";
// 	os << "Functions:\n" << std::endl;

// 	for(auto const &func : la.functions)
// 		os << "  " << func << "\n";

// 	os << std::endl;
// 	return os;
// }




// /**
//  * Compiled version of the problem
//  */
// template<typename scalar_t, typename param_t>
// struct LAProblem
// {
// 	std::string className;

// 	/**
// 	 * Information about the input variables
// 	 */
// 	// struct variable_info_t
// 	// {
// 	// 	int offset;
// 	// 	int size;
// 	// 	std::string name;
// 	// };
// 	std::vector<variable_info_t> variables;
// 	int numVariables; // Number of vector variables
// 	int inputSize; // Total length of input vector

// 	/**
// 	 * Information about the functions
// 	 */
// 	struct function_info_t
// 	{
// 		std::string name;
// 		int numFunctionCalls;
// 		std::vector<int> functionCalls;
// 		std::vector<std::vector<int>> call_args;

// 		int totalNumArgs;
// 		std::vector<int> functionArguments;

// 		// Hessian information
//     int hessian_nnzBlocks;
//     int hessian_nnzBlockColumns;
//     int hessian_nnzEstimate;
// 	};

// 	std::vector<function_info_t> functions;

// 	/**
// 	 * Callable information
// 	 */
// 	struct callable_t
// 	{
// 		std::string signature;
// 		std::string name;
// 		int num_args;

// 		// Structure of the hessian wrt args of the callable
// 		Eigen::SparseMatrix<bool> hessian;
// 		Eigen::SparseMatrix<bool> jacobian;
// 	};
// 	std::vector<callable_t> callables;

// 	LAProblem(LACompiler &data)
// 	{
// 		className = data.className;

// 		for(auto const &c : data.callables)
// 		{
// 			callable_t callable;
// 			callable.signature = c.signature;
// 			callable.name = c.name;
// 			callable.num_args = c.num_args;
// 			callables.push_back(callable);
// 		}

// 		int offset = 0;
// 		for(auto const &v : data.variables)
// 		{
// 			variables.push_back({
// 				.offset = offset,
// 				.size = v->len,
// 				.name = v->name});
// 			offset += v->len;
// 		}
// 		inputSize = std::accumulate(variables.begin(), variables.end(), 
// 							0, [](int acc, auto const& v){return acc + v.size;});
// 		numVariables = data.variables.size();

// 		// Iterate over each of the functions
// 		for(auto const &f : data.functions)
// 		{
// 			function_info_t info;
// 			info.name = f->name;
// 			info.numFunctionCalls = f->calls.size();

// 			// Compute all the arguments in order for the calling sequence
// 			for(auto const &c : f->calls)
// 			{
// 				info.functionCalls.push_back(c.callable);
// 				std::vector<int> args;
// 				for(auto const &arg : c.args)
// 				{
// 					auto it = std::find_if(variables.begin(), variables.end(), [&](auto const &v){
// 						return v.name == arg->name;
// 					});
// 					assert(it != variables.end() && "Unknown variable");
// 					int index = std::distance(variables.begin(), it);
// 					info.functionArguments.push_back(index);

// 					args.push_back(index);
// 				}
// 				info.call_args.push_back(args);
// 			}
// 			info.totalNumArgs = info.functionArguments.size();

// 			// Compute the hessian structure
// 			for(auto const &c : f->calls)
// 			{

// 			}			

// 			functions.push_back(info);
// 		}
// 	}




// 	void toFile(std::ostream &o)
// 	{
// 		o << "struct " << className << "\n"
// 		  << "{\n"
// 		  << "  using scalar_t = " << type_name<scalar_t>() << ";\n"
// 		  << "  using param_t = typename " << type_name<param_t>() << ";\n"
// 		  << "  using functions = std::tuple<\n";
// 		  for(auto it=callables.begin(); it != callables.end(); it++)
// 		  {
// 		o << "    " << it->signature;
// 		if(it == callables.end() - 1) o << ">;\n\n";
// 		else o << ",\n";
// 		  }

// 		// Variable information
// 		o << "struct variables_info\n"
// 		  << "{\n"
// 		  << "  static const int inputSize = " << inputSize << ";\n"
// 		  << "  static const int numVariables = " << numVariables << ";\n"
// 		  << "  static constexpr LA::variable_info_t variable_info[]\n"
// 		  << "  {\n";
// 		bool first = true;
// 		for(auto v=variables.begin(); v!=variables.end(); v++)
// 		{
// 		o << "    {.offset = " << v->offset << ", .size = " << v->size << "}";
// 		if(v != variables.end() - 1) o << ", "; else o << "  ";
// 		o << "// " << v->name << "\n";
// 		}
// 		o << "  };\n"
// 		  << "};\n\n";
// 		o << "using variables_t = Eigen::Vector<scalar_t, variables_info::inputSize>;\n\n";

// 		for(auto const &f : functions)
// 		{
// 		o << "struct " << f.name << "_info\n"
// 		  << "{\n"
// 		  << "  static constexpr int numFunctionCalls = " << f.numFunctionCalls << ";\n"
// 		//   << "  static constexpr int functionCalls[] = {";
// 	 //    std::copy(f.functionCalls.begin(),
// 	 //              f.functionCalls.end(),
// 	 //              std::experimental::make_ostream_joiner(o, ", "));
// 		// o << "};\n"
//   		  << "  \n"
//   		  << "  static constexpr int totalNumArgs = " << f.totalNumArgs << ";\n"
// 		  << "  static constexpr int functionArguments[] = {\n";
// 		auto it = f.functionArguments.begin();
// 		int offset = 0;
// 		for(int i=0; i<f.numFunctionCalls; i++)
// 		{
// 			o << "    ";
// 			auto orig_it = it;
// 			for(int j=0; j < f.call_args[i].size(); j++)
// 			{
// 				o << *it;
// 				if(it != f.functionArguments.end()-1) o << ", ";
// 				it++;
// 			}

// 			// Give the names of the arguments for readability
// 			o << "  // ";
// 			it = orig_it;
// 			for(int j=0; j < f.call_args[i].size(); j++)
// 			{
// 				o << variables[*it].name << ", ";
// 				it++;
// 			}
// 		    o << "\n";
// 		}
// 		o << "  };\n";
// 		o << "};\n";

// 		o << "using " << f.name << "_t = LA::Function<scalar_t, param_t, functions,\n";
// 		o << "  variables_info, " << f.name << "_info,\n";
// 		o << "  ";
// 	    std::copy(f.functionCalls.begin(),
// 	              f.functionCalls.end(),
// 	              std::experimental::make_ostream_joiner(o, ", "));
// 	    o << ">; // Function call sequence\n";

// 		}
// 		o << "};\n";
// 	}
// };

#endif