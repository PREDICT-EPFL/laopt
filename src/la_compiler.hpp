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

#include <experimental/iterator>

#include "lampc_utility.hpp"

struct LACompiler
{
	std::string className; // Name of the class that will be generated

	struct variable_t
	{
		std::string name;
		int len;

		LACompiler &prob; // Reference to parent problem

		variable_t(std::string name_, int len_, LACompiler *ptr)
			: name(name_), len(len_), prob(*ptr)
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

		callable_t(std::string signature_, std::string name_, int num_args_)
			: signature(signature_), name(name_), num_args(num_args_)
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
			return callable_t(std::string(type_name<F>()), F::name, F::num_input_vars);
		}
	};
	std::vector<callable_t> callables;

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
	};

	friend std::ostream &operator<<(std::ostream &os, LACompiler::variables_t const &variables);
	friend std::ostream &operator<<(std::ostream &os, std::shared_ptr<LACompiler::function_t> const &function);
	friend std::ostream &operator<<(std::ostream &os, LACompiler const &la);

	variables_t variables;
	std::vector<std::shared_ptr<function_t>> functions;

	LACompiler(std::string className_)
		: className(className_)
		{};

	/**
	 * Create a variable
	 */
	std::shared_ptr<variable_t> variable(std::string name, int len)
	{
		auto var = std::make_shared<variable_t>(name, len, this);
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
std::ostream &operator<<(std::ostream &os, LACompiler::variables_t const &variables) 
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
std::ostream &operator<<(std::ostream &os, std::shared_ptr<LACompiler::function_t> const &function_ptr) 
{ 
	LACompiler::function_t &f = *function_ptr;
	os << f.name << " [" << f.calls.size() << " function calls]" << std::endl;
	for(auto const &call : f.calls)
	{
		os << "\t" << f.parent.callables[call.callable].name << "(" << call.args << ")" << std::endl;
	}
    return os;
}

/**
 * Display a summary of the problem
 */
std::ostream &operator<<(std::ostream &os, LACompiler const &la) 
{ 
	int maxRowLen = 30; // Tunable parameter
	int len = (maxRowLen + 2 - 2 - la.className.length())/2;

	os << std::string(len, '=') << " " << la.className << " "
	   << std::string(maxRowLen - la.className.length() + 2 - 2 - len, '=') << "\n";

	os << "Variables:\n  ";
	int rowlen = 0;
	std::string s;
	std::stringstream vars;
	vars << la.variables; // Comma seperated list
	while(vars >> s) // Wrap list
	{
		if(s.length() + rowlen > maxRowLen)
		{
			os << "\n  ";
			rowlen = 0;
		}	
		os << s << " ";
		rowlen += s.length();
	}
	os << "\n\n";
	os << "Functions:\n" << std::endl;

	for(auto const &func : la.functions)
		os << "  " << func << "\n";

	os << std::endl;
	return os;
}


/**
 * Compiled version of the problem
 */
template<typename scalar_t, typename param_t>
struct LAProblem
{
	std::string className;

	/**
	 * Information about the input variables
	 */
	// struct variable_info_t
	// {
	// 	int offset;
	// 	int size;
	// 	std::string name;
	// };
	std::vector<variable_info_t> variables;
	int numVariables; // Number of vector variables
	int inputSize; // Total length of input vector

	/**
	 * Information about the functions
	 */
	struct function_info_t
	{
		std::string name;
		int numFunctionCalls;
		std::vector<int> functionCalls;
		std::vector<std::vector<int>> call_args;

		int totalNumArgs;
		std::vector<int> functionArguments;

		// Hessian information
    int hessian_nnzBlocks;
    int hessian_nnzBlockColumns;
    int hessian_nnzEstimate;
	};

	std::vector<function_info_t> functions;

	/**
	 * Callable information
	 */
	struct callable_t
	{
		std::string signature;
		std::string name;
		int num_args;

		// Structure of the hessian wrt args of the callable
		Eigen::SparseMatrix<bool> hessian;
		Eigen::SparseMatrix<bool> jacobian;
	};
	std::vector<callable_t> callables;

	LAProblem(LACompiler &data)
	{
		className = data.className;

		for(auto const &c : data.callables)
		{
			callable_t callable;
			callable.signature = c.signature;
			callable.name = c.name;
			callable.num_args = c.num_args;
			callables.push_back(callable);
		}

		int offset = 0;
		for(auto const &v : data.variables)
		{
			variables.push_back({
				.offset = offset,
				.size = v->len,
				.name = v->name});
			offset += v->len;
		}
		inputSize = std::accumulate(variables.begin(), variables.end(), 
							0, [](int acc, auto const& v){return acc + v.size;});
		numVariables = data.variables.size();

		// Iterate over each of the functions
		for(auto const &f : data.functions)
		{
			function_info_t info;
			info.name = f->name;
			info.numFunctionCalls = f->calls.size();

			// Compute all the arguments in order for the calling sequence
			for(auto const &c : f->calls)
			{
				info.functionCalls.push_back(c.callable);
				std::vector<int> args;
				for(auto const &arg : c.args)
				{
					auto it = std::find_if(variables.begin(), variables.end(), [&](auto const &v){
						return v.name == arg->name;
					});
					assert(it != variables.end() && "Unknown variable");
					int index = std::distance(variables.begin(), it);
					info.functionArguments.push_back(index);

					args.push_back(index);
				}
				info.call_args.push_back(args);
			}
			info.totalNumArgs = info.functionArguments.size();

			// Compute the hessian structure
			for(auto const &c : f->calls)
			{

			}			

			functions.push_back(info);
		}
	}




	void toFile(std::ostream &o)
	{
		o << "struct " << className << "\n"
		  << "{\n"
		  << "  using scalar_t = " << type_name<scalar_t>() << ";\n"
		  << "  using param_t = typename " << type_name<param_t>() << ";\n"
		  << "  using functions = std::tuple<\n";
		  for(auto it=callables.begin(); it != callables.end(); it++)
		  {
		o << "    " << it->signature;
		if(it == callables.end() - 1) o << ">;\n\n";
		else o << ",\n";
		  }

		// Variable information
		o << "struct variables_info\n"
		  << "{\n"
		  << "  static const int inputSize = " << inputSize << ";\n"
		  << "  static const int numVariables = " << numVariables << ";\n"
		  << "  static constexpr LA::variable_info_t variable_info[]\n"
		  << "  {\n";
		bool first = true;
		for(auto v=variables.begin(); v!=variables.end(); v++)
		{
		o << "    {.offset = " << v->offset << ", .size = " << v->size << "}";
		if(v != variables.end() - 1) o << ", "; else o << "  ";
		o << "// " << v->name << "\n";
		}
		o << "  };\n"
		  << "};\n\n";
		o << "using variables_t = Eigen::Vector<scalar_t, variables_info::inputSize>;\n\n";

		for(auto const &f : functions)
		{
		o << "struct " << f.name << "_info\n"
		  << "{\n"
		  << "  static constexpr int numFunctionCalls = " << f.numFunctionCalls << ";\n"
		//   << "  static constexpr int functionCalls[] = {";
	 //    std::copy(f.functionCalls.begin(),
	 //              f.functionCalls.end(),
	 //              std::experimental::make_ostream_joiner(o, ", "));
		// o << "};\n"
  		  << "  \n"
  		  << "  static constexpr int totalNumArgs = " << f.totalNumArgs << ";\n"
		  << "  static constexpr int functionArguments[] = {\n";
		auto it = f.functionArguments.begin();
		int offset = 0;
		for(int i=0; i<f.numFunctionCalls; i++)
		{
			o << "    ";
			auto orig_it = it;
			for(int j=0; j < f.call_args[i].size(); j++)
			{
				o << *it;
				if(it != f.functionArguments.end()-1) o << ", ";
				it++;
			}

			// Give the names of the arguments for readability
			o << "  // ";
			it = orig_it;
			for(int j=0; j < f.call_args[i].size(); j++)
			{
				o << variables[*it].name << ", ";
				it++;
			}
		    o << "\n";
		}
		o << "  };\n";
		o << "};\n";

		o << "using " << f.name << "_t = LA::Function<scalar_t, param_t, functions,\n";
		o << "  variables_info, " << f.name << "_info,\n";
		o << "  ";
	    std::copy(f.functionCalls.begin(),
	              f.functionCalls.end(),
	              std::experimental::make_ostream_joiner(o, ", "));
	    o << ">; // Function call sequence\n";

		}
		o << "};\n";
	}
};

#endif