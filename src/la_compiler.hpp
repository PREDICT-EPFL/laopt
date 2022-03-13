/**
 * Compile an LA Problem into data structures that can be incorporated into a solver.
 * 
 */

#ifndef __LACOMPILER_HPP
#define __LACOMPILER_HPP

#include "lampc.hpp"
#include <experimental/iterator>

/**
 * callable_t 
 * - defines a c++ callable function
 * - information about how to generate code to call the function, variable sizes, etc
 * - this is a function that can compute its value, jacobian and hessian
 * 
 * call_t
 * - defined a callable that has been called
 * - link to the callable and to the list of variables that it's being called on
 * 
 * constraint_t
 * - a call that also contains an upper and lower bound
 * 
 * function_t
 * - a list of call's that are being concantenated into a single function
 * 
 * weightedsum_t
 * - a weighted sum of the rows of a function_t
 * 
 */

// Forward declarations
struct variable_t;
struct variableset_t;
struct callable_t;
struct function_t;
struct weightedsum_t;
struct LACompiler;
struct callable_info;
struct call_t;
struct constraint_t;

using variable_p = std::shared_ptr<variable_t>;
using variableset_p = std::shared_ptr<variableset_t>;
using callable_p = std::shared_ptr<callable_t>;
using call_p = std::shared_ptr<call_t>;
using constraint_p = std::shared_ptr<constraint_t>;
using function_p = std::shared_ptr<function_t>;
using weightedsum_p = std::shared_ptr<weightedsum_t>;

/**
 * A function is a list of callables evaluated at a set of arguments
 * 
 * f = [f1(x,y); f2(x,z); f3(z,x); ...]
 * 
 */
class function_t
{
public:
	function_t(std::string _name, LACompiler &compiler) : name(_name), compiler(compiler) {};

	// Add a call to the function
	void call(callable_p callable, std::vector<variable_p> args);
	void add_call(call_p call) {calls.push_back(call);}

	int num_outputs();
	Eigen::SparseMatrix<int> jacobianStructure();
	Eigen::SparseMatrix<int> hessianStructure();

	friend std::ostream &operator<<(std::ostream &os, function_p const &f);
	friend LACompiler;

// protected:
	// List of functions with the arguments
	std::vector<call_p> calls;
	std::string name;

	LACompiler &compiler;

	// Generate code to evaluate the function
	std::string generate();

	std::string generate_eval();
	std::string generate_jacobian();

	std::ostringstream o_postfix;
	std::string generate_sequence(std::string name, std::vector<std::vector<seqinfo>> &blocks);
	std::string generate_sequence(std::string name, std::vector<seqinfo> &blocks);	
};

/**
 * Represents a weighted sum of calls sum <w_i, f_i(x)>
 */
class weightedsum_t : public function_t
{
public:
	weightedsum_t(std::string name, LACompiler &compiler) : function_t(name, compiler) {};

	friend LACompiler;

protected:
	std::string generate();

	std::string generate_eval();
	std::string generate_gradient();
	std::string generate_hessian();
};

struct LACompiler
{
	// Name of the class, scalar type and parameter structure used in the generated code
	std::string className; 
	std::string param_t;
	std::string scalar_t;

	std::vector<variable_p> variables;
	std::vector<variableset_p> variablesets;
	std::vector<function_p> functions;
	std::vector<weightedsum_p> weightedsums;
	std::vector<callable_p> callables;

	LACompiler(std::string className, std::string param_t, std::string scalar_t) 
		:	className(className), param_t(param_t), scalar_t(scalar_t)
		{};

	// Generate code for the problem
	std::string generate(); 

	// Create variables
	variable_p variable(std::string name, int len); 
	std::vector<variable_p> variable(std::string name, int len, int number);

	// Create callable and function
	callable_p callable(callable_info info);
	// param_callable_p parametric_function(parametric_function_info info);

	function_p function(std::string name);
	weightedsum_p weighted_sum(std::string name);

	// Get information about the problem
	int num_variables();

	// Any material that should be placed before and after the generated code
	// Set before calling generate
	std::ostringstream o_postfix;

protected:
	variable_p variable_impl(std::string name, int len);
};

// Human-readable display of problem components
std::ostream &operator<<(std::ostream &os, variable_p const &var);
std::ostream &operator<<(std::ostream &os, std::vector<variable_p> const &vars);
std::ostream &operator<<(std::ostream &os, std::vector<variableset_p> const &vars);
std::ostream &operator<<(std::ostream &os, callable_p const callable);
std::ostream &operator<<(std::ostream &os, std::vector<callable_p> const &callables);
std::ostream &operator<<(std::ostream &os, call_p const &call);
std::ostream &operator<<(std::ostream &os, call_t const &call);

// Information about a runtime callable
struct callable_info
{
	std::string signature;
	std::string name;
	int num_args;
	int num_outputs;
	std::vector<int> input_sizes;

	Eigen::SparseMatrix<int> jacobianStructure;
	std::vector<Eigen::SparseMatrix<int>> hessianStructure;

	// Pull out the required info from the template type F for the callable
	template<typename F>
	callable_info(F)
	{
		signature = std::string(type_name<F>());
		name = std::string(F::name);
		num_args = int(F::num_input_vars);
		num_outputs = int(F::num_outputs);

		jacobianStructure = F::jacobianStructure();

		for(int i=0; i<F::num_outputs; i++)
			hessianStructure.push_back(F::hessianStructure(i));

		input_sizes = F::get_input_sizes();
	}

	// Define an identity callable from the variable
	callable_info(variable_p var);
};

std::ostream &operator<<(std::ostream &os, callable_info const &callable);


/**
 * A called callable. 
 * i.e., we call the function callable at a specific set of arguments (variables) args
 */
struct call_t
{
	callable_t *callable;
	std::vector<variable_p> args;

	call_t(callable_t *callable, std::vector<variable_p> args)
		: callable(callable), args(args) {};

	call_t(callable_p callable, std::vector<variable_p> args)
		: callable(callable.get()), args(args) {};

	int num_outputs() const;

	friend std::ostream &operator<<(std::ostream &os, callable_p const &callable);
	friend std::ostream &operator<<(std::ostream &os, callable_t const &callable);

	// Human-readable
	std::string str() const
	{
		std::stringstream o;
		o << *this;
		return o.str();
	}
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

	LACompiler &compiler;

	Eigen::SparseMatrix<int> jacobianStructure;
	std::vector<Eigen::SparseMatrix<int>> hessianStructure; // Structure per output

	callable_t(std::string signature_, std::string name_, 
			   int num_args_, int num_outputs_, std::vector<int> input_sizes, LACompiler *compiler)
		: signature(signature_), name(name_), 
		  num_args(num_args_), num_outputs(num_outputs_),
		  input_sizes(input_sizes), compiler(*compiler)
	{}

	callable_t(callable_info &info, LACompiler *compiler)
		: signature(info.signature), name(info.name), 
		  num_args(info.num_args), num_outputs(info.num_outputs),
		  input_sizes(info.input_sizes), compiler(*compiler)
	{
		jacobianStructure = info.jacobianStructure;

		for(int i=0; i<num_outputs; i++)
			hessianStructure.push_back(info.hessianStructure[i]);
	}

	// bool operator==(const callable_t& rhs) const noexcept
	// {
	//     return (signature == rhs.signature) &&
	//     	   (name == rhs.name) &&
	//     	   (num_args == rhs.num_args) &&
	//     	   (num_outputs == rhs.num_outputs);
	// }

	template<typename... Vars>
	call_t operator()(const Vars... variables) 
	{
	    std::vector<variable_p>  args = { variables... };
		return call_t(this, args);
	}

};


#endif