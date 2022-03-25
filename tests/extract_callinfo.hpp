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
