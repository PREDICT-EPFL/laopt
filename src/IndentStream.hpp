#ifndef __INDENT_STREAM__HPP
#define __INDENT_STREAM__HPP

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

#endif