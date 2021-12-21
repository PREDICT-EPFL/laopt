#include <iostream>
#include <tuple>
#include <array>

struct BaseBlock
{
	virtual inline int getSize() = 0;
};

// template<int rows>
// struct Block final : BaseBlock
// {
// 	inline int getSize() override
// 	{
// 		return rows * 2;
// 	} 
// };

struct Block final : public BaseBlock
{
	const int rows;
	Block(int rows_) : rows(rows_) {}

	virtual inline int getSize() override
	{
		return rows * 2;
	} 
};

struct OtherBlock final : public BaseBlock
{
	const int rows;
	OtherBlock(int rows_) : rows(rows_) {}

	virtual inline int getSize() override
	{
		return rows * 3;
	} 
};


// template<typename range>
// struct Test;

template<int N>
struct Test
{
	std::array<BaseBlock*, N> myarray;

	// std::tuple<Block<ind>...> tup; // Static memory here

	Test()
	{
		// for(int i=0; i<N; i++)
		// 	myarray[i] = new Block(i);

		for(int i=0; i<N/2; i++)
			myarray[i] = new Block(i);
		for(int i=N/2; i<N; i++)
			myarray[i] = new OtherBlock(i);

	}

	int test()
	{
		int acc = 0;
		for(int j=0; j<1000000; j++)
			for(int i=0; i<N; i++)
			{
				acc += myarray[i]->getSize();
			}
		return acc;
	}
};

int main()
{
	std::cout << "Hello world\n";

	const std::size_t N = 1000;
	// Test<std::make_integer_sequence<std::size_t, N>> test;

	Test<N> test;

	std::cout << "Size = " << test.test() << std::endl;

	return 0;
}


// template<typename Derived>
// struct VarTrait;

// template<typename Derived>
// struct VarBase
// {
// 	static const std::size_t len = VarTrait<Derived>::len;
// };

// template<std::size_t len_>
// struct Var : VarBase<Var<len_>>
// {
// 	using Base = VarBase<Var<len_>>;
// 	static const std::size_t len = VarTrait<Var<len_>>::len;
// };

// template<std::size_t len_>
// struct VarTrait<Var<len_>>
// {
// 	static const std::size_t len = len_;
// };


// template<typename Tuple>
// struct BSMatrix;

// template<typename... rows>
// struct BSMatrix<std::tuple<rows...>>
// {
// 	using rowTuple = std::tuple<rows...>;
// 	rowTuple m_rows;

// 	template<typename row>
// 	std::size_t printme()
// 	{
// 		return row::len;
// 	}

// 	std::size_t test()
// 	{
// 		std::size_t sum = 0;
// 	    (void)std::initializer_list<int>{
// 	        (
// 	        	sum += printme<rows>(),
// 	            0
// 	        )...
// 	    };
// 	    return sum;
// 	}
// };

// template<typename range>
// struct Test;

// template<std::size_t... ind>
// struct Test<std::integer_sequence<std::size_t, ind...>>
// {
// 	template<std::size_t i>
// 	struct x : Var<i> {};

// 	BSMatrix<std::tuple<x<ind>...>> B;

// 	void test()
// 	{
// 		std::cout << "Sum of lengths = " << B.test() << std::endl;
// 	}
// };

// int main()
// {
// 	std::cout << "Hello world\n";

// 	const std::size_t N = 1000;
// 	Test<std::make_integer_sequence<std::size_t, N>> test;

// 	test.test();
// 	return 0;
// }
