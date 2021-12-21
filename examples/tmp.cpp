#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "type_name.hpp"
#include <iostream>
#include <tuple>


// class MatrixXd_t : public  Eigen::MatrixXd
// {
// public:
//     MatrixXd_t(void) : Eigen::MatrixXd() {}

//     typedef Eigen::MatrixXd Base;

//     // This constructor allows you to construct MatrixXd_t from Eigen expressions
//     template<typename OtherDerived>
//     MatrixXd_t(const Eigen::MatrixBase<OtherDerived>& other)
//         : Eigen::MatrixXd(other)
//     { }

//     // This method allows you to assign Eigen expressions to MatrixXd_t
//     template<typename OtherDerived>
//     MatrixXd_t & operator= (const Eigen::MatrixBase <OtherDerived>& other)
//     {
//         this->Base::operator=(other);
//         return *this;
//     }
// };

// class MatrixBlockBase
// {
// 	virtual void 
// }

// template<typename Matrix_t>
// class MatrixBlock : public MatrixBlockBase
// {
// 	Matrix_t& M;

// 	MatrixBlock(Matrix_t& _M) : M(_M) {}
// }

// template<int _Rows, int _Cols, typename... Block_t>
// class BSMatrix
// {

// 	std::tuple<Block_t&...> blocks;

// 	// Map from (row, column) => blocks tuple index or -1 if null
// 	Eigen::Matrix<int, _Rows, _Cols> block_ptr;

// public:

// 	BSMatrix(std::array<std::pair<int, int>, sizeof...(Block_t)> block_locations,
// 		Block_t&... _blocks) : blocks{_blocks...}
// 	{
// 		block_ptr.array() = -1; // Default to zero matrices
// 		for(int ind=0; ind<block_locations.size(); ind++)
// 		{
// 			int row, col;
// 			std::tie<int, int>(row, col) = block_locations[ind];
// 			block_ptr(row, col) = ind;
// 		}
// 	}

// 	template<int 

// 	void print()
// 	{
// 		for(int row=0; row<_Rows; row++)
// 		{
// 			for(int col=0; col<_Cols; col++)
// 			{
// 				std::cout << row << ", " << col << "\n";
// 				// std::cout << std::get<

// 			}
// 		}
// 	}
// };

// int main()
// {

// 	// NullMatrix N;
// 	// Eigen::MatrixXd, Eigen::Matrix<double, 3, 4>,
// 	// Eigen::SparseMatrix<double>, Eigen::Matrix<double, 2, 2>, Eigen::Matrix<float, 2, 4>>

// 	/*
//     [0:2,2] [A:2,3] [B:2,2]
//     [B:2,2] [0:2,3] [D:2,2]
//    */

// 	using A_t = Eigen::MatrixXd;
// 	A_t A(2, 3);
// 	A << 1, 2, 3, 4, 5, 6;

// 	using B_t = Eigen::Matrix<double, 2, 2>;
// 	B_t B {{7, 8}, {9, 10}};
	
// 	using D_t = Eigen::SparseMatrix<double>;
// 	D_t D(2,2);
// 	D.insert(0,0) = 11;
// 	D.insert(1,0) = 12;

// 	BSMatrix<2, 3, A_t, B_t, B_t, D_t>
// 		BS({{{0,1}, {0,2}, {1,0}, {1,2}}},
// 			A, B, B, D);

// 	std::cout << "A = \n" << A << std::endl;
// 	std::cout << "B = \n" << B << std::endl;
// 	std::cout << "D = \n" << D << std::endl;
// }


// class MyMatrix : public Eigen::MatrixXd
// {
// public:
//     MyMatrix(void):Eigen::MatrixXd() {}
 
//     // This constructor allows you to construct MyMatrix from Eigen expressions
//     template<typename OtherDerived>
//     MyMatrix(const Eigen::MatrixBase<OtherDerived>& other)
//         : Eigen::MatrixXd(other)
//     { }
 
//     // This method allows you to assign Eigen expressions to MyMatrix
//     template<typename OtherDerived>
//     MyMatrix& operator=(const Eigen::MatrixBase <OtherDerived>& other)
//     {
//         this->Eigen::MatrixXd::operator=(other);
//         return *this;
//     }
// };


// class base {
// public:
// 	template<int Rows>
//   Eigen::Ref<Eigen::Matrix<double, Rows, 1>> getColumn(int i) {}; 
// };
 
// template<int Rows>
// class derived : public base {
// public:
// 	Eigen::Matrix<double, Rows, 2> dat;

// 	derived() 
// 	{
// 		dat.col(0).array() = 1.0;
// 		dat.col(1).array() = 2.0;
// 	}

// 	Eigen::Ref<Eigen::Matrix<double, Rows, 1>> getColumn(int i)
// 	{
// 		return dat.col(i);
// 	}
// };

// class BlockBase
// {
// public:
// 	virtual void print()
// 	{
// 		std::cout << "HELLO\n";
// 	}

// 	virtual auto getBlock() {
// 		std::cout << "BlockBase.getBlock\n";
// 	};
// 	// {
// 	// 	return -1;
// 	// }
// };

// template<typename scalar_t>
// class Block_MatrixX : public BlockBase
// {
// 	Eigen::MatrixX<scalar_t> mat;

// public:
// 	virtual void print()
// 	{
// 		std::cout << "HELLO from Block_MatrixX\n";
// 	}

// 	virtual auto getBlock()
// 	{
// 		std::cout << "Block_MatrixX.getBlock\n";
// 		return &mat;
// 	}
// };

// template<typename scalar_t, int rows, int cols>
// class Block_Matrix : public BlockBase
// {
// 	Eigen::Matrix<scalar_t, rows, cols> mat;

// public:
// 	virtual void print()
// 	{
// 		std::cout << "HELLO from Block_Matrix\n";
// 	}

// 	virtual auto getBlock()
// 	{
// 		std::cout << "Block_Matrix.getBlock\n";
// 		return &mat;
// 	}
// };



int main()
{

	Block_MatrixX<double> M;
	Block_Matrix<double, 3, 4> D;

	std::array<BlockBase*, 2> arr{&M, &D};

	std::cout << "type(M) = " << type_name<decltype(M)>() << std::endl;
	std::cout << "type(D) = " << type_name<decltype(D)>() << std::endl;

	for(int i=0; i<2; i++)
	{
		arr[i]->print();
		// auto x = arr[i]->getBlock();
		// std::cout << "type(arr[" << i << "]) = " << type_name<decltype(arr[i]->getBlock())>() << std::endl;
	}


	// #define ROWS 20
	// {
	// 	derived<ROWS> d;

	// 	std::cout << "d(1) = " << d.getColumn(1).transpose() << std::endl;

	//   time_point start = get_time();
	//   for(int i=0; i<1000000; i++)
	//   	d.getColumn(1).array() = i;
	//   time_point stop = get_time();
	//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	// 	std::cout << "d(1) = " << d.getColumn(1).transpose() << std::endl;

	//   std::cout << "Overload time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
	// }

	// {
	// 	Eigen::Matrix<double, ROWS, 2> d;
	// 	d.col(0).array() = 1.0;
	// 	d.col(1).array() = 2.0;

	// 	std::cout << "d(1) = " << d.col(1).transpose() << std::endl;

	//   time_point start = get_time();
	//   for(int i=0; i<1000000; i++)
	//   	d.col(1).array() = i;
	//   time_point stop = get_time();
	//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	// 	std::cout << "d(1) = " << d.col(1).transpose() << std::endl;

	//   std::cout << "Direct time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
	// }


	// MyMatrix A(Eigen::MatrixXd({{1,2,3},{4,5,6}}));
	// Eigen::Matrix<double, 1, 3> D{7, 8, 9};

	// Eigen::Ref<Eigen::MatrixXd> B(A);

	// std::cout << "type(A) = " << type_name<decltype(A)>() << std::endl;
	// std::cout << "type(B) = " << type_name<decltype(B)>() << std::endl;
	// std::cout << "type(D) = " << type_name<decltype(D)>() << std::endl;
	// std::cout << "\n";

	// std::cout << "A = \n" << A << std::endl;
	// std::cout << "B = \n" << B << std::endl;
	// std::cout << "D = \n" << D << std::endl;
	// std::cout << "\n";

	// A(0,0) = -1;
	// A(1,2) = 50;
	// std::cout << "A = \n" << A << std::endl;
	// std::cout << "B = \n" << B << std::endl;
	// std::cout << "D = \n" << D << std::endl;
	// std::cout << "\n";

	// Eigen::Ref<Eigen::MatrixXd> DD(A.block(1,1,1,2));

	// std::array<Eigen::Ref<Eigen::DenseCoeffsBase<MyMatrix>>, 4> arr;

	// std::array<Eigen::Ref<Eigen::DenseBase<Eigen::MatrixXd>>, 4> arr{A, D, A, DD};

	// std::cout << "arr[0] = \n" << arr[0] << std::endl;
	// std::cout << "arr[1] = \n" << arr[1] << std::endl;
	// std::cout << "arr[2] = \n" << arr[2] << std::endl;
	// std::cout << "arr[2] = \n" << arr[3] << std::endl;
	// std::cout << "\n";

	// D(0,1) = 10;
	// std::cout << "arr[0] = \n" << arr[0] << std::endl;
	// std::cout << "arr[1] = \n" << arr[1] << std::endl;
	// std::cout << "arr[2] = \n" << arr[2] << std::endl;
	// std::cout << "arr[3] = \n" << arr[3] << std::endl;
	// std::cout << "\n";

	// double x = 0;
	// for(int i=0; i<4; i++)
	// {
	// 	std::cout << "type(arr[" << i << "]) = " << type_name<decltype(arr[i])>() << std::endl;
	// 	x += arr[i](0,0);
	// }
	// std::cout << "x = " << x << std::endl;

}
