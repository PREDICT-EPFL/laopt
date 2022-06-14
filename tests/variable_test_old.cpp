/**
 * Unit test for the construction and computation of LAMPC variables.
 */

#include <iostream>
#include "lampc.hpp"

// #include "variable.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}


// template<typename MatrixType, int MapOptions=Eigen::Unaligned, typename StrideType = Eigen::Stride<0,0> > class IndexedMap;


// namespace Eigen {

// namespace internal { 
// template<typename PlainObjectType, int MapOptions, typename StrideType>
// struct traits<IndexedMap<PlainObjectType, MapOptions, StrideType> >
//   : public traits<PlainObjectType>
// {
//   typedef traits<PlainObjectType> TraitsBase;
//   enum {
//     PlainObjectTypeInnerSize = ((traits<PlainObjectType>::Flags&RowMajorBit)==RowMajorBit)
//                              ? PlainObjectType::ColsAtCompileTime
//                              : PlainObjectType::RowsAtCompileTime,

//     InnerStrideAtCompileTime = StrideType::InnerStrideAtCompileTime == 0
//                              ? int(PlainObjectType::InnerStrideAtCompileTime)
//                              : int(StrideType::InnerStrideAtCompileTime),
//     OuterStrideAtCompileTime = StrideType::OuterStrideAtCompileTime == 0
//                              ? (InnerStrideAtCompileTime==Dynamic || PlainObjectTypeInnerSize==Dynamic
//                                 ? Dynamic
//                                 : int(InnerStrideAtCompileTime) * int(PlainObjectTypeInnerSize))
//                              : int(StrideType::OuterStrideAtCompileTime),
//     Alignment = int(MapOptions)&int(AlignedMask),
//     Flags0 = TraitsBase::Flags & (~NestByRefBit),
//     Flags = is_lvalue<PlainObjectType>::value ? int(Flags0) : (int(Flags0) & ~LvalueBit)
//   };
// private:
//   enum { Options }; // Expressions don't have Options
// };
// }


// template<typename PlainObjectType, int MapOptions, typename StrideType> class IndexedMap
//   : public Eigen::MapBase<IndexedMap<PlainObjectType, MapOptions, StrideType> >
// {
// 	using Base = Eigen::Map<PlainObjectType>;
// 	using Base::Base;

// 	Eigen::Vector<Eigen::Index, PlainObjectType::RowsAtCompileTime> m_row_indices;
// 	Eigen::Vector<Eigen::Index, PlainObjectType::ColsAtCompileTime> m_col_indices;

// // public:
// //   /** Constructor in the fixed-size case (we only allow this one).
// //     *
// //     * \param dataPtr pointer to the array to map
// //     * \param stride optional Stride object, passing the strides.
// //     */
// //   EIGEN_DEVICE_FUNC
// //   explicit inline IndexedMap(typename Base::PointerArgType dataPtr, const StrideType& stride = StrideType())
// //     : Base(dataPtr, stride)
// //   {
// //   	for(int i=0; i<PlainObjectType::RowsAtCompileTime; i++) m_row_indices[i] = i;
// //   	for(int i=0; i<PlainObjectType::ColsAtCompileTime; i++) m_col_indices[i] = i;
// //   }

// // 	inline auto row_indices() {	return m_row_indices;	}
// // 	inline auto col_indices() {	return m_col_indices;	}

// // 	template<typename RowIndex>
// // 	inline auto operator()(RowIndex rowindex)
// // 	{
// // 		std::cout << "vector bob" << std::endl;
// // 		return operator()(rowindex, 0);
// // 	}

// // 	template<typename RowIndex, typename ColIndex>
// // 	inline auto operator()(RowIndex rowindex, ColIndex colindex)
// // 	{
// // 		std::cout << "bob" << std::endl;
// // 		return Base::operator()(rowindex,colindex);
// // 	}

// };


template<typename _Scalar, int _Rows>
class IndexedVector : public Eigen::Vector<_Scalar, _Rows>
{
	using scalar_t = _Scalar;
	static constexpr int Rows = _Rows;
	using Base = Eigen::Vector<scalar_t, Rows>;

public:
  IndexedVector(void) : Base() {  	set_offset(4);
}

  IndexedVector(const Eigen::MatrixBase<IndexedVector<scalar_t, Rows>>& other)
      : Base(other)
  { 
  	set_offset(4);
  	std::cout << "In IndexedVector constructor..." << std::endl;
  }

  // This constructor allows you to construct IndexedVector from Eigen expressions
  template<typename OtherDerived>
  IndexedVector(const Eigen::MatrixBase<OtherDerived>& other)
      : Base(other)
  { 
  	set_offset(4);
	}

  // This method allows you to assign Eigen expressions to IndexedVector
  template<typename OtherDerived>
  IndexedVector& operator=(const Eigen::MatrixBase <OtherDerived>& other)
  {
      this->Base::operator=(other);
      return *this;
  }

	Eigen::Vector<int, Rows> m_indices;

	void set_offset(int offset)
	{
		for(int i=0; i<Rows; i++) m_indices[i] = offset + i;
	}

	// template<typename RowIndex>
	// inline auto operator()(RowIndex rowindex)
	// {
	// 	std::cout << "vector bob" << std::endl;
	// 	return operator()(rowindex, 0);
	// }

	// template<typename RowIndex, typename ColIndex>
	// inline auto operator()(RowIndex rowindex, ColIndex colindex)
	// {
	// 	std::cout << "bob" << std::endl;
	// 	return Base::operator()(rowindex,colindex);
	// }
};

using namespace Eigen;

// [circulant_func]
template<class ArgType>
class circulant_functor {
  const ArgType &m_vec;
public:
  circulant_functor(const ArgType& arg) : m_vec(arg) {}

  const typename ArgType::Scalar& operator() (Index row, Index col) const {
    Index index = row - col;
    if (index < 0) index += m_vec.size();
    return m_vec(index);
  }
};
// [circulant_func]

// [square]
template<class ArgType>
struct circulant_helper {
  typedef Matrix<typename ArgType::Scalar,
                 ArgType::SizeAtCompileTime,
                 ArgType::SizeAtCompileTime,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType::MaxSizeAtCompileTime> MatrixType;
};
// [square]

// [makeCirculant]
template <class ArgType>
CwiseNullaryOp<circulant_functor<ArgType>, typename circulant_helper<ArgType>::MatrixType>
makeCirculant(const Eigen::MatrixBase<ArgType>& arg)
{
  typedef typename circulant_helper<ArgType>::MatrixType MatrixType;
  return MatrixType::NullaryExpr(arg.size(), arg.size(), circulant_functor<ArgType>(arg.derived()));
}
// [makeCirculant]







// // [circulant_func]
// template<class ArgType>
// class index_functor {
//   const ArgType &m_vec;
// public:
//   index_functor(const ArgType& arg) : m_vec(arg) {}

//   const typename ArgType::Scalar& operator() (Index row, Index col) const {
//     Index index = row - col;
//     if (index < 0) index += m_vec.size();
//     return m_vec(index);
//   }
// };
// // [index_func]

// // [square]
// template<class ArgType>
// struct index_helper {
//   typedef Matrix<typename ArgType::Scalar,
//                  ArgType::SizeAtCompileTime,
//                  ArgType::SizeAtCompileTime,
//                  ColMajor,
//                  ArgType::MaxSizeAtCompileTime,
//                  ArgType::MaxSizeAtCompileTime> MatrixType;
// };
// // [square]

// // [makeindex]
// template <class ArgType>
// CwiseNullaryOp<index_functor<ArgType>, typename index_helper<ArgType>::MatrixType>
// makeindex(const Eigen::MatrixBase<ArgType>& arg)
// {
//   typedef typename index_helper<ArgType>::MatrixType MatrixType;
//   return MatrixType::NullaryExpr(arg.size(), arg.size(), index_functor<ArgType>(arg.derived()));
// }
// // [makeCirculant]









// TODO : Add enable_if for xprtype == indexedvector
template<typename XprType, typename RowIndices, typename ColIndices, typename Rows>
inline void getIndices(const Eigen::IndexedView<XprType, RowIndices, ColIndices>& x,
							  			 Eigen::MatrixBase<Rows>&& rows)
							  			 // Eigen::MatrixBase<Cols>&& cols)
{
	using Base = typename Eigen::IndexedView<XprType, RowIndices, ColIndices>;

	// const auto& mat = x.nestedExpression();
	std::cout << "x.nestedExpression() = " << type_name<decltype(x.nestedExpression())>() << std::endl;
	std::cout << "XprType = " << type_name<XprType>() << std::endl;

	assert((int)Rows::RowsAtCompileTime == (int)Base::RowsAtCompileTime && "Incorrect length for rows.");
	const RowIndices& row_indices = x.rowIndices();
	// for(int i=0; i<Rows::RowsAtCompileTime; i++) rows[i] = mat.m_indices[row_indices[i]];

	// assert((int)Cols::RowsAtCompileTime == (int)Base::ColsAtCompileTime && "Incorrect length for cols.");
	// const ColIndices& col_indices = x.colIndices();
	// for(int i=0; i<Cols::RowsAtCompileTime; i++) cols[i] = m_indicies[col_indices[i]];
}

namespace {

TEST(Problem, Variable) {
	using scalar_t = double;

	Eigen::Vector<int, 10> rows;
	Eigen::Vector<int, 10> cols;
	rows.array() = 0; cols.array() = 0;

	IndexedVector<scalar_t, 7> x;
	// x.set_offset(12);
	// x << 1,2,3,4,5,6,7;

	std::cout << "type(x) = " << type_name<decltype(x)>() << std::endl;
	std::cout << "type(x) = " << type_name<decltype(x.segment(2,0))>() << std::endl;

	auto C = makeCirculant(x);
	// std::cout << "C = \n " << C << std::endl;
	std::cout << "type(C) = " << type_name<decltype(C)>() << std::endl;
	std::cout << "type(C.topLeftCorner(2,3)) = " << type_name<decltype(C.topLeftCorner(2,3))>() << std::endl;

	// std::cout << "rows = " << rows.transpose() << std::endl;
	// getIndices(x({3,2,1}), rows(seqN(3,fix<3>)));
	// std::cout << "rows = " << rows.transpose() << std::endl;
}

}