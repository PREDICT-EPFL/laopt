#ifndef LAMPC_VARIABLE_H
#define LAMPC_VARIABLE_H

namespace lampc
{

/**
 * A class derived from any Eigen Vector type that records an internal index 
 * vector. As slicing operations are done to the vector, the same slices are
 * executed on the index, so the index vector will always be associated to the 
 * indices of the original data elements
 */
template<typename Base>
class IndexedVector : public Base
{
	using index_t = typename Eigen::Vector<int, Base::RowsAtCompileTime>;

public:
	// Indices of the elements of this vector wrt the original IndexedMap
	index_t m_indices;


    IndexedVector() : Base()
    {
    	m_indices.array() = -1;
    }

    IndexedVector(decltype(NULL)) : Base(NULL)
    {
    	m_indices.array() = -1;
    }

    // IndexedVector() : Base(NULL)
    // {
    // 	m_indices.array() = -1;
    // }

    // This constructor allows you to construct IndexedVector from Eigen expressions
    template<typename OtherDerived>
    IndexedVector(const Eigen::MatrixBase<OtherDerived>& other) : Base(other)
    {
			m_indices.array() = -1;
		}

    template<typename... Args>
    IndexedVector(Args... args) : Base(args...)
    {
    	m_indices.array() = -1;
    }
 
    // This method allows you to assign Eigen expressions to IndexedVector
    template<typename OtherDerived>
    IndexedVector& operator=(const Eigen::MatrixBase <OtherDerived>& other)
    {
      this->Base::operator=(other);
      m_indices.array() = -1;
      return *this;
    }

  const index_t& indices() { return m_indices; }
  int offset() { return m_indices[0]; } // Returns the first index (even if they're not contiguous)

	/**
	 * Sets the index of the first element, and assumes the rest are contiguous.
	 */  
  void set_offset(int offset)
  {
  	for(int i=0; i<m_indices.rows(); i++) m_indices[i] = offset + i;
  }

	/**
	 * Sets the indices to the given vector.
	 */
  void set_indices(Eigen::MatrixBase<Eigen::Vector<int, Base::RowsAtCompileTime>>& indices)
  {
  	m_indices = indices;
  }

  /**
   * Applies the requested slice operation to both the vector
   * itself, and to the associated index vector
   */
	template<typename RowSlice>
	inline auto operator()(const RowSlice& slice)
	{
		using RetType = decltype(Base::operator()(slice,0));
		IndexedVector<RetType> ret(Base::operator()(slice,0));
		ret.m_indices = m_indices(slice);
		return ret;
	}

	/**
	 * Indices given by raw array or initializer list
	 */
	template<typename RowIndicesT, std::size_t RowIndicesN>
	inline auto operator()(const RowIndicesT (&rowIndices)[RowIndicesN])
	{
		using RetType = decltype(Base::operator()(rowIndices,0));
		IndexedVector<RetType> ret(Base::operator()(rowIndices,0));
		ret.m_indices = m_indices(rowIndices);
		return ret;
	}
};

}

#endif // LAMPC_VARIABLE_H
