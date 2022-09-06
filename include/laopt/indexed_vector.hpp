#ifndef LAOPT_INDEXED_VECTOR_HPP
#define LAOPT_INDEXED_VECTOR_HPP

#include "Eigen/Dense"
#include "expressions/base_expr.hpp"

namespace laopt {

template<typename Derived>
class VariableBase : public BaseExpr<Derived> {};

/**
 * A class derived from any Eigen Vector type that records an internal index 
 * vector. As slicing operations are done to the vector, the same slices are
 * executed on the index, so the index vector will always be associated to the 
 * indices of the original data elements
 */
template<typename Base>
class IndexedVector : public Base, public VariableBase<IndexedVector<Base>> {
    using index_t = typename Eigen::Vector<int, Base::RowsAtCompileTime>;
    // Indices of the elements of this vector wrt the original IndexedMap
    index_t m_indices;

    // If Base is Eigen::Map we have to initialize with a nullptr, otherwise we use the default constructor
    template<typename Derived>
    struct constructor_selector {};

    template<typename PlainObjectType, int MapOptions, typename StrideType>
    explicit IndexedVector(constructor_selector<Eigen::Map<PlainObjectType, MapOptions, StrideType>>) : Base(nullptr) {};

    template<typename Derived>
    explicit IndexedVector(constructor_selector<Derived>) : Base() {};

public:

    static constexpr int n_inputs = Base::RowsAtCompileTime;
    static constexpr int n_outputs = Base::RowsAtCompileTime;

    static_assert(Base::RowsAtCompileTime > 0, "Variables must have a compile time size. If you used Eigen::seq or Eigen::seqN, make sure you use Eigen::fix<> for compile time size information.");
    static_assert(Base::ColsAtCompileTime == 1, "You tired using a matrix on an indexed vector");

    IndexedVector() : IndexedVector(constructor_selector<Base>{}) {
        m_indices.array() = -1;
    }

    // This constructor forwards the constructor to the underlying vector type
    template<typename ...Args>
    explicit IndexedVector(Args&& ...args) : Base(std::forward<Args>(args)...) {
        m_indices.array() = -1;
    }

    /**
     * Sets the memory of the indexed vector to a section of the global decision variable with an offset.
     */
    void set_memory(int offset, typename Base::Scalar* master_variable)
    {
        this->set_offset(offset);
        new (this) Base(master_variable + offset);
    }

    // This method allows you to assign Eigen expressions to IndexedVector
    template<typename OtherDerived>
    IndexedVector& operator=(const Eigen::MatrixBase<OtherDerived> &other) {
        this->Base::operator=(other);
        m_indices.array() = -1;
        return *this;
    }

    const index_t& indices() const
    {
        return m_indices;
    }

    index_t& indices() 
    {
        return m_indices;
    }

    int offset()
    {
        return m_indices[0]; // Returns the first index (even if they're not contiguous)
    }

    /**
     * Sets the index of the first element, and assumes the rest are contiguous.
     */
    void set_offset(int offset) {
        for (int i = 0; i < m_indices.rows(); i++)
        {
            m_indices[i] = offset + i;
        }
    }

    /**
     * Sets the indices to the given vector.
     */
    template <typename Derived>
    void set_indices(const Eigen::MatrixBase<Derived> &indices) {
        static_assert(std::is_same<typename Derived::Scalar, int>::value, "indices have to be integer");
        m_indices = indices;
    }

    Base& cast_base()
    {
        return static_cast<Base&>(*this);
    }

    const Base& cast_base() const
    {
        return static_cast<const Base&>(*this);
    }

    /**
     * Forwards vector access operator
     */
    template<typename ...Args>
    EIGEN_STRONG_INLINE auto operator()(Args&& ...args) {
        using RetType = decltype(Base::operator()(std::forward<Args>(args)...));
        IndexedVector<RetType> ret(Base::operator()(std::forward<Args>(args)...));
        ret.set_indices(m_indices(std::forward<Args>(args)...));
        return ret;
    }

    /**
     * Indices given by raw array or initializer list
     */
    template<typename RowIndicesT, std::size_t RowIndicesN>
    EIGEN_STRONG_INLINE auto operator()(const RowIndicesT (&rowIndices)[RowIndicesN]) {
        using RetType = decltype(Base::operator()(rowIndices));
        IndexedVector<RetType> ret(Base::operator()(rowIndices));
        ret.set_indices(m_indices(rowIndices));
        return ret;
    }
};

template<typename Scalar, int n>
using Variable = IndexedVector<Eigen::Map<Eigen::Vector<Scalar, n>>>;

}

#endif // LAOPT_INDEXED_VECTOR_HPP
