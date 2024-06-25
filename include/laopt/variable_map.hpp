#ifndef LAOPT_VARIABLE_MAP_HPP
#define LAOPT_VARIABLE_MAP_HPP

#include <Eigen/Dense>

namespace laopt {

template <typename MatrixType, int MapOptions = Eigen::Unaligned, typename StrideType = Eigen::Stride<0, 0>>
class VariableMap;

template<typename Scalar, int n>
using Variable = VariableMap<Eigen::Vector<Scalar, n>>;

} // namespace laopt

namespace Eigen {

namespace internal {

template <typename PlainObjectType, int MapOptions, typename StrideType>
struct traits<laopt::VariableMap<PlainObjectType, MapOptions, StrideType>> : public traits<Map<PlainObjectType, MapOptions, StrideType>> {};

} // namespace internal

} // namespace Eigen

namespace laopt {

template <typename PlainObjectType, int MapOptions, typename StrideType>
class VariableMap : public Eigen::MapBase<VariableMap<PlainObjectType, MapOptions, StrideType>> {
public:
    typedef Eigen::MapBase<VariableMap> Base;

    static_assert(Base::RowsAtCompileTime >= 0, "Variables must have a compile time size.");
    static_assert(Base::ColsAtCompileTime == 1, "The underlying type of a variable must be a vector.");

    EIGEN_DENSE_PUBLIC_INTERFACE(VariableMap)

    typedef typename Base::PointerType PointerType;
    typedef PointerType PointerArgType;
    EIGEN_DEVICE_FUNC inline PointerType cast_to_pointer_type(PointerArgType ptr) { return ptr; }

    EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Eigen::Index innerStride() const {
        return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
    }

    EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Eigen::Index outerStride() const {
        return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
            : Eigen::internal::traits<VariableMap>::OuterStrideAtCompileTime != Eigen::Dynamic
                ? Eigen::Index(Eigen::internal::traits<VariableMap>::OuterStrideAtCompileTime)
            : IsVectorAtCompileTime           ? (this->size() * innerStride())
            : int(Flags) & Eigen::RowMajorBit ? (this->cols() * innerStride())
                                              : (this->rows() * innerStride());
    }

    EIGEN_DEVICE_FUNC inline VariableMap() : VariableMap(nullptr) {};

    /** Constructor in the fixed-size case.
     *
     * \param dataPtr pointer to the array to map
     * \param stride optional Stride object, passing the strides.
     */
    EIGEN_DEVICE_FUNC explicit inline VariableMap(PointerArgType dataPtr, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr)), m_stride(stride), m_index_offset(0) {}

    /** Constructor in the dynamic-size vector case.
     *
     * \param dataPtr pointer to the array to map
     * \param size the size of the vector expression
     * \param stride optional Stride object, passing the strides.
     */
    EIGEN_DEVICE_FUNC inline VariableMap(PointerArgType dataPtr, Eigen::Index size, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr), size), m_stride(stride), m_index_offset(0) {}

    /** Constructor in the dynamic-size matrix case.
     *
     * \param dataPtr pointer to the array to map
     * \param rows the number of rows of the matrix expression
     * \param cols the number of columns of the matrix expression
     * \param stride optional Stride object, passing the strides.
     */
    EIGEN_DEVICE_FUNC inline VariableMap(PointerArgType dataPtr, Eigen::Index rows, Eigen::Index cols, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr), rows, cols), m_stride(stride), m_index_offset(0) {}

    using Base::operator=;
    inline VariableMap &operator=(const VariableMap& other) {
        Base::operator=(other);
        return *this;
    }
    template<typename OtherDerived>
    inline VariableMap &operator=(const Eigen::DenseBase<OtherDerived> &other) {
        Base::operator=(other.derived());
        return *this;
    }
    VariableMap(const VariableMap&) = default;

    const Eigen::Index& index_offset() const { return m_index_offset; };
    Eigen::Index& index_offset() { return m_index_offset; };

    void set_memory(int offset, Scalar* master_variable)
    {
        m_index_offset = offset;
        new (this) Base(master_variable + offset);
    }

protected:
    StrideType m_stride;
    Eigen::Index m_index_offset;
};

} // namespace laopt

namespace Eigen {

namespace internal {

template <typename PlainObjectType, int MapOptions, typename StrideType>
struct evaluator<laopt::VariableMap<PlainObjectType, MapOptions, StrideType> >
        : public mapbase_evaluator<laopt::VariableMap<PlainObjectType, MapOptions, StrideType>, PlainObjectType> {
    typedef laopt::VariableMap<PlainObjectType, MapOptions, StrideType> XprType;
    typedef typename XprType::Scalar Scalar;
    typedef typename packet_traits<Scalar>::type PacketScalar;

    enum {
        InnerStrideAtCompileTime = StrideType::InnerStrideAtCompileTime == 0
                                   ? int(PlainObjectType::InnerStrideAtCompileTime)
                                   : int(StrideType::InnerStrideAtCompileTime),
        OuterStrideAtCompileTime = StrideType::OuterStrideAtCompileTime == 0
                                   ? int(PlainObjectType::OuterStrideAtCompileTime)
                                   : int(StrideType::OuterStrideAtCompileTime),
        HasNoInnerStride = InnerStrideAtCompileTime == 1,
        HasNoOuterStride = StrideType::OuterStrideAtCompileTime == 0,
        HasNoStride = HasNoInnerStride && HasNoOuterStride,
        IsDynamicSize = PlainObjectType::SizeAtCompileTime == Dynamic,

        PacketAccessMask = bool(HasNoInnerStride) ? ~int(0) : ~int(PacketAccessBit),
        LinearAccessMask =
        bool(HasNoStride) || bool(PlainObjectType::IsVectorAtCompileTime) ? ~int(0) : ~int(LinearAccessBit),
        Flags = int(evaluator<PlainObjectType>::Flags) & (LinearAccessMask & PacketAccessMask),

        Alignment = int(MapOptions) & int(AlignedMask)
    };

    EIGEN_DEVICE_FUNC explicit evaluator(const XprType& map) : mapbase_evaluator<XprType, PlainObjectType>(map) {}
};

} // namespace internal

} // namespace Eigen

namespace laopt {

template<typename T>
struct is_variable_base : std::false_type {};

template<typename PlainObjectType, int MapOptions, typename StrideType>
struct is_variable_base<VariableMap<PlainObjectType, MapOptions, StrideType>> : std::true_type {};

template<typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
struct is_variable_base<Eigen::Block<Derived, BlockRows, BlockCols, InnerPanel>> : is_variable_base<Derived> {};

template<typename VectorType, int Size>
struct is_variable_base<Eigen::VectorBlock<VectorType, Size>> : is_variable_base<VectorType> {};

template<typename Derived, typename RowIndices, typename ColIndices>
struct is_variable_base<Eigen::IndexedView<Derived, RowIndices, ColIndices>> : is_variable_base<Derived> {};

template<typename Derived>
struct is_variable_base<Eigen::MapBase<Derived>> : is_variable_base<Derived> {};

template<typename Derived>
struct is_variable_base<Eigen::MatrixBase<Derived>> : is_variable_base<Derived> {};

template<typename Derived>
struct is_variable_base<Eigen::DenseBase<Derived>> : is_variable_base<Derived> {};

template<typename Derived>
struct is_variable_base<Eigen::EigenBase<Derived>> : is_variable_base<Derived> {};

template<typename T>
struct is_variable : is_variable_base<typename Eigen::internal::remove_all<T>::type> {};


template<typename T, bool>
struct variable_info_base;

template<typename T>
struct variable_info_base<T, false>
{
    // If it's not a variable we ignore it
    static constexpr int size = 0;
};

template<typename T>
struct variable_info_base<T, true>
{
    static constexpr int size = Eigen::internal::remove_all<T>::type::RowsAtCompileTime;
};

template<typename T>
struct variable_info : public variable_info_base<T, is_variable<T>::value> {};

template<typename PlainObjectType, int MapOptions, typename StrideType>
EIGEN_STRONG_INLINE Eigen::Vector<int, PlainObjectType::RowsAtCompileTime>
variable_indices(const VariableMap<PlainObjectType, MapOptions, StrideType>& map)
{
    Eigen::Vector<int, PlainObjectType::RowsAtCompileTime> indices;
    for (int i = 0; i < indices.rows(); i++) {
        indices[i] = i + map.index_offset();
    }
    return indices;
}

template<typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
EIGEN_STRONG_INLINE Eigen::Vector<int, BlockRows>
variable_indices(const Eigen::Block<Derived, BlockRows, BlockCols, InnerPanel>& block)
{
    static_assert(BlockRows >= 0, "Number of rows must be known at compile time. If using Eigen::seqN, make sure to use Eigen::fix for the size.");
    static_assert(BlockCols == 1, "Numbers of columns must be one, i.e., it must be a vector.");

    auto derived_indices = variable_indices(block.nestedExpression());

    Eigen::Vector<int, BlockRows> indices(block.rows());
    for (int i = 0; i < block.rows(); i++) {
        indices(i) = derived_indices[block.startRow() + i];
    }
    return indices;
}

template<typename Derived, typename RowIndices, typename ColIndices>
EIGEN_STRONG_INLINE Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime>
variable_indices(const Eigen::IndexedView<Derived, RowIndices, ColIndices>& view)
{
    static_assert(Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime >= 0, "Number of rows must be known at compile time.");
    static_assert(Eigen::IndexedView<Derived, RowIndices, ColIndices>::ColsAtCompileTime == 1, "Numbers of columns must be one, i.e., it must be a vector.");

    auto derived_indices = variable_indices(view.nestedExpression());

    Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime> indices(view.rows());
    for (int i = 0; i < view.rows(); i++)
    {
        indices(i) = derived_indices[view.rowIndices()[i]];
    }
    return indices;
}

template<typename Derived>
EIGEN_STRONG_INLINE Eigen::Vector<int, Derived::RowsAtCompileTime>
variable_indices(const Eigen::MapBase<Derived>& mat)
{
    return variable_indices(mat.derived());
}

template<typename Derived>
EIGEN_STRONG_INLINE Eigen::Vector<int, Derived::RowsAtCompileTime>
variable_indices(const Eigen::MatrixBase<Derived>& mat)
{
    return variable_indices(mat.derived());
}

template<typename Derived>
EIGEN_STRONG_INLINE Eigen::Vector<int, Derived::RowsAtCompileTime>
variable_indices(const Eigen::DenseBase<Derived>& mat)
{
    return variable_indices(mat.derived());
}

template<typename Derived>
EIGEN_STRONG_INLINE Eigen::Vector<int, Derived::RowsAtCompileTime>
variable_indices(const Eigen::EigenBase<Derived>& mat)
{
    return variable_indices(mat.derived());
}

} // namespace laopt

#endif //LAOPT_VARIABLE_MAP_HPP
