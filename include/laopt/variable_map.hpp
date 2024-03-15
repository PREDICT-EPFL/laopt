#ifndef LAOPT_VARIABLE_MAP_HPP
#define LAOPT_VARIABLE_MAP_HPP

namespace laopt {

const unsigned int VariableBit = 0x800;

template <typename MatrixType, int MapOptions = Eigen::Unaligned, typename StrideType = Eigen::Stride<0, 0>>
class VariableMap;

} // namespace laopt

namespace Eigen {

namespace internal {

template <typename PlainObjectType, int MapOptions, typename StrideType>
struct traits<laopt::VariableMap<PlainObjectType, MapOptions, StrideType>> : public traits<Map<PlainObjectType, MapOptions, StrideType>> {
    enum {
        Flags = traits<Map<PlainObjectType, MapOptions, StrideType>>::Flags | laopt::VariableBit
    };
};

} // namespace internal

} // namespace Eigen

namespace laopt {

template <typename PlainObjectType, int MapOptions, typename StrideType>
class VariableMap : public Eigen::MapBase<VariableMap<PlainObjectType, MapOptions, StrideType>> {
public:
    typedef Eigen::MapBase<VariableMap> Base;
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

    /** Constructor in the fixed-size case.
     *
     * \param dataPtr pointer to the array to map
     * \param stride optional Stride object, passing the strides.
     */
    EIGEN_DEVICE_FUNC explicit inline VariableMap(PointerArgType dataPtr, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr)), m_stride(stride) {}

    /** Constructor in the dynamic-size vector case.
     *
     * \param dataPtr pointer to the array to map
     * \param size the size of the vector expression
     * \param stride optional Stride object, passing the strides.
     */
    EIGEN_DEVICE_FUNC inline VariableMap(PointerArgType dataPtr, Eigen::Index size, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr), size), m_stride(stride) {}

    /** Constructor in the dynamic-size matrix case.
     *
     * \param dataPtr pointer to the array to map
     * \param rows the number of rows of the matrix expression
     * \param cols the number of columns of the matrix expression
     * \param stride optional Stride object, passing the strides.
     */
    EIGEN_DEVICE_FUNC inline VariableMap(PointerArgType dataPtr, Eigen::Index rows, Eigen::Index cols, const StrideType& stride = StrideType())
        : Base(cast_to_pointer_type(dataPtr), rows, cols), m_stride(stride) {}

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

protected:
    StrideType m_stride;
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

#endif //LAOPT_VARIABLE_MAP_HPP
