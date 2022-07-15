#ifndef LAMPC_VARIABLE_HPP
#define LAMPC_VARIABLE_HPP

#include "indexed_vector.hpp"

namespace lampc
{

/**
* A Variable is just an Eigen Map that also maintains an index vector into the global
* decision variable
*/
template<typename _scalar_t, int _size>
class Variable : public IndexedVector<Eigen::Map<Eigen::Vector<_scalar_t, _size>>>
{
    static constexpr int m_size = _size;
    using IndexedVectorBase = IndexedVector<Eigen::Map<Eigen::Vector<_scalar_t, _size>>>;

    using IndexedVectorBase::IndexedVectorBase;

public:
    using base = typename IndexedVectorBase::base;

    using scalar_t = _scalar_t;

    Variable() : IndexedVectorBase(NULL) {} // The map initially points to NULL

    constexpr int size() { return m_size; }
    bool has_data() { return this->data() != NULL; }

    base& cast_base()
    {
        return static_cast<base&>(*this);
    }

    const base& cast_base() const
    {
        return static_cast<const base&>(*this);
    }

    /**
     * offset = offset of this variable into the global decision variable.
     *
     * Returns a callback function used to set the optimization variable
     */
    auto register_variable(int offset)
    {
        this->set_offset(offset);

        Variable<_scalar_t, _size>* p_this = this;
        return [p_this](scalar_t* master_variable)
        {
            new (p_this) Eigen::Map<Eigen::Vector<scalar_t, m_size>>(master_variable + p_this->offset());
        };
    }
};

}

#endif //LAMPC_VARIABLE_HPP
