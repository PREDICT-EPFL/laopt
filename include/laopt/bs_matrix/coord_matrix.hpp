#ifndef LAOPT_COORD_MATRIX_HPP
#define LAOPT_COORD_MATRIX_HPP

#include <Eigen/Dense>

namespace laopt {

class CoordMatrix;

struct Coord
{
    Eigen::Index row;
    Eigen::Index col;
};

} // namespace laopt

namespace Eigen {

namespace internal {

template<>
struct traits<laopt::CoordMatrix>
{
    using StorageKind = Eigen::Dense;
    using XprKind = Eigen::MatrixXpr;
    using StorageIndex = Eigen::Index;
    using Scalar = laopt::Coord;
    enum {
        Flags = Eigen::NoPreferredStorageOrderBit | Eigen::NestByRefBit,
        RowsAtCompileTime = Eigen::Dynamic,
        ColsAtCompileTime = Eigen::Dynamic,
        MaxRowsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };
};

} // namespace internal

template<typename Rhs, typename BinOp>
struct ScalarBinaryOpTraits<laopt::Coord, Rhs, BinOp>
{
    typedef laopt::Coord ReturnType;
};

} // namespace Eigen

namespace laopt {

class CoordMatrix : public Eigen::MatrixBase<CoordMatrix>
{
private:
    Eigen::Index m_rows, m_cols;

public:
    explicit CoordMatrix(Eigen::Index rows = 0, Eigen::Index cols = 0)
        : m_rows(rows), m_cols(cols) {}

    typedef Eigen::Index Index;
    Index rows() const { return m_rows; }
    Index cols() const { return m_cols; }

    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        m_rows = rows;
        m_cols = cols;
    }
};

} // namespace laopt

namespace Eigen {
namespace internal {

template<>
struct evaluator<laopt::CoordMatrix>
        : evaluator_base<laopt::CoordMatrix>
{
    using XprType = laopt::CoordMatrix;
    using CoeffReturnType = typename XprType::CoeffReturnType;

    enum {
        CoeffReadCost = 1,
        Flags = traits<XprType>::Flags
    };

    explicit evaluator(const XprType& xpr) {}

    CoeffReturnType coeff(Index row, Index col) const
    {
        return {row, col};
    }
};

} // namespace internal
} // namespace Eigen

#endif //LAOPT_COORD_MATRIX_HPP
