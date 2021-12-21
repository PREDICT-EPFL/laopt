template<typename Scalar>
struct MyProblem
{

	// Something that provides jacobian computation
	template<typename xp, typename x, typename u>
	struct myfunc : Function<myfunc, 3, xp, x, u>
	{
		template<typename T>
		static void eval(param& p, var<x> x, var<y> y, output& out)
		{
			asdfasdfasdf
		}
	};


	template<int i>
	struct X : Variable<2> {};

	struct U : Variable<3, N-1> {};
	struct z : Variable<2> {};

	template<int i>
	struct dynamics_con : DiffConstraint<myfunc, X<i+1>, X<i>, U<i>, z> {};

	struct con2 : DenseRow<Scalar, 2, x> {};
	struct con3 : DenseRow<Scalar, 4, z, y> {};

	BSMatrix<std::tuple<dynamics_con<i>..., con2, con3>, std::tuple<X<i>..., U<i>..., z>> jacobian;
	BSVector<std::tuple<dynamics_con<i>..., con2, con3>, std::tuple<X<i>..., U<i>..., z>> jacobian;

	jacobian.visit_rowwise(eval);
	jacobian.toSparse(S);

	Problem<Variable<x, y, z>, Constraints<con1, con2, con3>>

}


template<typename Scalar, 
struct DiffConstraint : Row<Scalar, 




template<typename row, typename column, typename Derived>
struct BlockBase
{

}

template<typename row, typename column, typename Scalar>
struct DenseBlock : BlockBase<row, column, DenseBlock<row,column,Scalar>>
{
	Eigen::Matrix<Scalar, row::len, column::len> m_matrix;
}

template<typename row, typename column, typename block>
using make_block = 

std::tuple<
