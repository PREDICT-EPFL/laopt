/**
 * Write sparse matrix in triplet format to cout
 */
template<typename scalar_t>
void dump_sparse(std::string name, Eigen::SparseMatrix<scalar_t> S)
{
    std::cout << std::endl;
    std::cout << "auto " << name << " = triplet_to_sparse<" << type_name<scalar_t>() << ">(" << S.rows() << "," << S.cols() << ",";
    std::cout << "{";
    bool first = true;
    std::cout.precision(5);
    for (int k=0; k<S.outerSize(); ++k)
      for (typename SparseMatrix<scalar_t>::InnerIterator it(S,k); it; ++it)
      {
        if(first) first = false; else std::cout << ",";
        std::cout << "{" << it.row() << "," << it.col() << "," << it.value() << "}";
      }
    std::cout << "});";
    std::cout << std::endl;
}

template<typename scalar_t>
Eigen::SparseMatrix<scalar_t> triplet_to_sparse(int rows, int cols, std::initializer_list<Eigen::Triplet<scalar_t>> trip)
{
    Eigen::SparseMatrix<scalar_t> S;
    S.resize(rows, cols);
    S.setFromTriplets(trip.begin(), trip.end());
    S.makeCompressed();    
    return S;
}
