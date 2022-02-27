function print_sparse_matrix(S, name)

[r,c,s] = find(S);

fprintf('Eigen::SparseMatrix<S> %s(%i,%i);\n', name, size(S,1), size(S,2));
fprintf('{\n')
fprintf('  std::vector<T> trip = {T');
join = '';
for i = 1:length(r)
  fprintf('%s{%i,%i,%i}%s', join, r(i)-1, c(i)-1, s(i));
  join = ',';
end
fprintf('};\n')
fprintf('  %s.setFromTriplets(trip.begin(), trip.end());\n', name);
fprintf('}\n')
