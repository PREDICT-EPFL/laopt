function print_sparse_matrix(S)

[r,c,s] = find(S);

fprintf('\nstd::vector<T> trip = {T');
join = '';
for i = 1:length(r)
  fprintf('%s{%i,%i,%i}%s', join, r(i), c(i), s(i));
  join = ',';
end
fprintf('\n')
