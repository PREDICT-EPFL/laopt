function S = make_random_sparse_matrix(m,n,density)
S = round(sprand(m,n,density)*100);
full(S)
