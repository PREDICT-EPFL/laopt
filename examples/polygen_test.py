from polygen import *

nlp = NLP()

N = nlp.const("N", 5) # Prediction horizon
n = nlp.const("n", 2) # State dimension
m = nlp.const("m", 1) # Input dimension

X   = nlp.var("X",   n, N)
U   = nlp.var("U",   m, N-1)
xss = nlp.var("xss", n)
uss = nlp.var("uss", m)

sys = Function('sys', n, ('xp', n), ('x', n), ('u', m))

i = Index(range(1, N-1))
nlp.equality(sys(X[i+1], X[i], U[i+1]))
nlp.equality(equal(X[0], xparam))
nlp.equality(sys(xss, xss, uss))
nlp.equality(equal(X[N-1], xss))

# i = Index(range(1, N))
# lb <= out_bnd(X[i], U[i]) <= "ub_func(i)"


nlp.generate()
