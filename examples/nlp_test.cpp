/*[[[cog 
from polygen import *
nlp = NLP()

N = nlp.const("N", 5) # Prediction horizon
n = nlp.const("n", 2) # State dimension
m = nlp.const("m", 1) # Input dimension

X   = nlp.var("X",   n, N)
U   = nlp.var("U",   m, N-1)
xss = nlp.var("xss", n)
uss = nlp.var("uss", m)

i = Index(range(1, N-1))
sys0(X[0], U[0]) == 0
sys(X[i+1], X[i], U[i+1]) == 0

sys(xss, xss, uss) == 0
equal(X[N-1], xss) == 0

# i = Index(range(1, N))
# lb <= out_bnd(X[i], U[i]) <= "ub_func(i)"


nlp.generate()
]]]*/


// [[[end]]]
