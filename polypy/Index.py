# TODO: Simplifying arithmetic for index. Convert to linear expression only, and do simplification of coefficients. i + i = 2*i

class Index:
    def __init__(self, **kwargs):
        self.rng = kwargs.get('rng')
        self.index_name = kwargs.get('name')

        self.left = kwargs.get('left')
        self.op = kwargs.get('op')
        self.right = kwargs.get('right')

        from polypy.poly import validate_name
        if self.index_name:
            validate_name(self.index_name)

    @property
    def num_iterations(self):
        # Compute the number of iterations that this index represents
        # i.e., max(i) - min(i)
        # TODO: Incorporate multiple indices
        return len(list(self.rng))

    @property
    def indices(self):
        # Return list of all indices used in this index expression
        if self.index_name is not None:
            return {self}
        else:
            indices = set()
            if type(self.left) == Index:
                indices = indices.union(self.left.indices)
            if type(self.right) == Index:
                indices = indices.union(self.right.indices)
            return indices

    def __str__(self):
        if not self.left:
            return self.index_name
        else:
            return f"({str(self.left)}{self.op}{str(self.right)})"

    def __repr__(self):
        return str(self)

    def makeop(self, other, op):
        return Index(left=self, right=other, op=op)

    def __add__(self, other):
        return self.makeop(other, '+')

    def __sub__(self, other):
        return self.makeop(other, '-')

    def __mul__(self, other):
        assert type(other) != Index, "Cannot multiply two indices"
        return self.makeop(other, '*')

    def __floordiv__(self, other):
        assert type(other) != Index, "Cannot divide two indices"
        return self.makeop(other, '/')

    def __truediv__(self, other):
        assert type(other) != Index, "Cannot divide two indices"
        return self.makeop(other, '/')

    def __radd__(self, other):
        return self.makeop(other, '+')

    def __rsub__(self, other):
        return self.makeop(other, '-')

    def __rmul__(self, other):
        return self.makeop(other, '*')
        assert type(other) != Index, "Cannot multiply two indices"

    @property
    def cpp_name(self):
        """Return a unique name compatible with C++ for this index"""
        if not self.left:
            return self.index_name
        else:
            op = {"+": "plus", "-": "minus", "/": "divide", "*": "times"}[self.op]

            if isinstance(self.left, Index):
                left = self.left.cpp_name
            else:
                left = str(self.left)
            if isinstance(self.right, Index):
                right = self.right.cpp_name
            else:
                right = str(self.right)

            return f"_{left}_{op}_{right}_"


if __name__ == '__main__':
    N = 4
    i = Index(rng=range(1, N - 1), name='i')
    j = Index(rng=range(1, N - 1), name='j')

    print(i*9+i/4.4*6+j)

    print((i*9+i/4.4*6+j).indices)
