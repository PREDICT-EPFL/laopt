from collections.abc import Iterable
import polypy as pp

# TODO: Simplifying arithmetic for index. Convert to linear expression only, and do simplification of coefficients. i + i = 2*i

class Range(Iterable):
    # def __init__(self, name=None, rng=None, **kwargs):
    def __init__(self, start=None, stop=None, step=None):
        if stop is None and step is None:
            self.rng = range(start)
        elif step is None:
            self.rng = range(start, stop)
        else:
            self.rng = range(start, stop, step)

        self.left = None
        self.right = None
        self.op = None

        self.index_name = 'i'  # = kwargs.get('name')

        from polypy.generator import validate_name
        if self.index_name:
            validate_name(self.index_name)

    def __iter__(self):
        # We "iterate" here over ourselves
        return iter([self, ])

    @property
    def num_iterations(self):
        # Compute the number of iterations that this index represents
        # i.e., max(i) - min(i)
        # TODO: Incorporate multiple indices
        return len(list(self.rng))

    @property
    def indices(self):
        # Return list of all indices used in this index expression
        # if self.index_name is not None:
        if type(self) == Range:
            return {self}
        else:
            indices = set()
            if type(self.left) == Range:
                indices = indices.union(self.left.indices)
            if type(self.right) == Range:
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
        return RangeExpression(left=self, right=other, op=op)

    def __add__(self, other):
        return self.makeop(other, '+')

    def __sub__(self, other):
        return self.makeop(other, '-')

    def __mul__(self, other):
        assert type(other) != Range, "Cannot multiply two indices"
        return self.makeop(other, '*')

    def __floordiv__(self, other):
        assert type(other) != Range, "Cannot divide two indices"
        return self.makeop(other, '/')

    def __truediv__(self, other):
        assert type(other) != Range, "Cannot divide two indices"
        return self.makeop(other, '/')

    def __radd__(self, other):
        return self.makeop(other, '+')

    def __rsub__(self, other):
        return self.makeop(other, '-')

    def __rmul__(self, other):
        return self.makeop(other, '*')
        assert type(other) != Range, "Cannot multiply two indices"

    @property
    def cpp_name(self):
        """Return a unique name compatible with C++ for this index"""
        if not self.left:
            return self.index_name
        else:
            op = {"+": "plus", "-": "minus", "/": "divide", "*": "times"}[self.op]

            if isinstance(self.left, Range):
                left = self.left.cpp_name
            else:
                left = str(self.left)
            if isinstance(self.right, Range):
                right = self.right.cpp_name
            else:
                right = str(self.right)

            return f"_{left}_{op}_{right}_"


class RangeExpression(Range):
    def __init__(self, **kwargs):
        self.left = kwargs.get('left')
        self.op = kwargs.get('op')
        self.right = kwargs.get('right')



if __name__ == '__main__':
    N = 4
    i = Range(rng=range(1, N - 1), name='i')
    j = Range(rng=range(1, N - 1), name='j')

    print(i*9+i/4.4*6+j)

    print((i*9+i/4.4*6+j).indices)
