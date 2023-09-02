from typing import List, Optional, Tuple
from hidet.ir.expr import Expr, Var, Add, Sub, Multiply, Constant, convert
from hidet.ir.dtypes import int32
from hidet.ir.functors import IRVisitor


class Affine:
    def __init__(self, weights: List[Expr]):
        self.weights: List[Expr] = weights

    def has_only_constant_entry(self) -> bool:
        for i in range(len(self.weights) - 1):
            w = self.weights[i]
            if not (isinstance(w, Constant) and w.value == 0):
                return False
        return True

    def __add__(self, other):
        return Affine([a + b for a, b in zip(self.weights, other.weights)])

    def __sub__(self, other):
        return Affine([a - b for a, b in zip(self.weights, other.weights)])

    def __mul__(self, other):
        a = self
        b = other
        if not a.has_only_constant_entry() and b.has_only_constant_entry():
            a, b = b, a
        if a.has_only_constant_entry():
            const_weight = a.weights[-1]
            return Affine([const_weight * bw for bw in b.weights])
        else:
            return None


class AffineDecomposer(IRVisitor):
    def __init__(self, axes: List[Expr]):
        super().__init__()
        self.axes: List[Expr] = axes

    def visit_Var(self, e: Var):
        if e in self.axes:
            pos = self.axes.index(e)
            val = int32.one
        else:
            pos = len(self.axes)
            val = e
        return Affine([val if i == pos else int32.zero for i in range(len(self.axes) + 1)])

    def visit_Constant(self, e: Constant):
        return Affine([e if i == len(self.axes) else int32.zero for i in range(len(self.axes) + 1)])

    def visit_Add(self, e: Add):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None or b is None:
            return None
        else:
            return a + b

    def visit_Sub(self, e: Sub):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None or b is None:
            return None
        else:
            return a - b

    def visit_Multiply(self, e: Multiply):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None or b is None:
            return None
        else:
            return a * b


def affine_decompose(expr: Expr, axes: List[Var]) -> Optional[List[Expr]]:
    decomposer = AffineDecomposer(axes)
    affine: Optional[Affine] = decomposer.visit(expr)
    return affine.weights if affine is not None else None
