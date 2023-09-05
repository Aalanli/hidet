from typing import List

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.type import TileType
from hidet.ir.type import BaseType
from .creation import zeros


class Dot(TileOp):
    def __init__(self, a: Expr, b: Expr, c: Expr):
        super().__init__(args=[a, b, c], attrs={})
        self.a: Expr = a
        self.b: Expr = b
        self.c: Expr = c

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return arg_types[2]


class SimtDot(Dot):
    pass


class MmaDot(Dot):
    pass


def dot(a: Expr, b: Expr):
    from hidet.ir.tools import infer_type

    a_type = infer_type(a)
    b_type = infer_type(b)
    assert isinstance(a_type, TileType) and isinstance(b_type, TileType), (a_type, b_type)
    assert a_type.type == b_type.type
    c = zeros([a_type.shape[0], b_type.shape[1]], a_type.type)
    return Dot(a, b, c).make_call()
