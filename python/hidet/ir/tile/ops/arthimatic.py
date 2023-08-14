from typing import Union, Optional, List
from hidet.ir.type import BaseType
from hidet.ir.expr import Var, Expr
from hidet.ir.tile.type import tile_type, void_layout, TileLayout, TileType
from hidet.ir.tile.expr import TileOp
from hidet.utils import same_list


class UnaryTileOp(TileOp):
    def __init__(self, x: Expr):
        super().__init__(args=[x])
        self.x: Expr = x

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        a_type = arg_types[0]
        assert isinstance(a_type, TileType)

        return arg_types[0]


class BinaryTileOp(TileOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(args=[x, y])
        self.x: Expr = x
        self.y: Expr = y

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        a_type = arg_types[0]
        b_type = arg_types[1]
        assert isinstance(a_type, TileType) and isinstance(b_type, TileType)
        assert same_list(a_type.shape, b_type.shape)

        return arg_types[0]


class Neg(UnaryTileOp):
    pass


class LogicalNot(UnaryTileOp):
    pass


class BitwiseNot(UnaryTileOp):
    pass


class Add(BinaryTileOp):
    pass


class Sub(BinaryTileOp):
    pass


class Multiply(BinaryTileOp):
    pass


class Div(BinaryTileOp):
    pass


class Mod(BinaryTileOp):
    pass


class LessThan(BinaryTileOp):
    pass


class LessEqual(BinaryTileOp):
    pass


class Equal(BinaryTileOp):
    pass


class NotEqual(BinaryTileOp):
    pass

