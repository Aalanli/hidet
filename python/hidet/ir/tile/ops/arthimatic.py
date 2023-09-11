from typing import Union, Optional, List
from hidet.ir.type import BaseType
from hidet.ir.expr import Var, Expr
from hidet.ir.tile.type import tile_type, TileType
from hidet.ir.tile.expr import TileOp


class UnaryTileOp(TileOp):
    def __init__(self, x: Expr):
        super().__init__(args=[x])
        self.x: Expr = x

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        a_type = arg_types[0]
        assert isinstance(a_type, TileType)

        return arg_types[0]

    def apply_scalar(self, x: Expr) -> Expr:
        import hidet.ir.expr

        cls_map = {
            'Neg': hidet.ir.expr.Neg,
            'LogicalNot': hidet.ir.expr.LogicalNot,
            'BitwiseNot': hidet.ir.expr.BitwiseNot,
        }

        cls_name = self.__class__.__name__

        if cls_name not in cls_map:
            raise NotImplementedError(f'No implementation for {cls_name} binary op')

        expr_cls = cls_map[cls_name]
        return Expr._unary(expr_cls, x)


class BinaryTileOp(TileOp):
    def __init__(self, x: Expr, y: Expr):
        super().__init__(args=[x, y])
        self.x: Expr = x
        self.y: Expr = y

        if isinstance(x, Var) and isinstance(y, Var) and isinstance(x.type, TileType) and isinstance(y.type, TileType):
            if x.type.layout and y.type.layout:
                assert x.type.layout == y.type.layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        from hidet.ir.dtypes import boolean

        a_type = arg_types[0]
        b_type = arg_types[1]
        assert isinstance(a_type, TileType) and isinstance(b_type, TileType)
        assert a_type.layout == b_type.layout and a_type.scope == b_type.scope

        if isinstance(self, (Add, Sub, Multiply, Div, Mod)):
            return arg_types[0]
        elif isinstance(self, (LessThan, LessEqual, Equal, NotEqual, LogicalAnd, LogicalOr)):
            return tile_type(elem_type=boolean, shape=a_type.shape, scope=a_type.scope, layout=a_type.layout)
        else:
            raise NotImplementedError()

    def apply_scalar(self, x: Expr, y: Expr) -> Expr:
        import hidet.ir.expr

        cls_name = self.__class__.__name__
        if not hasattr(hidet.ir.expr, cls_name):
            raise NotImplementedError(f'No implementation for {cls_name} binary op')
        expr_cls = getattr(hidet.ir.expr, cls_name)
        return Expr._binary(expr_cls, x, y)


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


class LogicalAnd(BinaryTileOp):
    pass


class LogicalOr(BinaryTileOp):
    pass
