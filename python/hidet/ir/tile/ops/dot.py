from typing import Union, Optional, List
from hidet.ir.expr import Expr
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp, call_tile_op


class Dot(TileOp):
    def __init__(self, a: Expr, b: Expr, kind: Optional[str] = None, layout: Optional[TileLayout] = None):
        super().__init__(args=[a, b], attrs={"layout": layout})
        self.a: Expr = a
        self.b: Expr = b
        self.kind: Optional[str] = kind
        self.layout: Optional[TileLayout] = layout

        if kind:
            assert kind in ['mma', 'simt']

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        a_type = arg_types[0]
        b_type = arg_types[1]
        assert isinstance(a_type, TileType) and isinstance(b_type, TileType)
        assert isinstance(a_type.type, DataType) and isinstance(b_type.type, DataType)
        assert a_type.type == b_type.type
        a_shape: List[int] = a_type.shape
        b_shape: List[int] = b_type.shape
        assert len(a_shape) == len(b_shape) == 2
        assert a_shape[1] == b_shape[0]
        m, n = a_shape[0], b_shape[1]
        return tile_type(type_=a_type.type, shape=[m, n], layout=self.layout)


def dot(a: Expr, b: Expr):
    return Dot(a, b).make_call()
