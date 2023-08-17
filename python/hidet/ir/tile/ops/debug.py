from typing import Union, Optional, List
from hidet.ir.expr import Expr
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp, call_tile_op


class DebugPrint(TileOp):
    def __init__(self, x: Expr):
        super().__init__(args=[x])
        self.x: Expr = x

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return void


def debug_print(x: Expr):
    return DebugPrint(x).make_call()
