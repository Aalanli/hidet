from typing import List

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.type import BaseType, void


class DebugPrint(TileOp):
    def __init__(self, x: Expr):
        super().__init__(args=[x])
        self.x: Expr = x

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return void


def debug_print(x: Expr):
    return DebugPrint(x).make_call()
