from typing import List

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.type import BaseType, void
from .smem import ProcedureOp


class DebugPrint(TileOp):
    def __init__(self, x: Expr):
        super().__init__(args=[x])
        self.x: Expr = x

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return void


class DebugSyncThreads(ProcedureOp):
    def __init__(self):
        super().__init__(attrs={})


def debug_print(x: Expr):
    return DebugPrint(x).make_call()
