from typing import Type, Dict, Union, List

from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr
from hidet.ir.functors import IRRewriter
from hidet.ir.func import Function
from hidet.ir import expr
from hidet.ir.type import PointerType, DataType
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.ops.transform import broadcast, full
from hidet.ir.utils.broadcast_utils import broadcast_shape
from .base import TileFunctionPass
from hidet.utils import same_list


class InjectExplicitBroadcastRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Binary(self, e: BinaryExpr):
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer.visit(a)
        b_type = self.type_infer.visit(b)
        if isinstance(a_type, TileType) and isinstance(b_type, TileType):
            if not same_list(a_type.shape, b_type.shape):
                shape: List[int] = broadcast_shape(a_type.shape, b_type.shape)
                if not same_list(a_type.shape, shape):
                    a = broadcast(a, shape)
                if not same_list(b_type.shape, shape):
                    b = broadcast(b, shape)
                return e.__class__(a, b)
            else:
                return super().visit_Binary(e)
        elif isinstance(a_type, TileType) and isinstance(b_type, (PointerType, DataType)):
            b = full(b, a_type.shape)
            return e.__class__(a, b)
        elif isinstance(a_type, (PointerType, DataType)) and isinstance(b_type, TileType):
            a = full(a, b_type.shape)
            return e.__class__(a, b)
        else:
            return super().visit_Binary(e)


class InjectExplicitBroadcastPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InjectExplicitBroadcastRewriter()
        return rewriter.visit(func)


def inject_explicit_broadcast_pass() -> TileFunctionPass:
    return InjectExplicitBroadcastPass()
