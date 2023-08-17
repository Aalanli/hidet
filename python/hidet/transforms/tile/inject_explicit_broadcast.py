from typing import Type, Dict, Union, List

from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr
from hidet.ir.stmt import AssignStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir import expr
from hidet.ir.type import PointerType, DataType
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.ops import broadcast, full
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

    def visit_AssignStmt(self, stmt: AssignStmt):
        rhs = self.visit(stmt.value)
        lhs = self.visit(stmt.var)
        rhs_type = self.type_infer.visit(rhs)
        lhs_type = self.type_infer.visit(lhs)

        if isinstance(lhs_type, TileType) and isinstance(rhs_type, (PointerType, DataType)):
            rhs = full(rhs, lhs_type.shape)
            return AssignStmt(lhs, rhs)
        elif isinstance(lhs_type, (PointerType, DataType)) and isinstance(rhs_type, TileType):
            raise ValueError('Cannot assign a tile to a non-tile variable')
        else:
            return super().visit_AssignStmt(stmt)


class InjectExplicitBroadcastPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InjectExplicitBroadcastRewriter()
        return rewriter.visit(func)


def inject_explicit_broadcast_pass() -> TileFunctionPass:
    return InjectExplicitBroadcastPass()
