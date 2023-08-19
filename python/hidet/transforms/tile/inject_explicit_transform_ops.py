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
from hidet.ir.tile.ops import broadcast, full, expand_dims
from hidet.ir.utils.broadcast_utils import broadcast_shape
from .base import TileFunctionPass
from hidet.utils import same_list


class InjectExplicitTransformOpsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def transform_to(self, src: Expr, src_shape: List[int], dst_shape: List[int]) -> Expr:
        src_shape = list(src_shape)  # copy to avoid modifying the original list
        while len(src_shape) < len(dst_shape):
            src_shape.insert(0, 1)
            src = expand_dims(src, 0)
        for a, b in zip(src_shape, dst_shape):
            if a != b and a != 1:
                raise ValueError('Cannot transform from shape {} to shape {} with expand_dims and broadcast'.format(
                    src_shape, dst_shape
                ))
        if not same_list(src_shape, dst_shape):
            src = broadcast(src, dst_shape)
        return src

    def visit_Binary(self, e: BinaryExpr):
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer.visit(a)
        b_type = self.type_infer.visit(b)
        if isinstance(a_type, TileType) and isinstance(b_type, TileType):
            if not same_list(a_type.shape, b_type.shape):
                shape: List[int] = broadcast_shape(a_type.shape, b_type.shape)
                a = self.transform_to(a, a_type.shape, shape)
                b = self.transform_to(b, b_type.shape, shape)
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


class InjectExplicitTransformOpsPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InjectExplicitTransformOpsRewriter()
        return rewriter.visit(func)


def inject_explicit_transform_ops_pass() -> TileFunctionPass:
    return InjectExplicitTransformOpsPass()
