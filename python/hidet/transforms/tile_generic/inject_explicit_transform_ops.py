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
from hidet.ir.tile.ops import Load, Store
from hidet.ir.utils.broadcast_utils import broadcast_shape
from hidet.transforms.base import TileFunctionPass
from hidet.utils import same_list


class InjectExplicitTransformOpsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def transform_to(self, src: Expr, dst_shape: List[int]) -> Expr:
        src_type = self.type_infer(src)
        if isinstance(src_type, TileType):
            src_shape: List[int] = list(src_type.shape)
            while len(src_shape) < len(dst_shape):
                src_shape.insert(0, 1)
                src = expand_dims(src, 0)
            for a, b in zip(src_shape, dst_shape):
                if a != b and a != 1:
                    raise ValueError(
                        'Cannot transform from shape {} to shape {} with expand_dims and broadcast'.format(
                            src_shape, dst_shape
                        )
                    )
            if not same_list(src_shape, dst_shape):
                src = broadcast(src, dst_shape)
            return src
        else:
            return full(src, dst_shape)

    def visit_Binary(self, e: BinaryExpr):
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer.visit(a)
        b_type = self.type_infer.visit(b)
        if isinstance(a_type, TileType) and isinstance(b_type, TileType):
            if not same_list(a_type.shape, b_type.shape):
                shape: List[int] = broadcast_shape(a_type.shape, b_type.shape)
                a = self.transform_to(a, shape)
                b = self.transform_to(b, shape)
        elif isinstance(a_type, TileType) and isinstance(b_type, (PointerType, DataType)):
            b = full(b, a_type.shape)
        elif isinstance(a_type, (PointerType, DataType)) and isinstance(b_type, TileType):
            a = full(a, b_type.shape)
        return e.__class__(a, b)

    def visit_Load(self, e: Load):
        ptr = self.visit(e.ptr)
        mask = self.visit(e.mask)
        other = self.visit(e.other)

        ptr_type: TileType = self.type_infer(ptr)

        if mask is not None:
            mask = self.transform_to(mask, ptr_type.shape)

        if other is not None:
            other = self.transform_to(other, ptr_type.shape)

        if ptr is e.ptr and mask is e.mask and other is e.other:
            return e
        else:
            return e.reforward([ptr, mask, other])

    def visit_Store(self, e: Store):
        ptr = self.visit(e.ptr)
        mask = self.visit(e.mask)
        value = self.visit(e.value)

        ptr_type: TileType = self.type_infer(ptr)

        if mask is not None:
            mask = self.transform_to(mask, ptr_type.shape)

        value = self.transform_to(value, ptr_type.shape)

        if ptr is e.ptr and mask is e.mask and value is e.value:
            return e
        else:
            return e.reforward([ptr, mask, value])


class InjectExplicitTransformOpsPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InjectExplicitTransformOpsRewriter()
        return rewriter.visit(func)


def inject_explicit_transform_ops_pass() -> TileFunctionPass:
    return InjectExplicitTransformOpsPass()
