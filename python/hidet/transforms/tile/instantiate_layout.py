from typing import Type, Dict, Union, List

from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr
from hidet.ir.functors import IRRewriter
from hidet.ir.func import Function
from hidet.ir import expr
from hidet.ir.type import PointerType, DataType
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.ops.transform import broadcast, full
from hidet.ir.utils.broadcast_utils import broadcast_shape
from hidet.ir.tile.type import TileType, BlockLayout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.mixin import TileOpMixin
from hidet.utils import same_list
from hidet.ir.tile.ops import Arange, Full, Broadcast, Reshape, Load, Store, ConvertLayout, UnaryTileOp, BinaryTileOp

from .base import TileFunctionPass


class InstantiateLayoutRewriter(IRRewriter, TileOpMixin):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps
        self.type_infer = TypeInfer()

    def block_layout_from_shape(self, shape: List[int]) -> BlockLayout:
        pass

    def visit_CallTileOp(self, call: CallTileOp):
        return self.dispatch_CallTileOp(call)

    def visit_Arange(self, e: Arange):
        raise NotImplementedError()

    def visit_Load(self, e: Load):
        raise NotImplementedError()

    def visit_Store(self, e: Store):
        raise NotImplementedError()

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        raise NotImplementedError()

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        raise NotImplementedError()

    def visit_Broadcast(self, e: Broadcast):
        raise NotImplementedError()

    def visit_Reshape(self, e: Reshape):
        raise NotImplementedError()

    def visit_Full(self, e: Full):
        raise NotImplementedError()

    def visit_ConvertLayout(self, e: ConvertLayout):
        pass


class InstantiateLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        if func.kind != 'cuda_tile':
            return func
        block_dim = func.attrs['cuda_block_dim']
        try:
            block_dim = int(block_dim)
        except ValueError:
            raise ValueError(f"cuda.block_dim must be a constant integer, got {block_dim}")
        rewriter = InstantiateLayoutRewriter(block_dim)
        return rewriter.visit(func)


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()
