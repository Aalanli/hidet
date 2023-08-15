from typing import Type, Dict, Union, List

from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr
from hidet.ir.functors import IRRewriter
from hidet.ir.func import Function
from hidet.ir import expr
from hidet.ir.type import PointerType, DataType
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.type import TileType, BlockLayout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.utils import same_list
from hidet.ir.tile.ops import Arange, Full, Broadcast, Reshape, Load, Store, ConvertLayout, UnaryTileOp, BinaryTileOp
from hidet.ir.tile.ops import ExpandDims, convert_layout

from .base import TileFunctionPass


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps
        self.type_infer = TypeInfer()

    def block_layout_from_shape(self, shape: List[int]) -> BlockLayout:
        pass

    def visit_Arange(self, e: Arange):
        layout = self.block_layout_from_shape([e.end - e.begin])
        return Arange(e.begin, e.end, layout)

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        x = self.visit(e.x)
        y = self.visit(e.y)
        x_type = self.type_infer.visit(x)
        y_type = self.type_infer.visit(y)
        assert isinstance(x_type, TileType) and isinstance(y_type, TileType)
        assert same_list(x_type.shape, y_type.shape)

        if x_type.layout != y_type.layout:
            y = convert_layout(y, x_type.layout)
            return e.reforward([x, y])
        else:
            return super().visit_BinaryTileOp(e)

    def visit_Broadcast(self, e: Broadcast):
        x = self.visit(e.x)

    def visit_ExpandDims(self, e: ExpandDims):
        pass

    def visit_Reshape(self, e: Reshape):
        raise NotImplementedError()

    def visit_Full(self, e: Full):
        raise NotImplementedError()


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
