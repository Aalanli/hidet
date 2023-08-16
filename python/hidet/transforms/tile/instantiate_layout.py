from typing import List

from hidet.ir.expr import var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp
from hidet.ir.tile.ops import ExpandDims, convert_layout
from hidet.ir.tile.type import TileType, BlockLayout, block_layout, flatten_block_layout
from hidet.ir.tools import TypeInfer
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from .base import TileFunctionPass


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps
        self.type_infer = TypeInfer()

    def visit_Arange(self, e: Arange):
        layout = BlockLayout.from_shape([e.end - e.begin], self.num_warps)
        return Arange(e.begin, e.end, layout)

    def visit_Full(self, e: Full):
        layout = BlockLayout.from_shape(e.shape, self.num_warps)
        value = self.visit(e.value)
        return Full(value, e.shape, layout)

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
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        if isinstance(x_type.layout, BlockLayout):
            y_layout = block_layout(
                size_per_thread=x_type.layout.size_per_thread,
                thread_per_warp=x_type.layout.thread_per_warp,
                warps_per_block=x_type.layout.warps_per_block,
            )
        else:
            raise NotImplementedError()
        return Broadcast(x, e.shape, y_layout)

    def visit_ExpandDims(self, e: ExpandDims):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        if isinstance(x_type.layout, BlockLayout):
            y_shape = x_type.shape[:e.axis] + [1] + x_type.shape[e.axis:]
            y_layout = BlockLayout.from_shape(y_shape, self.num_warps)
            return ExpandDims(
                x=convert_layout(x, layout=flatten_block_layout(y_layout, axis=e.axis)),
                axis=e.axis,
                layout=y_layout
            )
        else:
            raise NotImplementedError()

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for orig_var, orig_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(orig_value)
            bind_var = var(orig_var.hint, self.type_infer(bind_value))
            self.memo[orig_var] = bind_var
            bind_vars.append(bind_var)
            bind_values.append(bind_value)
        body = self.visit(stmt.body)
        return LetStmt(bind_vars, bind_values, body)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt.init is None:
            return super().visit_DeclareStmt(stmt)
        else:
            init = self.visit(stmt.init)
            stmt_var = var(stmt.var.hint, self.type_infer(init))
            self.memo[stmt.var] = stmt_var
            return DeclareStmt(stmt_var, init)


class InstantiateLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        if func.kind != 'cuda_tile':
            return func
        if 'cuda.block_dim' not in func.attrs:
            raise ValueError("cuda.block_dim must be specified for 'cuda_tile' function")
        block_dim = func.attrs['cuda.block_dim']
        try:
            block_dim = int(block_dim)
        except ValueError:
            raise ValueError(f"cuda.block_dim must be a constant integer, got {block_dim}")
        num_warps = block_dim // 32
        if block_dim % 32 != 0:
            raise ValueError(f"cuda.block_dim must be a multiple of 32, got {block_dim}")
        rewriter = InstantiateLayoutRewriter(num_warps)
        return rewriter.visit(func)


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()
