from typing import Type, Dict, Union, List

from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr, var
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.functors import IRRewriter
from hidet.ir.func import Function
from hidet.ir import expr
from hidet.ir.type import PointerType, DataType
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.type import TileType, BlockLayout, block_layout, tile_type, flatten_block_layout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.utils import same_list
from hidet.ir.tile.ops import Arange, Full, Broadcast, Reshape, Load, Store, ConvertLayout, UnaryTileOp, BinaryTileOp
from hidet.ir.tile.ops import ExpandDims, convert_layout
from hidet.utils import prod, is_power_of_two

from .base import TileFunctionPass


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps
        self.type_infer = TypeInfer()

    def block_layout_from_shape(self, shape: List[int]) -> BlockLayout:
        num_elements = prod(shape)
        if not is_power_of_two(num_elements):
            raise ValueError(f"The tensor must have a power of 2 number of elements, got {num_elements}")
        size_per_thread = []
        thread_per_warp = []
        warps_per_block = []
        remaining_threads = 32
        remaining_warps = self.num_warps
        for extent in shape:
            size_per_thread.append(1)
            if extent <= remaining_threads:
                assert remaining_threads % extent == 0
                thread_per_warp.append(extent)
                warps_per_block.append(1)
                remaining_threads //= extent
            elif extent <= remaining_threads * remaining_warps:
                assert extent % remaining_threads == 0
                assert remaining_warps % (extent // remaining_threads) == 0
                thread_per_warp.append(remaining_threads)
                warps_per_block.append(extent // remaining_threads)
                remaining_threads = 1
                remaining_warps //= (extent // remaining_threads)
            else:
                thread_per_warp.append(remaining_threads)
                warps_per_block.append(remaining_warps)
                remaining_threads = 1
                remaining_warps = 1
        return block_layout(size_per_thread, thread_per_warp, warps_per_block)

    def visit_Arange(self, e: Arange):
        layout = self.block_layout_from_shape([e.end - e.begin])
        return Arange(e.begin, e.end, layout)

    def visit_Full(self, e: Full):
        layout = self.block_layout_from_shape(e.shape)
        return Full(e.value, e.shape, layout)

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
            y_layout = self.block_layout_from_shape(y_shape)
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
        rewriter = InstantiateLayoutRewriter(block_dim)
        return rewriter.visit(func)


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()
