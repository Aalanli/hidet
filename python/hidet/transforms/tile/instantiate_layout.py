from typing import List

from hidet.ir.expr import Var, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, convert_layout
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import (
    TileLayout,
    SharedLayout,
    BlockLayout,
    DotOperandLayout,
    FlattenBlockLayout,
    BlockDotOperandLayout,
)
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from .base import TileFunctionPass
from .convert_tile_expr_to_let import convert_to_let


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps
        self.type_infer = TypeInfer()

    def visit_CallTileOp(self, call: CallTileOp):
        op = self.visit(call.op)
        if op is call.op:
            ret = call
        else:
            ret = op.make_call()
        ttype = self.type_infer.visit(ret)
        if isinstance(ttype, TileType) and ttype.layout is None:
            raise NotImplementedError(
                'The layout of the following tile op has not been instantiated:\n'
                + '  {}\n'.format(type(call.op).__name__)
            )
        return ret

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

    def visit_AssignStmt(self, stmt: AssignStmt):
        value = self.visit(stmt.value)
        stmt_var: Var = self.visit(stmt.var)
        value_type = self.type_infer.visit(value)
        if isinstance(value_type, TileType):
            assert isinstance(stmt_var.type, TileType)
            if value_type.layout != stmt_var.type.layout:
                value = convert_layout(value, stmt_var.type.layout)
            return AssignStmt(stmt_var, value)
        else:
            assert not isinstance(stmt_var.type, TileType)
            super().visit_AssignStmt(stmt)

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
            y_layout = BlockLayout(
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
            y_shape = x_type.shape[: e.axis] + [1] + x_type.shape[e.axis :]
            y_layout = BlockLayout.from_shape(y_shape, self.num_warps)
            return ExpandDims(
                x=convert_layout(x, layout=FlattenBlockLayout(y_layout, axis=e.axis)), axis=e.axis, layout=y_layout
            )
        else:
            raise NotImplementedError()

    def visit_ReduceOp(self, e: ReduceOp):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        if isinstance(x_type.layout, BlockLayout):
            y_type = e.infer_type([x_type])
            if e.keepdims:
                layout = x_type.layout
            else:
                layout = FlattenBlockLayout(x_type.layout, axis=e.axis)
            assert isinstance(y_type, TileType)
            return ReduceOp(x=x, axis=e.axis, keepdims=e.keepdims, kind=e.kind, layout=layout)
        else:
            raise NotImplementedError()

    def visit_Dot(self, e: Dot):
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer.visit(a)
        b_type = self.type_infer.visit(b)
        assert isinstance(a_type, TileType) and isinstance(b_type, TileType)
        m: int = a_type.shape[0]
        n: int = b_type.shape[1]
        num_threads = self.num_warps * 32
        if m * n >= num_threads * 16:
            size_per_thread = [4, 4]
        elif m * n >= num_threads * 4:
            size_per_thread = [2, 2]
        else:
            size_per_thread = [1, 1]
        layout = BlockLayout.from_shape([m, n], num_warps=self.num_warps, size_per_thread=size_per_thread)
        a = convert_layout(a, BlockDotOperandLayout(parent=layout, op_idx=0))
        b = convert_layout(b, BlockDotOperandLayout(parent=layout, op_idx=1))
        return Dot(a, b, e.kind, layout)


class InstantiateLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
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
        return convert_to_let(rewriter.visit(func))


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()
