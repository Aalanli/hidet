from typing import List, Dict, Union, Optional

from hidet.ir.type import DataType, PointerType
from hidet.ir.expr import Var, Expr, var, tensor_var
from hidet.ir.stmt import Stmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp
from hidet.ir.tile.ops import ExpandDims, convert_layout
from hidet.ir.tile.type import TileType, BlockLayout, block_layout, flatten_block_layout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tools import TypeInfer
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from .base import TileFunctionPass


class Buffer:
    def __getitem__(self, item):
        raise NotImplementedError()

class BlockBuffer(Buffer):
    def __init__(self, buf_var: Var, shape: List[int], local_shape: List[int], layout: BlockLayout):
        self.buf_var: Var = buf_var
        self.layout: BlockLayout = layout
        self.shape: List[int] = shape
        self.local_shape: List[int] = local_shape

    def __getitem__(self, item):
        pass


class TileToSirRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.var2buffer: Dict[Var, Buffer] = {}
        self.op2var: Dict[TileOp, Var] = {}
        self.stmt_buffer: List[Stmt] = []

    def alloc_buffer(self, hint: str, ttype: TileType):
        if isinstance(ttype.layout, BlockLayout):
            return self.alloc_block_buffer(hint, ttype.type, ttype.shape, ttype.layout)
        else:
            raise NotImplementedError()

    def append_stmt(self, stmt: Stmt):
        self.stmt_buffer.append(stmt)

    def declare_buffer(self, buffer: Buffer):
        if isinstance(buffer, BlockBuffer):
            self.append_stmt(DeclareStmt(buffer.buf_var))
        else:
            raise NotImplementedError()

    @staticmethod
    def alloc_block_buffer(
        hint: str,
        dtype: Union[DataType, PointerType],
        shape: List[int],
        layout: BlockLayout
    ) -> BlockBuffer:
        local_shape: List[int] = []
        for extent, a, b, c in zip(shape, layout.size_per_thread, layout.thread_per_warp, layout.warps_per_block):
            layout_extent = a * b * c
            assert extent % layout_extent == 0 or layout_extent % extent == 0
            if extent <= layout_extent == 0:
                local_extent = a
            else:
                local_extent = a * (extent // layout_extent)
            local_shape.append(local_extent)
        buf_var = tensor_var(hint=hint, shape=local_shape, dtype=dtype)
        return BlockBuffer(buf_var, shape, local_shape, layout)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            assert isinstance(bind_var, Var)
            if bind_var.type.is_tile_type():
                buffer = self.alloc_buffer(
                    hint=bind_var.name,
                    ttype=bind_var.type.as_tile_type()
                )
                self.var2buffer[bind_var] = buffer
                self.declare_buffer(buffer)
                self.visit(bind_value)

    def visit_CallTileOp(self, call: CallTileOp) -> Optional[Buffer]:
        pass

    def visit_Arange(self, e: Arange):
        return Arange(e.begin, e.end, e.layout)


class TileToSirPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = TileToSirRewriter()
        return rewriter.rewrite(func)


def tile_to_sir_pass() -> TileFunctionPass:
    return TileToSirPass()
