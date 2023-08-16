from typing import List, Dict, Union, Optional, Callable

import hidet.ir.expr
from hidet.ir.type import DataType, PointerType, VoidType, void_p
from hidet.ir.expr import Var, Expr, var, tensor_var, logical_not, if_then_else, logical_and
from hidet.ir.stmt import Stmt, SeqStmt, EvaluateStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, BufferStoreStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, Store, Load
from hidet.ir.tile.ops import ExpandDims, convert_layout
from hidet.ir.tile.type import TileType, BlockLayout, block_layout, flatten_block_layout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tools import TypeInfer
from hidet.ir.mapping import repeat_map
from hidet.ir.builders import StmtBuilder
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from hidet.ir.primitives.cuda import threadIdx
from .base import TileFunctionPass


class Buffer:
    def __init__(self, base_type: Union[PointerType, DataType]):
        self.base_type: Union[PointerType, DataType] = base_type

    def __getitem__(self, item):
        raise NotImplementedError()


class BlockBuffer(Buffer):
    def __init__(self, buf_var: Var, shape: List[int], local_shape: List[int], layout: BlockLayout):
        super().__init__(buf_var.type.as_tensor_type().dtype)
        self.buf_var: Var = buf_var
        self.layout: BlockLayout = layout
        self.shape: List[int] = shape
        self.local_shape: List[int] = local_shape

    def __getitem__(self, item):
        return self.buf_var[item]


class TileToSirRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        # used communicate between CallTileOp and all kinds of TileOp
        self.op2buffer: Dict[TileOp, Buffer] = {}
        # the mapping from the defined var (with LetStmt or DeclareStmt) to the corresponding buffer
        self.var2buffer: Dict[Var, Buffer] = {}
        self.stmt_buffer: List[Stmt] = []
        self.type_infer = TypeInfer()

    def alloc_buffer(self, hint: str, ttype: TileType):
        if isinstance(ttype.layout, BlockLayout):
            return self.alloc_block_buffer(hint, ttype.type, ttype.shape, ttype.layout)
        else:
            raise NotImplementedError()

    def append_stmt(self, stmt: Stmt):
        self.stmt_buffer.append(stmt)

    def flush_stmts(self):
        stmts = self.stmt_buffer
        self.stmt_buffer = []
        return stmts

    @staticmethod
    def alloc_block_buffer(
        hint: str,
        dtype: Union[DataType, PointerType],
        shape: List[int],
        layout: BlockLayout
    ) -> BlockBuffer:
        local_shape = layout.local_shape(shape)
        buf_var = tensor_var(hint=hint, shape=local_shape, dtype=dtype)
        return BlockBuffer(buf_var, shape, local_shape, layout)

    def iterate_buffer_and_apply(self, buf, f_apply: Callable[[List[Expr], List[Expr], Expr, StmtBuilder], None]):
        layout = buf.layout
        local_shape: List[int] = layout.local_shape(buf.shape)

        sb = StmtBuilder()
        with sb.for_mapping(
            iter_names=[f'i{i}' for i in range(len(local_shape))],
            mapping=repeat_map(local_shape)
        ) as local_indices:
            global_indices, is_repeated = layout.local_to_global(local_indices, tid=threadIdx.x, global_shape=buf.shape)
            f_apply(local_indices, global_indices, is_repeated, sb)
        self.append_stmt(sb.finish())

    def declare_and_do(self, buf, do_func: Callable[[List[Expr], List[Expr], Expr, StmtBuilder], None]):
        if isinstance(buf, BlockBuffer):
            self.append_stmt(DeclareStmt(buf.buf_var))
        else:
            raise NotImplementedError()
        self.iterate_buffer_and_apply(buf, do_func)

    def declare_and_compute(self, buf, fcompute: Callable[[List[Expr], List[Expr], Expr], Expr]):
        def do_func(local_indices, global_indices, is_repeated, sb):
            value = fcompute(local_indices, global_indices, is_repeated)
            sb.append(BufferStoreStmt(buf.buf_var, indices=local_indices, value=value))
        self.declare_and_do(buf, do_func)

    def visit_CallTileOp(self, call: CallTileOp):
        return self.visit(call.op)

    def visit_LetStmt(self, stmt: LetStmt):
        stmts: List[Stmt] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallTileOp):
                # tile expression
                buf = self.visit(bind_value)
                assert isinstance(buf, BlockBuffer)
                self.var2buffer[bind_var] = buf
                stmts.extend(self.flush_stmts())
            else:
                # scalar expression
                stmts.append(DeclareStmt(bind_var, self.visit(bind_value)))
        stmts.append(self.visit(stmt.body))
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.init, CallTileOp):
            buf = self.visit(stmt.init)
            assert isinstance(buf, BlockBuffer)
            self.var2buffer[stmt.var] = buf
            stmts = self.flush_stmts()
            if len(stmts) == 1:
                return stmts[0]
            else:
                return SeqStmt(stmts)
        else:
            return super().visit_DeclareStmt(stmt)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallTileOp):
            ret = self.visit(stmt.expr)
            assert isinstance(ret, BlockBuffer) or ret is None
            stmts = self.flush_stmts()
            if len(stmts) == 1:
                return stmts[0]
            else:
                return SeqStmt(stmts)
        else:
            return super().visit_EvaluateStmt(stmt)

    def visit_Function(self, func: Function):
        ret = super().visit_Function(func)
        if ret.kind == 'cuda_tile':
            ret.kind = 'cuda_kernel'
        return ret

    def visit_Arange(self, e: Arange):
        out_type: TileType = self.type_infer(CallTileOp(e))
        buf = self.alloc_buffer('arange', out_type)

        if isinstance(buf, BlockBuffer):
            assert len(buf.shape) == 1
            self.declare_and_compute(
                buf, lambda local_indices, global_indices, is_repeated: global_indices[0] + e.begin
            )
        else:
            raise NotImplementedError()
        return buf

    def visit_Full(self, e: Full):
        out_type: TileType = self.type_infer(CallTileOp(e))
        buf = self.alloc_buffer('full', out_type)

        if isinstance(buf, BlockBuffer):
            self.declare_and_compute(buf, lambda local_indices, global_indices, is_repeated: self.visit(e.value))
        else:
            raise NotImplementedError()
        return buf

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        out_type: TileType = self.type_infer(CallTileOp(e))
        buf = self.alloc_buffer(e.name, out_type)
        cls_name = type(e).__name__
        if not hasattr(hidet.ir.expr, cls_name):
            raise NotImplementedError(f'No implementation for {cls_name} binary op')
        expr_cls = getattr(hidet.ir.expr, cls_name)
        assert isinstance(e.x, Var) and isinstance(e.y, Var)
        lhs_buf: Buffer = self.var2buffer[e.x]
        rhs_buf: Buffer = self.var2buffer[e.y]
        if isinstance(lhs_buf, BlockBuffer) and isinstance(rhs_buf, BlockBuffer):
            lhs_var = lhs_buf.buf_var
            rhs_var = rhs_buf.buf_var
            self.declare_and_compute(
                buf,
                lambda local_indices, global_indices, is_repeated: Expr._binary(
                    expr_cls, lhs_var[local_indices], rhs_var[local_indices]
                )
            )
        else:
            raise NotImplementedError()
        return buf

    def visit_Load(self, e: Load):
        from hidet.ir.primitives.cuda.ldst import load
        assert (
            isinstance(e.ptr, Var)
            and (e.mask is None or isinstance(e.mask, Var))
            and (e.other is None or isinstance(e.other, Var))
        )
        ptr_buf: Buffer = self.var2buffer[e.ptr]
        mask_buf: Optional[Buffer] = self.var2buffer.get(e.mask, None)
        other_buf: Optional[Buffer] = self.var2buffer.get(e.other, None)
        if isinstance(ptr_buf, BlockBuffer):
            assert mask_buf is None or isinstance(mask_buf, BlockBuffer)
            buf = self.alloc_buffer('load', self.type_infer(CallTileOp(e)))

            def f_compute(local_indices, global_indices, is_repeated):
                if mask_buf is None:
                    return load(ptr_buf[local_indices])
                else:
                    if other_buf is None:
                        assert isinstance(ptr_buf.base_type, PointerType)
                        value_type = ptr_buf.base_type.base_type
                        if isinstance(value_type, PointerType):
                            other_value = void_p(0)
                        elif isinstance(value_type, DataType):
                            other_value = value_type.zero
                        else:
                            raise NotImplementedError()
                    else:
                        other_value = other_buf[local_indices]

                    return if_then_else(mask_buf[local_indices], load(ptr_buf[local_indices]), other_value)

            self.declare_and_compute(buf, f_compute)
        else:
            raise NotImplementedError()
        return buf

    def visit_Store(self, e: Store):
        from hidet.ir.primitives.cuda.ldst import store
        assert isinstance(e.ptr, Var) and (e.mask is None or isinstance(e.mask, Var)) and isinstance(e.value, Var)
        ptr_buf: Buffer = self.var2buffer[e.ptr]
        value_buf: Buffer = self.var2buffer[e.value]
        mask_buf: Optional[Buffer] = self.var2buffer.get(e.mask, None)
        if isinstance(ptr_buf, BlockBuffer):
            assert isinstance(value_buf, BlockBuffer) and (mask_buf is None or isinstance(mask_buf, BlockBuffer))

            def f_apply(local_indices, global_indices, is_repeated, sb: StmtBuilder):
                assert isinstance(ptr_buf, BlockBuffer) and isinstance(value_buf, BlockBuffer)
                assert ptr_buf.layout == value_buf.layout

                if mask_buf:
                    assert isinstance(mask_buf, BlockBuffer) and ptr_buf.layout == mask_buf.layout
                    mask_value = mask_buf[local_indices]
                else:
                    mask_value = True

                with sb.if_then(logical_and(logical_not(is_repeated), mask_value)):
                    # the same element in the tile might be stored in multiple threads, this if statement
                    # ensures that only one thread stores the value
                    sb.append(store(
                        addr=ptr_buf.buf_var[local_indices],
                        value=value_buf.buf_var[local_indices]
                    ))

            self.iterate_buffer_and_apply(
                ptr_buf,
                f_apply
            )
        else:
            raise NotImplementedError()


class TileToSirPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = TileToSirRewriter()
        return rewriter.rewrite(func)


def tile_to_sir_pass() -> TileFunctionPass:
    return TileToSirPass()
