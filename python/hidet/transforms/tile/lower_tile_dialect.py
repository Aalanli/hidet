from typing import List, Dict, Union, Optional, Callable

import hidet.ir.expr
from hidet.ir.type import DataType, PointerType, VoidType, void_p, tensor_pointer_type, sizeof
from hidet.ir.expr import Var, Expr, var, tensor_var, logical_not, if_then_else, logical_and, tensor_pointer_var
from hidet.ir.stmt import Stmt, SeqStmt, EvaluateStmt, DeclareScope, AssignStmt
from hidet.ir.layout import DataLayout
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, BufferStoreStmt
from hidet.ir.primitives.cuda.tile import alloc_shared
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, Store, Load, DebugPrint
from hidet.ir.tile.ops import ExpandDims, ConvertLayout, ReduceOp
from hidet.ir.tile.type import TileType
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, TileLayout, SharedLayout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tools import TypeInfer
from hidet.ir.mapping import repeat_map
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.cuda import threadIdx
from .base import TileFunctionPass
from .lower_ops import Buffer, implement_tile_op


class LowerTileDialectRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        # the mapping from the defined var (within LetStmt or DeclareStmt and Buffer) to the corresponding buffer
        # the defined var can be either the var in LetStmt/DeclareStmt, or the one created in Buffer
        self.var2buffer: Dict[Var, Buffer] = {}
        self.stmt_buffer: List[Stmt] = []
        self.type_infer = TypeInfer()

    def get_buffer(self, e: Expr) -> Buffer:
        assert isinstance(e, Var)
        return self.var2buffer[e]

    def alloc_buffer(self, hint: str, tile_op_or_type: Union[TileOp, TileType]) -> Buffer:
        if isinstance(tile_op_or_type, TileOp):
            ttype: TileType = self.type_infer(CallTileOp(tile_op_or_type))
        else:
            ttype: TileType = tile_op_or_type
        layout: TileLayout = ttype.layout
        shape: List[int] = ttype.shape
        dtype: Union[DataType, PointerType] = ttype.type
        if isinstance(layout, BlockLayout):
            local_shape = layout.local_shape(shape)
            buf_var: Var = tensor_var(hint=hint, shape=local_shape, dtype=dtype)
            self.append_stmt(DeclareStmt(buf_var))
        elif isinstance(layout, FlattenBlockLayout):
            local_shape = layout.local_shape(shape)
            buf_var: Var = tensor_var(hint=hint, shape=local_shape, dtype=dtype)
            self.append_stmt(DeclareStmt(buf_var))
        elif isinstance(layout, SharedLayout):
            local_shape = layout.local_shape(shape)
            # buf_var: Var = tensor_pointer_var(hint, shape, dtype, layout=layout.data_layout)
            # self.append_stmt(DeclareStmt(buf_var, init=alloc_shared(sizeof(buf_var.type.tensor_type))))
            buf_var: Var = tensor_var(hint, shape, dtype, layout=layout.data_layout)
            self.append_stmt(DeclareStmt(buf_var, scope=DeclareScope.Shared))
        else:
            raise NotImplementedError()
        buf = Buffer(buf_var, dtype, shape, local_shape, layout)
        self.var2buffer[buf_var] = buf
        return buf

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmt_buffer.append(stmt)

    def flush_stmts(self):
        stmts = self.stmt_buffer
        self.stmt_buffer = []
        return stmts

    def iterate_block_buffer_and_apply(
        self, buf: Buffer, f_apply: Callable[[List[Expr], List[Expr], Expr, StmtBuilder], None]
    ):
        assert isinstance(buf.layout, (BlockLayout, FlattenBlockLayout))
        layout: Union[BlockLayout, FlattenBlockLayout] = buf.layout
        local_shape: List[int] = layout.local_shape(buf.shape)

        sb = StmtBuilder()
        with sb.for_mapping(repeat_map(local_shape)) as local_indices:
            global_indices, not_duplicated = layout.local_to_global(local_indices, global_shape=buf.shape)
            f_apply(local_indices, global_indices, not_duplicated, sb)
        self.append_stmt(sb.finish())

    def iterate_block_buffer_and_compute(self, buf: Buffer, f_compute: Callable[[List[Expr], List[Expr], Expr], Expr]):
        def f_apply(local_indices, global_indices, not_duplicated, sb):
            value = f_compute(local_indices, global_indices, not_duplicated)
            sb.append(BufferStoreStmt(buf.var, indices=local_indices, value=value))

        self.iterate_block_buffer_and_apply(buf, f_apply)

    def visit_CallTileOp(self, call: CallTileOp):
        return self.visit(call.op)

    def visit_LetStmt(self, stmt: LetStmt):
        stmts: List[Stmt] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallTileOp):
                # tile expression
                buf = self.visit(bind_value)
                if not isinstance(buf, Buffer):
                    raise NotImplementedError(
                        'The following tile expression has not been lowered to Buffer:\n' +
                        '  {}'.format(type(bind_value.op).__name__)
                    )
                self.var2buffer[bind_var] = buf
                stmts.extend(self.flush_stmts())
            elif isinstance(bind_value, Var) and isinstance(bind_value.type, TileType):
                self.memo[bind_var] = bind_value
            else:
                # scalar expression, or pure tile var to var binding
                stmts.append(DeclareStmt(bind_var, self.visit(bind_value)))
        stmts.append(self.visit(stmt.body))
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.var.type, TileType):
            assert stmt.init is None, 'canonicalize_declare pass should have removed the init value for tile var'
            buf = self.alloc_buffer(stmt.var.name, stmt.var.type)
            self.var2buffer[stmt.var] = buf
            return DeclareStmt(buf.var)
        else:
            return super().visit_DeclareStmt(stmt)

    def visit_AssignStmt(self, stmt: AssignStmt):
        lhs_var: Var = self.visit(stmt.var)
        rhs_expr: Expr = self.visit(stmt.value)
        if isinstance(lhs_var.type, TileType):
            assert isinstance(rhs_expr, Var), 'All call to tile op should in LetStmt'
            lhs: Buffer = self.get_buffer(lhs_var)
            rhs: Buffer = self.get_buffer(rhs_expr)

            assert lhs.layout == rhs.layout and lhs.is_block_like()

            def f_compute(local_indices, global_indices, not_duplicated):
                return rhs[local_indices]

            self.iterate_block_buffer_and_compute(lhs, f_compute)
            return SeqStmt(self.flush_stmts())
        else:
            return super().visit_AssignStmt(stmt)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallTileOp):
            ret = self.visit(stmt.expr)
            assert isinstance(ret, Buffer) or ret is None
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
        buf = self.alloc_buffer('arange', e)

        if isinstance(buf.layout, BlockLayout):
            assert len(buf.shape) == 1
            self.iterate_block_buffer_and_compute(
                buf, lambda local_indices, global_indices, not_duplicated: global_indices[0] + e.begin
            )
        else:
            raise NotImplementedError()
        return buf

    def visit_Full(self, e: Full):
        buf = self.alloc_buffer('full', e)

        if isinstance(buf.layout, BlockLayout):
            self.iterate_block_buffer_and_compute(
                buf, lambda local_indices, global_indices, not_duplicated: self.visit(e.value)
            )
        else:
            raise NotImplementedError()
        return buf

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        assert isinstance(e.x, Var) and isinstance(e.y, Var)
        lhs: Buffer = self.var2buffer[e.x]
        rhs: Buffer = self.var2buffer[e.y]
        buf = self.alloc_buffer(e.name, e)
        if (
            lhs.is_block()
            and rhs.is_block()
            and lhs.layout == rhs.layout
            or lhs.is_flatten_block()
            and rhs.is_flatten_block()
            and lhs.layout == rhs.layout
        ):
            self.iterate_block_buffer_and_compute(
                buf,
                lambda local_indices, global_indices, not_duplicated: e.apply_scalar(
                    lhs[local_indices], rhs[local_indices]
                ),
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
        if isinstance(ptr_buf.layout, BlockLayout):
            assert mask_buf is None or isinstance(mask_buf, Buffer)
            buf = self.alloc_buffer('load', e)

            def f_compute(local_indices, global_indices, not_duplicated):
                if mask_buf is None:
                    return load(ptr_buf[local_indices])
                else:
                    if other_buf is None:
                        assert isinstance(ptr_buf.dtype, PointerType)
                        value_type = ptr_buf.dtype.base_type
                        if isinstance(value_type, PointerType):
                            other_value = void_p(0)
                        elif isinstance(value_type, DataType):
                            other_value = value_type.zero
                        else:
                            raise NotImplementedError()
                    else:
                        other_value = other_buf[local_indices]

                    return if_then_else(mask_buf[local_indices], load(ptr_buf[local_indices]), other_value)

            self.iterate_block_buffer_and_compute(buf, f_compute)
        else:
            raise NotImplementedError()
        return buf

    def visit_Store(self, e: Store):
        from hidet.ir.primitives.cuda.ldst import store

        assert isinstance(e.ptr, Var) and (e.mask is None or isinstance(e.mask, Var)) and isinstance(e.value, Var)
        ptr_buf: Buffer = self.var2buffer[e.ptr]
        value_buf: Buffer = self.var2buffer[e.value]
        mask_buf: Optional[Buffer] = self.var2buffer.get(e.mask, None)
        if isinstance(ptr_buf.layout, BlockLayout):
            assert isinstance(value_buf, Buffer) and (mask_buf is None or isinstance(mask_buf, Buffer))

            def f_apply(local_indices, global_indices, not_duplicated, sb: StmtBuilder):
                assert isinstance(ptr_buf, Buffer) and isinstance(value_buf, Buffer)
                assert ptr_buf.layout == value_buf.layout

                if mask_buf:
                    assert isinstance(mask_buf, Buffer) and ptr_buf.layout == mask_buf.layout
                    mask_value = mask_buf[local_indices]
                else:
                    mask_value = True

                with sb.if_then(logical_and(not_duplicated, mask_value)):
                    # the same element in the tile might be stored in multiple threads, this if statement
                    # ensures that only one thread stores the value
                    sb.append(store(addr=ptr_buf.var[local_indices], value=value_buf.var[local_indices]))

            self.iterate_block_buffer_and_apply(ptr_buf, f_apply)
        else:
            raise NotImplementedError()

    def visit_ConvertLayout(self, e: ConvertLayout):
        from hidet.ir.primitives.cuda import syncthreads

        src: Buffer = self.get_buffer(e.x)
        dst: Buffer = self.alloc_buffer('cvt_layout', e)

        if src.is_block() and dst.is_block() and src.layout == dst.layout:
            return src
        elif src.is_block() and dst.is_flatten_block() and src.layout == dst.flatten_block_layout.parent:
            raise NotImplementedError()
        elif src.is_block() and dst.is_shared():

            def f_apply(local_indices, global_indices, not_duplicated, sb: StmtBuilder):
                with sb.if_then(not_duplicated):
                    sb.append(BufferStoreStmt(dst.var, global_indices, value=src[local_indices]))

            self.iterate_block_buffer_and_apply(src, f_apply)
            self.append_stmt(syncthreads())
        elif src.is_flatten_block() and dst.is_block() and src.flatten_block_layout.parent == dst.layout:
            raise NotImplementedError()
        elif src.is_flatten_block() and dst.is_flatten_block() and src.layout == dst.layout:
            raise NotImplementedError()
        elif src.is_flatten_block() and dst.is_shared():
            raise NotImplementedError()
        elif src.is_shared() and dst.is_block_like():

            def f_compute(local_indices, global_indices, not_duplicated):
                return src[global_indices]

            self.iterate_block_buffer_and_compute(dst, f_compute)
            self.append_stmt(syncthreads())
        elif src.is_shared() and dst.is_shared():
            raise NotImplementedError()
        else:
            raise ValueError(
                'can not convert the layout from {} to {}, please use canonicalize_convert_layout pass first.'.format(
                    src.layout, dst.layout
                )
            )
        return dst

    def visit_ExpandDims(self, e: ExpandDims):
        src: Buffer = self.get_buffer(e.x)
        dst: Buffer = self.alloc_buffer('expand_dims', e)

        if src.is_flatten_block() and dst.is_block() and src.flatten_block_layout.parent == dst.layout:
            assert src.flatten_block_layout.axis == e.axis

            def f_compute(local_indices, global_indices, not_duplicated):
                return src[local_indices]

            self.iterate_block_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()
        return dst

    def visit_Broadcast(self, e: Broadcast):
        src: Buffer = self.get_buffer(e.x)
        dst: Buffer = self.alloc_buffer('broadcast', e)

        broadcast_dims = [i for i in range(len(dst.shape)) if dst.shape[i] != src.shape[i]]

        if src.is_block_like() and dst.is_block_like() and src.layout == dst.layout:

            def f_compute(local_indices, global_indices, not_duplicated):
                local_indices = [idx if i not in broadcast_dims else 0 for i, idx in enumerate(local_indices)]
                return src[local_indices]

            self.iterate_block_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()
        return dst

    def visit_ReduceOp(self, e: ReduceOp):
        src: Buffer = self.get_buffer(e.x)
        dst: Buffer = self.alloc_buffer('reduce_op', e)

        self.append_stmt(implement_tile_op(e, args=[src], output=dst))

        return dst

    def visit_DebugPrint(self, e: DebugPrint):
        from hidet.ir.primitives.debug import printf
        from hidet.ir.primitives.cuda import syncthreads
        from hidet.ir.dtypes import float32, int32

        assert isinstance(e.x, Var)
        buf: Buffer = self.var2buffer[e.x]
        if buf.is_block_like():
            assert isinstance(buf.layout, (BlockLayout, FlattenBlockLayout))
            layout: Union[BlockLayout, FlattenBlockLayout] = buf.block_like_layout

            sb = StmtBuilder()
            shape = buf.shape
            with sb.for_mapping(repeat_map(buf.shape)) as indices:
                local_indices, is_valid = layout.global_to_local(indices, shape)
                dtype2fmt = {float32: '%.2f', int32: '%d'}
                with sb.if_then(is_valid):
                    if len(shape) == 0:
                        sb.append(printf(f'{dtype2fmt[buf.dtype]}\n', buf[local_indices]))
                    else:
                        if len(shape) == 1:
                            with sb.if_then(indices[0] == 0):
                                sb.append(printf('['))
                        else:
                            with sb.if_then(logical_and(indices[-2] == 0, indices[-1] == 0)):
                                sb.append(printf('[['))
                            with sb.otherwise():
                                with sb.if_then(indices[-1] == 0):
                                    sb.append(printf(' ['))
                        sb.append(printf(f'{dtype2fmt[buf.dtype]}', buf[local_indices]))
                        with sb.if_then(indices[-1] == shape[-1] - 1):
                            if len(shape) == 1:
                                sb.append(printf(']\n'))
                            else:
                                with sb.if_then(indices[-2] != shape[-2] - 1):
                                    sb.append(printf(']\n'))
                                with sb.otherwise():
                                    sb.append(printf(']]\n'))
                                    if len(shape) > 2:
                                        sb.append(printf('\n'))
                        with sb.otherwise():
                            sb.append(printf(', '))
                sb.append(syncthreads())

            self.append_stmt(sb.finish())
        else:
            raise NotImplementedError()


class LowerTileDialectPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = LowerTileDialectRewriter()
        return rewriter.rewrite(func)


def lower_tile_dialect_pass() -> TileFunctionPass:
    return LowerTileDialectPass()
