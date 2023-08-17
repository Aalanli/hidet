from typing import List, Dict, Union, Optional, Callable

import hidet.ir.expr
from hidet.ir.type import DataType, PointerType, VoidType, void_p
from hidet.ir.expr import Var, Expr, var, tensor_var, logical_not, if_then_else, logical_and
from hidet.ir.stmt import Stmt, SeqStmt, EvaluateStmt, DeclareScope
from hidet.ir.layout import DataLayout
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, BufferStoreStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, Store, Load, DebugPrint
from hidet.ir.tile.ops import ExpandDims, ConvertLayout
from hidet.ir.tile.type import TileType
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, TileLayout, SharedLayout
from hidet.ir.tile.layout import block_layout, flatten_block_layout
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tools import TypeInfer
from hidet.ir.mapping import repeat_map
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.cuda import threadIdx
from .base import TileFunctionPass


class Buffer:
    def __init__(
        self,
        hint: str,
        dtype: Union[PointerType, DataType],
        shape: List[int],
        local_shape: List[int],
        layout: TileLayout,
        var_data_layout: Optional[DataLayout] = None
    ):
        self.var: Var = tensor_var(hint=hint, shape=local_shape, dtype=dtype, layout=var_data_layout)
        self.dtype: Union[PointerType, DataType] = dtype
        self.shape: List[int] = shape
        self.local_shape: List[int] = local_shape
        self.layout: TileLayout = layout

    def __getitem__(self, item):
        return self.var[item]

    @property
    def block_layout(self) -> BlockLayout:
        assert isinstance(self.layout, BlockLayout)
        return self.layout

    @property
    def shared_layout(self) -> SharedLayout:
        assert isinstance(self.layout, SharedLayout)
        return self.layout

    @property
    def flatten_block_layout(self) -> FlattenBlockLayout:
        assert isinstance(self.layout, FlattenBlockLayout)
        return self.layout

    def is_shared(self):
        return isinstance(self.layout, SharedLayout)

    def is_block(self):
        return isinstance(self.layout, BlockLayout)

    def is_flatten_block(self):
        return isinstance(self.layout, FlattenBlockLayout)

    def is_block_like(self):
        return isinstance(self.layout, (BlockLayout, FlattenBlockLayout))


class TileToSirRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        # the mapping from the defined var (with LetStmt or DeclareStmt) to the corresponding buffer
        self.var2buffer: Dict[Var, Buffer] = {}
        self.stmt_buffer: List[Stmt] = []
        self.type_infer = TypeInfer()

    def get_buffer(self, e: Expr) -> Buffer:
        assert isinstance(e, Var)
        return self.var2buffer[e]

    def alloc_buffer(self, hint: str, top: TileOp) -> Buffer:
        ttype: TileType = self.type_infer(CallTileOp(top))
        layout: TileLayout = ttype.layout
        shape: List[int] = ttype.shape
        dtype: Union[DataType, PointerType] = ttype.type
        if isinstance(layout, BlockLayout):
            local_shape = layout.local_shape(shape)
        elif isinstance(layout, FlattenBlockLayout):
            local_shape = layout.local_shape(shape)
        elif isinstance(layout, SharedLayout):
            local_shape = layout.local_shape(shape)
        else:
            raise NotImplementedError()
        buf = Buffer(hint, dtype, shape, local_shape, layout)
        self.var2buffer[buf.var] = buf
        self.append_stmt(DeclareStmt(buf.var, scope=DeclareScope.Shared if buf.is_shared() else DeclareScope.Default))
        return buf

    def append_stmt(self, stmt: Stmt):
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
        with sb.for_mapping(
            iter_names=[f'i{i}' for i in range(len(local_shape))],
            mapping=repeat_map(local_shape)
        ) as local_indices:
            global_indices, is_repeated = layout.local_to_global(local_indices, tid=threadIdx.x, global_shape=buf.shape)
            f_apply(local_indices, global_indices, is_repeated, sb)
        self.append_stmt(sb.finish())

    def iterate_block_buffer_and_compute(
        self, buf: Buffer, f_compute: Callable[[List[Expr], List[Expr], Expr], Expr]
    ):
        def f_apply(local_indices, global_indices, is_repeated, sb):
            value = f_compute(local_indices, global_indices, is_repeated)
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
                assert isinstance(buf, Buffer), bind_value
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
            assert isinstance(buf, Buffer)
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
                buf, lambda local_indices, global_indices, is_repeated: global_indices[0] + e.begin
            )
        else:
            raise NotImplementedError()
        return buf

    def visit_Full(self, e: Full):
        buf = self.alloc_buffer('full', e)

        if isinstance(buf.layout, BlockLayout):
            self.iterate_block_buffer_and_compute(
                buf, lambda local_indices, global_indices, is_repeated: self.visit(e.value)
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
            lhs.is_block() and rhs.is_block() and lhs.layout == rhs.layout
            or lhs.is_flatten_block() and rhs.is_flatten_block() and lhs.layout == rhs.layout
        ):
            self.iterate_block_buffer_and_compute(
                buf,
                lambda local_indices, global_indices, is_repeated: e.apply_scalar(
                    lhs[local_indices], rhs[local_indices]
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
        if isinstance(ptr_buf.layout, BlockLayout):
            assert mask_buf is None or isinstance(mask_buf, Buffer)
            buf = self.alloc_buffer('load', e)

            def f_compute(local_indices, global_indices, is_repeated):
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

            def f_apply(local_indices, global_indices, is_repeated, sb: StmtBuilder):
                assert isinstance(ptr_buf, Buffer) and isinstance(value_buf, Buffer)
                assert ptr_buf.layout == value_buf.layout

                if mask_buf:
                    assert isinstance(mask_buf, Buffer) and ptr_buf.layout == mask_buf.layout
                    mask_value = mask_buf[local_indices]
                else:
                    mask_value = True

                with sb.if_then(logical_and(logical_not(is_repeated), mask_value)):
                    # the same element in the tile might be stored in multiple threads, this if statement
                    # ensures that only one thread stores the value
                    sb.append(
                        store(
                            addr=ptr_buf.var[local_indices],
                            value=value_buf.var[local_indices]
                        )
                    )

            self.iterate_block_buffer_and_apply(
                ptr_buf,
                f_apply
            )
        else:
            raise NotImplementedError()

    def visit_ConvertLayout(self, e: ConvertLayout):
        src: Buffer = self.get_buffer(e.x)
        dst: Buffer = self.alloc_buffer('cvt_layout', e)

        if src.is_block() and dst.is_block() and src.layout == dst.layout:
            return src
        elif src.is_block() and dst.is_flatten_block() and src.layout == dst.flatten_block_layout.parent:
            raise NotImplementedError()
        elif src.is_block() and dst.is_shared():
            def f_apply(local_indices, global_indices, is_repeated, sb: StmtBuilder):
                with sb.if_then(logical_not(is_repeated)):
                    sb.append(
                        BufferStoreStmt(
                            dst.var,
                            global_indices,
                            value=src[local_indices]
                        )
                    )

            self.iterate_block_buffer_and_apply(src, f_apply)
        elif src.is_flatten_block() and dst.is_block() and src.flatten_block_layout.parent == dst.layout:
            raise NotImplementedError()
        elif src.is_flatten_block() and dst.is_flatten_block() and src.layout == dst.layout:
            raise NotImplementedError()
        elif src.is_flatten_block() and dst.is_shared():
            raise NotImplementedError()
        elif src.is_shared() and dst.is_block_like():
            def f_compute(local_indices, global_indices, is_repeated):
                return src[global_indices]

            self.iterate_block_buffer_and_compute(dst, f_compute)
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

            def f_compute(local_indices, global_indices, is_repeated):
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
            def f_compute(local_indices, global_indices, is_repeated):
                local_indices = [idx if i not in broadcast_dims else 0 for i, idx in enumerate(local_indices)]
                return src[local_indices]
            self.iterate_block_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()
        return dst

    def visit_DebugPrint(self, e: DebugPrint):
        from hidet.ir.primitives.debug import printf
        from hidet.ir.primitives.cuda import syncthreads
        from hidet.ir.dtypes import float32, int32

        assert isinstance(e.x, Var)
        buf: Buffer = self.var2buffer[e.x]
        if isinstance(buf.layout, BlockLayout):
            layout: BlockLayout = buf.layout

            sb = StmtBuilder()
            shape = buf.shape
            with sb.for_mapping(
                iter_names=[f'i{i}' for i in range(len(shape))],
                mapping=repeat_map(buf.shape)
            ) as indices:
                local_indices, is_valid = layout.global_to_local(
                    indices, threadIdx.x, shape
                )
                dtype2fmt = {
                    float32: '%.2f',
                    int32: '%d'
                }
                with sb.if_then(is_valid):
                    assert len(shape) >= 1

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


class TileToSirPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = TileToSirRewriter()
        return rewriter.rewrite(func)


def tile_to_sir_pass() -> TileFunctionPass:
    return TileToSirPass()
