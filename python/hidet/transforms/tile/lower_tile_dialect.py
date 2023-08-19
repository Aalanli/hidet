from typing import List, Dict, Union, Optional, Callable

from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import Var, Expr, tensor_var, if_then_else, logical_and
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.mapping import repeat_map
from hidet.ir.type import BaseType
from hidet.ir.stmt import LetStmt, DeclareStmt, BufferStoreStmt
from hidet.ir.stmt import Stmt, SeqStmt, EvaluateStmt, DeclareScope, AssignStmt
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.layout import (
    BlockLayout, FlattenBlockLayout, TileLayout, SharedLayout, DistributedLayout
)
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, Store, Load, DebugPrint
from hidet.ir.tile.ops import ExpandDims, ConvertLayout, ReduceOp
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType, PointerType, void_p
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
        if isinstance(layout, DistributedLayout):
            local_shape = layout.calc_local_shape(shape)
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

    def iterate_dist_buffer_and_apply(
        self, buf: Buffer, f_apply: Callable[[List[Expr], List[Expr], Expr, StmtBuilder], None]
    ):
        assert isinstance(buf.layout, DistributedLayout)
        layout: DistributedLayout = buf.layout
        local_shape: List[int] = layout.calc_local_shape(buf.shape)

        sb = StmtBuilder()
        with sb.for_mapping(repeat_map(local_shape)) as local_indices:
            global_indices, not_duplicated = layout.local_to_global(local_indices, global_shape=buf.shape)
            f_apply(local_indices, global_indices, not_duplicated, sb)
        self.append_stmt(sb.finish())

    def iterate_dist_buffer_and_compute(self, buf: Buffer, f_compute: Callable[[List[Expr], List[Expr], Expr], Expr]):
        def f_apply(local_indices, global_indices, not_duplicated, sb):
            value = f_compute(local_indices, global_indices, not_duplicated)
            sb.append(BufferStoreStmt(buf.var, indices=local_indices, value=value))

        self.iterate_dist_buffer_and_apply(buf, f_apply)

    def visit_CallTileOp(self, call: CallTileOp):
        args: List[Union[Expr, Buffer]] = []
        for arg in call.op.args:
            arg_type = self.type_infer(arg)
            if isinstance(arg_type, TileType):
                assert isinstance(arg, Var)
                args.append(self.get_buffer(arg))
            else:
                args.append(self.visit(arg))

        output_type: BaseType = self.type_infer(call)
        if output_type.is_void():
            output: Optional[Buffer] = None
        elif isinstance(output_type, TileType):
            output: Optional[Buffer] = self.alloc_buffer(call.op.name, output_type)
        else:
            raise NotImplementedError()
        self.append_stmt(implement_tile_op(call.op, args=args, output=output))

        return output

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
                self.memo[bind_var] = buf.var
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

            assert lhs.layout == rhs.layout and lhs.is_distributed()

            def f_compute(local_indices, global_indices, not_duplicated):
                return rhs[local_indices]

            self.iterate_dist_buffer_and_compute(lhs, f_compute)
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


class LowerTileDialectPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = LowerTileDialectRewriter()
        return rewriter.rewrite(func)


def lower_tile_dialect_pass() -> TileFunctionPass:
    return LowerTileDialectPass()
