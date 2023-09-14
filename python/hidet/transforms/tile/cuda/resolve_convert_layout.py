from typing import List
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt, Stmt, EvaluateStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.ops.convert_layout import ConvertLayout, convert_layout
from hidet.ir.tile.ops.smem import AllocTensor
from hidet.ir.tile.ops.sync import SyncThreads
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import SharedLayout
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.generic.canonicalize_to_ssa import ConvertTileExprToLetRewriter
from hidet.transforms.expand_let_expr import LetExprExpander
from hidet.transforms.tile.utils import glue_let_chain


class ResolveConvertLayoutRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_LetStmt(self, stmt: LetStmt):
        seq: List[Stmt] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallTileOp) and isinstance(bind_value.op, ConvertLayout):
                op = bind_value.op
                x = self.visit(op.x)
                x_type = self.type_infer(x)
                x_scope: TileScope = x_type.scope
                y_scope: TileScope = op.scope

                assert isinstance(x_type, TileType)

                if x_scope.is_register() and y_scope.is_register() and x_type.layout == op.layout:
                    seq.append(LetStmt([bind_var], [bind_value]))
                elif x_scope.is_shared() and y_scope.is_shared():
                    seq.append(LetStmt([bind_var], [bind_value]))
                elif x_scope.is_register() and y_scope.is_register():
                    buf = AllocTensor(x_type.type, shape=x_type.shape).make_call()
                    buf_var = Var('buf', type=self.type_infer(buf))
                    updated_buf = StoreShared(x, buf_var).make_call()
                    updated_buf_var = Var('updated_buf', type=self.type_infer(updated_buf))
                    regs_buf = LoadShared(updated_buf_var, layout=op.layout).make_call()
                    seq.append(LetStmt([buf_var, updated_buf_var], [buf, updated_buf]))
                    seq.append(EvaluateStmt(SyncThreads().make_call()))
                    seq.append(LetStmt([bind_var], [regs_buf]))
                    seq.append(EvaluateStmt(SyncThreads().make_call()))
                elif x_scope.is_register() and y_scope.is_shared():
                    buf = AllocTensor(x_type.type, shape=x_type.shape, layout=op.layout).make_call()
                    buf_var = Var('buf', type=self.type_infer(buf))
                    updated_smem = StoreShared(x, buf_var).make_call()
                    seq.append(LetStmt([buf_var, bind_var], [buf, updated_smem]))
                elif x_scope.is_shared() and y_scope.is_register():
                    seq.append(LetStmt([bind_var], [LoadShared(x, layout=op.layout).make_call()]))
                else:
                    raise NotImplementedError()
            else:
                seq.append(LetStmt([bind_var], [bind_value]))
        body = self.visit(stmt.body)
        seq.append(body)
        return glue_let_chain(seq)

    def visit_ConvertLayout(self, e: ConvertLayout):
        raise ValueError('ConvertLayout should only appear in LetStmt in SSA form')


class ResolveConvertLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_transforms(
            func, [ResolveConvertLayoutRewriter(), ConvertTileExprToLetRewriter(), LetExprExpander()]
        )


def resolve_convert_layout_pass() -> TileFunctionPass:
    return ResolveConvertLayoutPass()
