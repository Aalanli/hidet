from typing import Dict, List, Optional, Union, Set
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import LetStmt, Stmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.ops import Full, Arange, Construct, Broadcast, ExpandDims, UnaryTileOp, BinaryTileOp
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType
from hidet.utils import same_list
from hidet.transforms.base import TileFunctionPass
from hidet.utils import repeat_until_converge


class DeadCodeEliminationRewriter(IRRewriter):

    def visit_Function(self, func: Function):
        self.memo.clear()   # in case calling this rewriter multiple times
        return super().visit_Function(func)

    def visit_LetStmt(self, stmt):
        body = self.visit(stmt.body)
        dead_vars: Set[Var] = set()
        for bind_var, bind_value in reversed(list(zip(stmt.bind_vars, stmt.bind_values))):
            if bind_var not in self.memo:
                dead_vars.add(bind_var)
            else:
                self.visit(bind_value)
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if bind_var not in dead_vars:
                bind_vars.append(bind_var)
                bind_values.append(bind_value)
        if len(bind_vars) == 0:
            return body
        elif len(bind_vars) == len(stmt.bind_vars) and body is stmt.body:
            return stmt
        else:
            return LetStmt(bind_vars=bind_vars, bind_values=bind_values, body=body)


class DeadCodeEliminationPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = DeadCodeEliminationRewriter()
        return rewriter.visit(func)


def eliminate_dead_code(node: Union[IRModule, Function, Stmt]):
    rewriter = DeadCodeEliminationRewriter()
    return rewriter.visit(node)


def dead_code_elimination_pass() -> TileFunctionPass:
    return DeadCodeEliminationPass()
