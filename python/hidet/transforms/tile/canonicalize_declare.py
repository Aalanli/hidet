from typing import Type, Dict, Union

from hidet.ir import expr
from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.type import TileType
from hidet.ir.stmt import DeclareStmt, SeqStmt, AssignStmt, EvaluateStmt
from hidet.ir.func import Function
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.ops.assign import assign
import hidet.ir.tile.ops.arthimatic as arith
from .base import TileFunctionPass


class CanonicalizeDeclareRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        # declare a: tile = b
        # =>
        # declare a: tlie
        # a = b (AssignStmt)
        init = self.visit(stmt.init)
        if isinstance(stmt.var.type, TileType) and init is not None:
            seq = [
                DeclareStmt(stmt.var, init=None, scope=stmt.scope),
                EvaluateStmt(assign(stmt.var, init)),
            ]
            return SeqStmt(seq)
        else:
            return super().visit_DeclareStmt(stmt)


class CanonicalizeDeclarePass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = CanonicalizeDeclareRewriter()
        return rewriter.visit(func)


def canonicalize_declare_pass() -> TileFunctionPass:
    return CanonicalizeDeclarePass()
