from typing import Dict, List, Optional, Union, Set, List
from collections import defaultdict
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import LetStmt, Stmt, EvaluateStmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.tools import collect
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.ops import Full, Arange, Construct, Broadcast, ExpandDims, UnaryTileOp, BinaryTileOp
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType
from hidet.utils import same_list
from hidet.transforms.base import TileFunctionPass
from hidet.utils import repeat_until_converge
from hidet.transforms.tile_generic.utils import DependencyAnalyzer, collect_yield_stmts


"""
Dead code elimination pass eliminates the code that does not affect the final result.

We solve following equation:

live[u] = u used in an operator that writes to memory (e.g., store)
        | any(live[v] for v that depends u)

where live[u] is a boolean value indicating whether u is live or not.

This pass assumes that the input function is in SSA form.
"""


class DeadCodeEliminationRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.live: Set[Var] = set()

    def visit_Function(self, func: Function):
        self.memo.clear()   # in case calling this rewriter multiple times

        # get the dependency relation-ship
        dependency_analyzer = DependencyAnalyzer()
        dependency_analyzer.visit_Function(func)
        depends: Dict[Var, List[Var]] = dependency_analyzer.depends

        # add the dependency relation-ship from pure for arg to its yield stmt
        for2yields: Dict[PureForStmt, List[YieldStmt]] = collect_yield_stmts(func)
        for for_stmt, yield_stmts in for2yields.items():
            for yield_stmt in yield_stmts:
                for arg, value in zip(for_stmt.args, yield_stmt.values):
                    depends[arg].append(value)

        # find all the CallTileOp and mark the args of the memory-writing ops as live
        roots: List[Var] = []
        for call_tile_op in collect(func, CallTileOp):
            op: TileOp = call_tile_op.op
            if op.write_memory_op():
                for arg in op.args:
                    assert isinstance(arg, Var), 'DeadCodeEliminationRewriter only works on SSA form'
                    roots.append(arg)

        # mark the roots and all its dependencies as live
        stack: List[Var] = roots
        self.live: Set[Var] = set(roots)
        while len(stack) > 0:
            u = stack.pop()
            for v in depends[u]:
                if v not in self.live:
                    self.live.add(v)
                    stack.append(v)

        return super().visit_Function(func)

    def visit_LetStmt(self, stmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if bind_var in self.live:
                bind_vars.append(bind_var)
                bind_values.append(bind_value)
        body = self.visit(stmt.body)
        if len(bind_vars) == len(stmt.bind_vars) and body is stmt.body:
            return stmt
        else:
            if len(bind_vars) == 0:
                return body
            else:
                return LetStmt(bind_vars, bind_values, body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        pass


class DeadCodeEliminationPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = DeadCodeEliminationRewriter()
        return rewriter.visit(func)


def eliminate_dead_code(node: Union[IRModule, Function, Stmt]):
    rewriter = DeadCodeEliminationRewriter()
    return rewriter.visit(node)


def dead_code_elimination_pass() -> TileFunctionPass:
    return DeadCodeEliminationPass()
