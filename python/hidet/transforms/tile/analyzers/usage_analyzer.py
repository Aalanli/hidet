from typing import List, Dict, Set, Union
from collections import defaultdict
from hidet.ir.expr import Let, Var, Expr
from hidet.ir.func import Function
from hidet.ir.stmt import LetStmt, EvaluateStmt, SeqStmt
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.functors import IRVisitor
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.ops import ConvertLayout


class LetUsage:
    """let ... = convert_layout(x)"""

    def __init__(self, let_stmt, idx):
        self.let_stmt: LetStmt = let_stmt
        self.idx: int = idx
        self.bind_var: Var = let_stmt.bind_vars[idx]
        self.bind_value: Expr = let_stmt.bind_values[idx]

    @property
    def op(self) -> TileOp:
        assert isinstance(self.bind_value, CallTileOp)
        return self.bind_value.op


class StmtUsage:
    """
    for ... in ... with arg=x, ...
        ...
        yield arg
    store(...)
    """

    def __init__(self, stmt):
        self.stmt = stmt


class VarUsage:
    def __init__(self):
        self.let_usages: List[LetUsage] = []
        self.stmt_usages: List[StmtUsage] = []

    def count(self):
        return len(self.let_usages) + len(self.stmt_usages)

    def call_op_let_usages(self):
        return [usage for usage in self.let_usages if isinstance(usage.bind_value, CallTileOp)]


class UsageAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.usages: Dict[Var, VarUsage] = defaultdict(VarUsage)
        self.used_vars: Set[Var] = set()

    def visit_Var(self, var: Var):
        self.used_vars.add(var)

    def collect_used_vars(self, expr) -> Set[Var]:
        self.used_vars.clear()
        self.visit(expr)
        return self.used_vars

    def visit_LetStmt(self, stmt: LetStmt):
        for idx, (bind_var, bind_value) in enumerate(zip(stmt.bind_vars, stmt.bind_values)):
            for used_var in self.collect_used_vars(bind_value):
                self.usages[used_var].let_usages.append(LetUsage(stmt, idx))
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for value in stmt.values:
            for used_var in self.collect_used_vars(value):
                self.usages[used_var].stmt_usages.append(StmtUsage(stmt))
        self.visit(stmt.body)
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for value in stmt.values:
            for used_var in self.collect_used_vars(value):
                self.usages[used_var].stmt_usages.append(StmtUsage(stmt))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        for used_var in self.collect_used_vars(stmt.expr):
            self.usages[used_var].stmt_usages.append(StmtUsage(stmt))


class LetSource:
    def __init__(self, let_stmt: LetStmt, idx: int):
        self.let_stmt: LetStmt = let_stmt
        self.idx: int = idx
        self.bind_var: Var = let_stmt.bind_vars[idx]
        self.bind_value: Expr = let_stmt.bind_values[idx]

# right now there is only the for statement in the tile dialect
# should there be any future statements, we can abstract the over the concept of a source via blocks
# eg. there are sources where the value is the result of a tile op, and sources where the value is the result
# of a control statement. This case is the basic block argument.
class ForArgSource:
    def __init__(self, stmt: PureForStmt, idx: int):
        self.stmt: PureForStmt = stmt
        self.idx: int = idx
        self.arg: Var = stmt.args[idx]
        self.value: Expr = stmt.values[idx]

class ForLetSource:
    def __init__(self, stmt: PureForStmt, idx: int):
        self.stmt: PureForStmt = stmt
        self.idx: int = idx
        self.let_var: Var = stmt.let_vars[idx]


class FuncSource:
    def __init__(self, func):
        self.func = func

class SourceAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.source: Dict[Var, Union[LetSource, ForArgSource, ForLetSource, FuncSource]] = {}
    
    def visit_LetStmt(self, stmt: LetStmt):
        for idx, (bind_var, bind_value) in enumerate(zip(stmt.bind_vars, stmt.bind_values)):
            assert bind_var not in self.source, f"var {bind_var} has multiple sources"
            self.source[bind_var] = LetSource(stmt, idx)
        self.visit(stmt.body)
    
    def visit_PureForStmt(self, stmt: PureForStmt):
        for idx, (arg, value) in enumerate(zip(stmt.args, stmt.values)):
            assert arg not in self.source, f"var {arg} has multiple sources"
            self.source[arg] = ForArgSource(stmt, idx)
        self.visit(stmt.body)

        for idx, let_var in enumerate(stmt.let_vars):
            assert let_var not in self.source, f"var {let_var} has multiple sources"
            self.source[let_var] = ForLetSource(stmt, idx)
        self.visit(stmt.let_body)

    def visit_Function(self, func: Function):
        for var in func.params:
            self.source[var] = FuncSource(func)

        return super().visit_Function(func)

def analyze_usage(node) -> Dict[Var, VarUsage]:
    analyzer = UsageAnalyzer()
    analyzer.visit(node)
    return analyzer.usages


def source_usage(node) -> Dict[Var, Union[LetSource, ForArgSource, ForLetSource, FuncSource]]:
    analyzer = SourceAnalyzer()
    analyzer.visit(node)
    return analyzer.source