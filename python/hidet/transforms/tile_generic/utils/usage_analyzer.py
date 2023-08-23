from typing import List, Dict, Set
from collections import defaultdict
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt, EvaluateStmt, SeqStmt
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.functors import IRVisitor
from hidet.ir.tile.expr import TileOp


class Usage:
    pass


class LetUsage(Usage):
    """ let ... = convert_layout(x) """

    def __init__(self, let_stmt, idx):
        self.let_stmt: LetStmt = let_stmt
        self.idx: int = idx


class StmtUsage(Usage):
    """
    for ... in ... with arg=x, ...
    yield x
    store(...)
    """
    def __init__(self, stmt):
        self.stmt = stmt


class VarUsage:
    def __init__(self):
        self.let_usages: List[LetUsage] = []
        self.stmt_usages: List[StmtUsage] = []


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
        for value in stmt.yields:
            for used_var in self.collect_used_vars(value):
                self.usages[used_var].stmt_usages.append(StmtUsage(stmt))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        for used_var in self.collect_used_vars(stmt.expr):
            self.usages[used_var].stmt_usages.append(StmtUsage(stmt))


def analyze_usage(node) -> Dict[Var, VarUsage]:
    analyzer = UsageAnalyzer()
    analyzer.visit(node)
    return analyzer.usages
