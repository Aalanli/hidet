from typing import Dict, List
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.functors import IRVisitor


class YieldStmtCollector(IRVisitor):
    def __init__(self):
        super().__init__()
        self.for2yield: Dict[PureForStmt, List[YieldStmt]] = {}

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        if for_stmt not in self.for2yield:
            self.for2yield[for_stmt] = []
        self.for2yield[for_stmt].append(stmt)


def collect_yield_stmts(node):
    collector = YieldStmtCollector()
    collector.visit(node)
    return collector.for2yield
