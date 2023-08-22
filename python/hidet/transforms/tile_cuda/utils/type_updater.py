from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.tile.stmt import PureForStmt


class TypeUpdater(IRRewriter):
    def visit_LetStmt(self, stmt: LetStmt):
        pass

    def visit_PureForStmt(self, stmt: PureForStmt):
        pass
