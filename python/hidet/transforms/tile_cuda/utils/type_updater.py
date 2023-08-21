from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt


class TypeUpdater(IRRewriter):
    def visit_LetStmt(self, stmt: LetStmt):
        pass

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        pass
