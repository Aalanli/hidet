from typing import List
from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import Stmt


class PureForStmt(Stmt):
    def __init__(
        self,
        args: List[Expr],
        values: List[Expr],
        loop_var: Var,
        extent: Expr,
        body: Stmt,
        let_vars: List[Var],
        let_body
    ):
        self.args: List[Expr] = args
        self.values: List[Expr] = values
        self.loop_var: Var = loop_var
        self.extent: Expr = extent
        self.body: Stmt = body
        self.let_vars: List[Var] = let_vars
        self.let_body: Stmt = let_body


class PureYieldStmt(Stmt):
    def __init__(self, yields: List[Expr]):
        self.yields: List[Expr] = yields

