from typing import List, Union
from hidet.ir.functors import IRRewriter
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.module import IRModule
from hidet.ir.func import Function
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.stmt import PureForStmt
from hidet.utils import same_list


class TypeUpdater(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            updated_bind_value = self.visit(bind_value)
            updated_bind_var = Var(bind_var.hint, self.type_infer(updated_bind_value))
            bind_values.append(updated_bind_value)
            bind_vars.append(updated_bind_var)
            self.memo[bind_var] = updated_bind_var
        body = self.visit(stmt.body)
        return LetStmt(bind_vars, bind_values, body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        values = self.visit(stmt.values)
        args: List[Var] = []
        let_vars: List[Var] = []
        for arg, value, let_var in zip(stmt.args, values, stmt.let_vars):
            updated_var = Var(arg.hint, self.type_infer(value))
            updated_let_var = Var(let_var.hint, self.type_infer(value))
            self.memo[arg] = updated_var
            self.memo[let_var] = updated_let_var
            args.append(updated_var)
            let_vars.append(updated_let_var)
        loop_var = self.visit(stmt.loop_var)
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)

        let_body = self.visit(stmt.let_body)
        return PureForStmt(
            args=args,
            values=values,
            loop_var=loop_var,
            extent=extent,
            body=body,
            let_vars=let_vars,
            let_body=let_body,
        )


def update_type(node: Union[Function, IRModule]):
    updater = TypeUpdater()
    return updater.visit(node)
