import contextlib
from typing import Dict, List, Optional

from hidet.ir.expr import Expr, Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LaunchKernelStmt, BlackBoxStmt
from hidet.ir.stmt import Stmt, DeclareStmt, SeqStmt, AssignStmt, EvaluateStmt, BufferStoreStmt, LetStmt, ForStmt
from hidet.ir.stmt import WhileStmt, ForMappingStmt, BreakStmt, ContinueStmt, IfStmt, ReturnStmt, AssertStmt, AsmStmt
from hidet.ir.tile.stmt import PureForStmt, PureYieldStmt
from hidet.ir.tools import TypeInfer, collect
from hidet.transforms.base import TileFunctionPass


def flatten_stmt_seq(seq: List[Stmt]) -> List[Stmt]:
    flattened = []
    for stmt in seq:
        if isinstance(stmt, SeqStmt):
            flattened.extend(flatten_stmt_seq(list(stmt.seq)))
        else:
            flattened.append(stmt)
    return flattened


class CanonicalizeToSSARewriter(IRRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.var2value: Dict[Var, Expr] = {}
        self.stmts: List[List[Stmt]] = []
        self.type_infer = TypeInfer()

    def visit_and_wrap(self, stmt: Stmt):
        self.stmts.append([])
        self.visit(stmt)
        seq: List[Stmt] = self.stmts.pop()
        seq = flatten_stmt_seq(seq)

        if len(seq) == 0:
            return SeqStmt([])
        elif len(seq) == 1:
            return seq[0]

        body = seq.pop()

        if isinstance(body, LetStmt) and body.body is None:
            body.body = SeqStmt([])

        if isinstance(body, PureForStmt) and body.let_body is None:
            body.let_body = SeqStmt([])

        while seq:
            s = seq.pop()
            if isinstance(s, PureForStmt) and s.let_body is None:
                s.let_body = body
                body = s
            elif isinstance(s, LetStmt) and s.body is None:
                if isinstance(body, LetStmt):
                    body = LetStmt(s.bind_vars + body.bind_vars, s.bind_values + body.bind_values, body.body)
                else:
                    body = LetStmt(s.bind_vars, s.bind_values, body)
            else:
                if isinstance(body, SeqStmt):
                    body = SeqStmt([s] + list(body.seq))
                else:
                    body = SeqStmt([s, body])
        return body

    def append(self, stmt: Stmt):
        self.stmts[-1].append(stmt)

    def visit_Function(self, func: Function):
        body = self.visit_and_wrap(func.body)
        return Function(func.name, func.params, body, func.ret_type, func.kind, func.attrs)

    def visit_Var(self, e: Var):
        value = self.var2value.get(e, None)
        if value is None:
            return e  # global variables or function parameters
        else:
            return value

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt.init is not None:
            bind_var = Var(stmt.var.hint, stmt.var.type)
            bind_value = self.visit(stmt.init)
            self.var2value[stmt.var] = bind_var
            self.append(LetStmt(bind_var, bind_value, body=None))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        self.append(super().visit_EvaluateStmt(stmt))

    def visit_AssignStmt(self, stmt: AssignStmt):
        updated_value = self.visit(stmt.value)
        v = Var(stmt.var.hint, self.type_infer(updated_value))  # create a new variable for the updated value
        self.var2value[stmt.var] = v
        self.append(LetStmt(v, updated_value, body=None))

    def visit_LetStmt(self, stmt: LetStmt):
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            updated_value = self.visit(bind_value)
            self.var2value[bind_var] = updated_value
            bind_values.append(updated_value)
        self.append(LetStmt(stmt.bind_vars, bind_values, self.visit_and_wrap(stmt.body)))

    def visit_ForStmt(self, stmt: ForStmt):
        assignments: List[AssignStmt] = collect(stmt.body, AssignStmt)

        modified_vars: List[Var] = []
        args: List[Var] = []
        values: List[Expr] = []
        for assign_stmt in assignments:
            assigned_var = assign_stmt.var
            value: Optional[Expr] = self.var2value.get(assign_stmt.var, None)
            if value is not None:
                arg = Var(assigned_var.hint, assigned_var.type)
                args.append(arg)
                values.append(value)
                self.var2value[assigned_var] = arg
                modified_vars.append(assigned_var)
            else:
                # not found in parent scopes, so it is a variable defined in the loop body
                pass

        loop_var: Var = stmt.loop_var
        extent: Expr = self.visit(stmt.extent)
        body: Stmt = self.visit_and_wrap(SeqStmt([stmt.body, PureYieldStmt(modified_vars)]))
        returns: List[Var] = [Var(arg.hint, arg.type) for arg in args]

        for modified_var, ret in zip(modified_vars, returns):
            self.var2value[modified_var] = ret
        self.append(
            PureForStmt(
                args=args, values=values, loop_var=loop_var, extent=extent, body=body, let_vars=returns,
                let_body=None
            )
        )

    def visit_PureForStmt(self, e: PureForStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        for s in stmt.seq:
            self.visit(s)

    def visit_PureYieldStmt(self, e: PureYieldStmt):
        self.append(super().visit_PureYieldStmt(e))

    def visit_IfStmt(self, stmt: IfStmt):
        raise NotImplementedError('tile-dialect does not support {} for now'.format(stmt.__class__.__name__))

    def visit_WhileStmt(self, stmt: WhileStmt):
        raise NotImplementedError('tile-dialect does not support {} for now'.format(stmt.__class__.__name__))

    def visit_BreakStmt(self, stmt: BreakStmt):
        raise NotImplementedError('tile-dialect does not support {} for now'.format(stmt.__class__.__name__))

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        raise NotImplementedError('tile-dialect does not support {} for now'.format(stmt.__class__.__name__))

    def visit_AssertStmt(self, stmt: AssertStmt):
        raise NotImplementedError('tile-dialect does not support {} for now'.format(stmt.__class__.__name__))

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        raise NotImplementedError('tile-dialect does not support {} for now'.format(stmt.__class__.__name__))

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        raise NotImplementedError('tile-dialect does not support {}'.format(stmt.__class__.__name__))

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        raise NotImplementedError('tile-dialect does not support {}'.format(stmt.__class__.__name__))

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        raise NotImplementedError('tile-dialect does not support {}'.format(stmt.__class__.__name__))

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        raise NotImplementedError('tile-dialect does not support {}'.format(stmt.__class__.__name__))

    def visit_AsmStmt(self, stmt: AsmStmt):
        raise NotImplementedError('tile-dialect does not support {}'.format(stmt.__class__.__name__))


class CanonicalizeToSSAPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = CanonicalizeToSSARewriter()
        return rewriter.visit(func)


def canonicalize_to_ssa_pass() -> TileFunctionPass:
    return CanonicalizeToSSAPass()
