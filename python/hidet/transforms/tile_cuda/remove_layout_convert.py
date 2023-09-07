from typing import List, Dict, Optional, Type
from collections import defaultdict
from hidet.ir.expr import Var, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt, EvaluateStmt, Stmt
from hidet.ir.tile.type import TileLayout
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store, Load
from hidet.ir.tile.ops import Construct, Assign, convert_layout, ConvertLayout, CastOp
from hidet.ir.tile.type import TileType
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.pattern_transform import apply_transforms, PatternTransform, TilePattern, Pattern
from hidet.transforms.tile_generic.dead_code_elimination import DeadCodeEliminationRewriter
from hidet.transforms.tile_generic.analyzers import VarUsage, UsageAnalyzer
from hidet.transforms.tile_generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.utils import same_list


class IdentityConvertLayoutTransform(PatternTransform):
    """
    convert_layout(tile with layout1, layout1) -> tile with layout1
    """

    def __init__(self):
        self.x = self.any_tile()
        self.cvt = self.convert_layout(self.x)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        if isinstance(x, Var) and isinstance(x.type, TileType) and x.type.layout == cvt.layout:
            return x
        else:
            return None


class FoldConvertLayoutTransform(PatternTransform):
    """
    convert_layout(convert_layout(x, layout1), layout2) -> convert_layout(x, layout2)
    """

    def __init__(self):
        self.x = self.any_tile()
        self.cvt1 = self.convert_layout(self.x)
        self.cvt2 = self.convert_layout(self.cvt1)

    def source(self) -> TilePattern:
        return self.cvt2

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        cvt2: ConvertLayout = self.get_tile_op(self.cvt2, matched, var2call)
        return ConvertLayout(x, cvt2.layout, cvt2.scope).make_call()


class ConvertConstructLayoutTransform(PatternTransform):
    """
    convert_layout(construct(..., layout1), layout2) -> construct(..., layout2)
    """

    def __init__(self):
        self.cst = self.construct()
        self.cvt = self.convert_layout(self.cst)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        cst: Construct = self.get_tile_op(self.cst, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)

        updated_cst = Construct(value=cst.value, shape=cst.shape, axes=cst.axes, layout=cvt.layout)
        return updated_cst.make_call()


class PushConvertLayoutForBinaryOpTransform(PatternTransform):
    """
    convert_layout(op(x, y), layout) -> op(convert_layout(x, layout), convert_layout(y, layout))
    """

    def __init__(self):
        self.x = self.any_tile()
        self.y = self.any_tile()
        self.z = self.binary(self.x, self.y)
        self.cvt = self.convert_layout(self.z)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        y = matched[self.y]
        op: BinaryTileOp = self.get_tile_op(self.z, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        return op.reforward(args=[convert_layout(x, cvt.layout), convert_layout(y, cvt.layout)]).make_call()


class FoldConvertLayoutBeforeAndAfterCast(PatternTransform):
    """
    convert_layout(cast(convert_layout(x, layout1)), layout2) -> cast(convert_layout(x, layout2))
    """

    def __init__(self):
        self.x = self.any_tile()
        self.cvt1 = self.convert_layout(self.x)
        self.cst = self.cast(self.cvt1)
        self.cvt2 = self.convert_layout(self.cst)

    def source(self) -> TilePattern:
        return self.cvt2

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        from hidet.ir.tile.ops import cast, convert_layout

        x = matched[self.x]
        cst: CastOp = self.get_tile_op(self.cst, matched, var2call)
        cvt2: ConvertLayout = self.get_tile_op(self.cvt2, matched, var2call)
        return cast(convert_layout(x, cvt2.layout), cst.dtype)


class ChangeForArgLayoutRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.usages: Dict[Var, VarUsage] = dict()

    def anchor_priority(self, op: Type[TileOp]):
        order = [Dot, Load, Store, ReduceOp, Broadcast, ExpandDims, ConvertLayout, BinaryTileOp, Arange, Full]
        for idx, cls in enumerate(order):
            if issubclass(op, cls):
                return len(order) - idx
        raise NotImplementedError(op)

    def stmt_priority(self, stmt: Type[Stmt]):
        if isinstance(stmt, PureForStmt):
            return 100
        elif isinstance(stmt, YieldStmt):
            return 90
        else:
            raise NotImplementedError()

    def visit_Function(self, func: Function):
        usage_analyzer = UsageAnalyzer()
        usage_analyzer.visit(func)
        self.usages = usage_analyzer.usages
        updated_func = super().visit_Function(func)
        if updated_func is not func:
            updated_func = canonicalize_to_ssa(updated_func)
        return updated_func

    def visit_PureForStmt(self, stmt: PureForStmt):
        arg2layout: Dict[Var, TileLayout] = {}
        for arg in stmt.args:
            if not isinstance(arg.type, TileType):
                continue
            layout = arg.type.layout

            usage: VarUsage = self.usages[arg]

            # the mapping from the layout to the list of tile operators that require the arg to have the layout
            layout2priority: Dict[TileLayout, int] = defaultdict(int)

            for let_usage in usage.call_op_let_usages():
                op = let_usage.op
                if isinstance(op, ConvertLayout):
                    bind_var = let_usage.bind_var
                    for cvt_usage in self.usages[bind_var].call_op_let_usages():
                        p = self.anchor_priority(type(cvt_usage.op))
                        layout2priority[op.layout] = max(layout2priority[op.layout], p)
                else:
                    layout2priority[layout] = max(layout2priority[layout], self.anchor_priority(type(op)))
            for stmt_usage in usage.stmt_usages:
                s = stmt_usage.stmt
                if isinstance(s, EvaluateStmt):
                    if isinstance(s.expr, CallTileOp):
                        layout2priority[layout] = max(layout2priority[layout], self.anchor_priority(type(s.expr.op)))
                    else:
                        raise NotImplementedError()
                else:
                    layout2priority[layout] = self.stmt_priority(type(s))
                    raise NotImplementedError()

            # find the layout that has the anchor with the highest priority
            best_layout = max(layout2priority.keys(), key=lambda l: layout2priority[l])
            if best_layout == layout:
                continue
            arg2layout[arg] = best_layout

        if len(arg2layout) == 0:
            return super().visit_PureForStmt(stmt)

        # update the layout
        args = []
        values = []
        let_vars = []
        for orig_arg, let_var, value in zip(stmt.args, stmt.let_vars, stmt.values):
            value = self.visit(value)
            if orig_arg in arg2layout:
                assert isinstance(orig_arg, Var) and isinstance(orig_arg.type, TileType)
                tp = orig_arg.type
                orig_layout = tp.layout
                args.append(Var(orig_arg.hint, TileType(tp.type, tp.shape, arg2layout[orig_arg])))
                let_vars.append(Var(let_var.hint, TileType(tp.type, tp.shape, arg2layout[orig_arg])))
                values.append(convert_layout(value, arg2layout[orig_arg]))
                self.memo[orig_arg] = convert_layout(args[-1], orig_layout)
                self.memo[let_var] = convert_layout(let_vars[-1], orig_layout)
            else:
                args.append(orig_arg)
                values.append(value)
                let_vars.append(let_var)

        loop_var = self.visit(stmt.loop_var)
        extent = self.visit(stmt.extent)
        self.pure_for_stmts.append(stmt)
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()
        let_body = self.visit(stmt.let_body)
        return PureForStmt(
            args=args, values=values, loop_var=loop_var, extent=extent, body=body, let_vars=let_vars, let_body=let_body
        )

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        yields = self.visit(stmt.values)
        updated_yields = []
        for arg, yield_value in zip(for_stmt.args, yields):
            if arg is not self.memo[arg]:
                call_cvt: CallTileOp = self.memo[arg]
                assert isinstance(call_cvt.op, ConvertLayout)
                cvt: ConvertLayout = call_cvt.op
                assert isinstance(cvt.x, Var)
                updated_arg = cvt.x
                assert isinstance(updated_arg, Var) and isinstance(updated_arg.type, TileType)
                updated_yields.append(convert_layout(yield_value, updated_arg.type.layout))
            else:
                updated_yields.append(yield_value)
        if same_list(updated_yields, yields):
            return stmt
        else:
            return YieldStmt(updated_yields)


class RemoveLayoutConvertPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        transforms = [
            ChangeForArgLayoutRewriter(),
            IdentityConvertLayoutTransform(),
            ConvertConstructLayoutTransform(),
            FoldConvertLayoutTransform(),
            PushConvertLayoutForBinaryOpTransform(),
            FoldConvertLayoutBeforeAndAfterCast(),
            DeadCodeEliminationRewriter(),
        ]
        while True:
            orig_func = func
            for transform in transforms:
                func_before = func
                func = transform(func)
                if func_before is not func:
                    func = canonicalize_to_ssa(func)
            if func is orig_func:
                break
        return func


def remove_layout_convert_pass() -> TileFunctionPass:
    return RemoveLayoutConvertPass()
