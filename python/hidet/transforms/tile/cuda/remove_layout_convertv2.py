# %%
from typing import List, Dict, Optional, Type, Any, Set, Union, Tuple
from collections import defaultdict
from hidet.ir.expr import Let, Var, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt, EvaluateStmt, Stmt
from hidet.ir.tile.type import TileLayout
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, StoreBaseOp, Load
from hidet.ir.tile.ops import Create, Assign, convert_layout, ConvertLayout, CastOp, DebugPrint, UnaryTileOp
from hidet.ir.tile.type import TileType
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.layout import BlockDotOperandLayout
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.analyzers import VarUsage, UsageAnalyzer, SourceAnalyzer, LetSource, ForArgSource, ForLetSource, FuncSource
from hidet.transforms.tile.generic.pattern_transform import apply_transforms, PatternTransform, TilePattern, Pattern
from hidet.transforms.tile.generic.dead_code_elimination import DeadCodeEliminationRewriter
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.utils import same_list

from hidet.lang.cuda import blockDim, blockIdx

from hidet.ir.tools.printer import IRPrinter
from hidet.utils.py import color, diff_lines, fuzzy_diff_text

class HighLightVarPrinter(IRPrinter):
    def __init__(self, vars: Set[Var]):
        super().__init__()
        self.should_highlight = vars

    def visit_Var(self, e: Var):
        if e in self.should_highlight:
            return color(super().visit_Var(e), fg='red')
        else:
            return super().visit_Var(e)

class VarScope:
    def __init__(self):
        self.scope: List[Dict[Var, Any]] = []
    
    def __enter__(self):
        self.scope.append(dict())
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.scope.pop()
    
    def contains(self, v: Var):
        if v is blockIdx.x:
            return True
        for scope in reversed(self.scope):
            if v in scope:
                return True
        return False
    
    def define(self, v: Var, value: Any):
        self.scope[-1][v] = value

class CheckStrandedVar(IRVisitor):
    def __init__(self):
        super().__init__()
        self._scope = VarScope()
        self.stranded_vars: Set[Var] = set()
    
    def open_scope(self):
        return self._scope
    
    def define(self, v: Var, value: Any):
        if self.is_defined(v):
            raise RuntimeError(f'{v} is already defined')
        self._scope.define(v, value)
    
    def is_defined(self, v: Var):
        return self._scope.contains(v)
    
    def visit_Function(self, func: Function):
        with self.open_scope():
            for param in func.params:
                self.define(param, param.type)
            self.visit(func.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        with self.open_scope():
            self.define(stmt.loop_var, stmt.loop_var.type)
            for arg in stmt.args:
                self.define(arg, arg.type)
            self.visit(stmt.body)
        with self.open_scope():
            for arg in stmt.let_vars:
                self.define(arg, arg.type)
            self.visit(stmt.let_body)
        
    def visit_LetStmt(self, stmt: LetStmt):
        for v in stmt.bind_vars:
            self.define(v, v.type)
        return super().visit_LetStmt(stmt)
    
    def visit_Var(self, e: Var):
        if not self.is_defined(e):
            self.stranded_vars.add(e)
        
    def visit_Create(self, e: Create):
        with self.open_scope():
            for v in e.axes:
                self.define(v, v.type)
            return super().visit_Create(e)

def check_for_stranded_var(func: Function): 
    checker = CheckStrandedVar()
    checker.visit(func)
    if len(checker.stranded_vars) > 0:
        print(HighLightVarPrinter(checker.stranded_vars).visit(func))
        raise RuntimeError('stranded vars:', list(checker.stranded_vars))


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
        cst: Create = self.get_tile_op(self.cst, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)

        updated_cst = Create(value=cst.value, shape=cst.shape, axes=cst.axes, layout=cvt.layout)
        return updated_cst.make_call()


class PushConvertLayoutForUnaryOpTransform(PatternTransform):
    """
    convert_layout(op(x), layout) -> op(convert_layout(x, layout))
    """

    def __init__(self):
        self.x = self.any_tile()
        self.y = self.unary(self.x)
        self.cvt = self.convert_layout(self.y)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        op: UnaryTileOp = self.get_tile_op(self.y, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        if isinstance(cvt.layout, BlockDotOperandLayout):
            # do not push convert_layout for BlockDotOperandLayout because it consumes too many registers
            return None
        return op.reforward(args=[convert_layout(x, cvt.layout)]).make_call()


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
        if isinstance(cvt.layout, BlockDotOperandLayout):
            # do not push convert_layout for BlockDotOperandLayout because it consumes too many registers
            return None
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


class CancanonicalSSAOperation:
    def __init__(self, op, args: List[Var], returns: List[Var]):
        assert all(isinstance(v, Var) for v in args)
        assert all(isinstance(v, Var) for v in returns)
        self.op = op
        self.args = args
        self.returns = returns

class ExtractAnchorOps(IRVisitor):
    def __init__(self):
        super().__init__()
        self.anchors: Set[CancanonicalSSAOperation] = set()
    
    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallTileOp):
                op: TileOp = bind_value.op
                if isinstance(op, (Load, Dot)):                    
                    self.anchors.add(CancanonicalSSAOperation(op, op.args, [bind_var]))
                
        super().visit_LetStmt(stmt)
    
    def visit_StoreBaseOp(self, e: StoreBaseOp):
        self.anchors.add(CancanonicalSSAOperation(e, e.args, []))

class ExtractYieldParent(IRVisitor):
    def __init__(self):
        super().__init__()
        self.yield_parent: Dict[YieldStmt, PureForStmt] = {}
        self.for_yield: Dict[PureForStmt, YieldStmt] = {}
        self.yield_to_return: Dict[Var, Var] = {}
        self.yield_to_arg: Dict[Var, Var] = {}
        self.for_arg_to_yield: Dict[Var, Var] = {}
        self.for_ret_to_yield: Dict[Var, Var] = {}
    
    def visit_YieldStmt(self, stmt: YieldStmt):
        self.yield_parent[stmt] = self.pure_for_stmts[-1]
        self.for_yield[self.pure_for_stmts[-1]] = stmt
        return super().visit_YieldStmt(stmt)
    
    def for_arg_to_yield_arg(self, for_stmt: PureForStmt, v: Var) -> Var:
        if v in self.for_arg_to_yield:
            return self.for_arg_to_yield[v]
        idx = for_stmt.args.index(v)
        yield_stmt = self.for_yield[for_stmt]
        varg = yield_stmt.values[idx]
        self.for_arg_to_yield[v] = varg
        return varg
    
    def for_return_to_yield_arg(self, for_stmt: PureForStmt, v: Var) -> Var:
        if v in self.for_ret_to_yield:
            return self.for_ret_to_yield[v]
        idx = for_stmt.let_vars.index(v)
        yield_stmt = self.for_yield[for_stmt]
        varg = yield_stmt.values[idx]
        self.for_ret_to_yield[v] = varg
        return varg
    
    def yield_arg_to_for_return(self, yield_stmt: YieldStmt, v: Var) -> Var:
        if v in self.yield_to_return:
            return self.yield_to_return[v]
        idx = yield_stmt.values.index(v)
        for_stmt = self.yield_parent[yield_stmt]
        vret = for_stmt.let_vars[idx]
        self.yield_to_return[v] = vret
        return vret
    
    def yield_arg_to_for_arg(self, yield_stmt: YieldStmt, v: Var) -> Var:
        if v in self.yield_to_arg:
            return self.yield_to_arg[v]
        idx = yield_stmt.values.index(v)
        for_stmt = self.yield_parent[yield_stmt]
        varg = for_stmt.args[idx]
        self.yield_to_arg[v] = varg
        return varg
    
    def for_arg_to_for_value(self, for_stmt: PureForStmt, v: Var) -> Var:
        idx = for_stmt.args.index(v)
        return for_stmt.values[idx]
    
    def parent_for(self, yield_stmt: YieldStmt) -> PureForStmt:
        return self.yield_parent[yield_stmt]
    
    def child_yield(self, for_stmt: PureForStmt) -> YieldStmt:
        return self.for_yield[for_stmt]

class PropagateLayoutAnalysis(IRVisitor):
    def __init__(self):
        super().__init__()
        self.layouts: Dict[Var, Set[TileLayout]] = {}
        self.source: Dict[Var, Union[LetSource, ForArgSource, ForLetSource, FuncSource]] = {}
        self.usage: Dict[Var, VarUsage] = {}
        self.yield_analysis: ExtractYieldParent = ExtractYieldParent()
    
    def is_anchor(self, op: TileOp) -> bool:
        return isinstance(op, (Load, StoreBaseOp, Dot))
    
    def is_terminal(self, op: TileOp) -> bool:
        return isinstance(op, (Create,))

    def verify_valid_arg(self, arg):
        assert isinstance(arg, Var), f"Not SSA form, arg {arg} is not Var"
        assert isinstance(arg.type, TileType)
        assert arg.type.layout is not None
    
    def visit_Function(self, func: Function):
        self.yield_analysis.visit_Function(func)
        source = SourceAnalyzer()
        source.visit_Function(func)
        self.source = source.source
        usage = UsageAnalyzer()
        usage.visit_Function(func)
        self.usage = usage.usages

        anchors = ExtractAnchorOps()
        anchors.visit_Function(func)
        anchors = anchors.anchors
        for op in anchors:
            print(op.op)
            if isinstance(op.op, Load):
                for arg in op.args:
                    print("load_arg", arg)
                    self.propagate_transient_parent(arg, arg.type.layout)
                print("load_ret", op.returns[0])
                self.propagate_transient_child(op.returns[0], op.returns[0].type.layout)
            elif isinstance(op.op, Dot):
                for arg in op.args:
                    
                    self.propagate_transient_parent(arg, arg.type.layout)
                self.propagate_transient_child(op.returns[0], op.returns[0].type.layout)
            elif isinstance(op.op, StoreBaseOp):
                for arg in op.args:
                    self.propagate_transient_parent(arg, arg.type.layout)    
    
    def propagate_transient_parent(self, var: Var, layout: TileLayout):
        assert isinstance(var, Var), f"var {var} does not have type Var, found {type(var)}"
        if var in self.layouts:
            if layout in self.layouts[var]:
                # already propagated through this chain
                return
            else:
                self.layouts[var].add(layout)
        else:
            self.layouts[var] = {layout}
        assert var in self.source, f"var {var} is not in source analysis"
        source = self.source[var]
        if isinstance(source, LetSource):
            assert isinstance(source.bind_value, CallTileOp)
            op: TileOp = source.bind_value.op
            if self.is_anchor(op):
                # stop propagating at anchor ops
                return
            if self.is_terminal(op):
                # stop propagating at creation ops, where there are non-ssa forms
                return
            for arg in op.args:
                assert isinstance(arg, Var), f"PropagateLayoutAnalysis only works on SSA form, found non-var arg {arg} of op {op}"
                self.propagate_transient_parent(arg, layout)
        elif isinstance(source, ForArgSource):
            # we are within for loop
            assert isinstance(source.value, Var), f"PropagateLayoutAnalysis only works on SSA form, found non-var arg {source.value} of for arg {var}"
            self.propagate_transient_parent(source.value, layout)
            cooresponding_yield_arg = self.yield_analysis.for_arg_to_yield_arg(source.stmt, source.arg)
            # simulate loop until fixed point
            self.propagate_transient_parent(cooresponding_yield_arg, layout)
            # propagate to for return
            self.propagate_transient_child(cooresponding_yield_arg, layout)
            cooresponding_let_var = source.stmt.let_vars[source.idx]
            self.propagate_transient_child(cooresponding_let_var, layout)
        elif isinstance(source, ForLetSource):
            let_var = source.let_var
            for_yield = self.yield_analysis.for_return_to_yield_arg(source.stmt, let_var)
            # for_arg = self.yield_analysis.yield_arg_to_for_arg(source.stmt, let_var)
            self.propagate_transient_parent(for_yield, layout)
            
    def propagate_transient_child(self, var: Var, layout: TileLayout):
        assert isinstance(var, Var), f"var {var} does not have type Var, found {type(var)}"
        if var in self.layouts:
            if layout in self.layouts[var]:
                # already propagated through this chain
                return
            else:
                self.layouts[var].add(layout)
        else:
            self.layouts[var] = {layout}
        if var not in self.usage:
            print(f"var {var} not found in usage")
            return
        # assert var in self.usage, f"var {var} is not in usage analysis"
        usage: VarUsage = self.usage[var]
        for lets in usage.let_usages:
            if self.is_anchor(lets.op):
                # stop propagating at anchor ops
                continue
            assert isinstance(lets.bind_var, Var), f"PropagateLayoutAnalysis only works on SSA form, found non-var arg {lets.bind_var} of op {lets.op}"
            self.propagate_transient_child(lets.bind_var, layout)
        for stmt_usage in usage.stmt_usages:
            stmt = stmt_usage.stmt
            if isinstance(stmt, EvaluateStmt):
                assert isinstance(stmt.expr, CallTileOp), f"PropagateLayoutAnalysis only works on SSA form, found non-call-expr {stmt.expr} of evaluate stmt {stmt}"
                # op: TileOp = stmt.expr.op
                # if self.is_anchor(op):
                #     continue
            elif isinstance(stmt, PureForStmt):
                assert var in stmt.values, f"for stmt does not contain var {var}"
                idx = stmt.values.index(var)
                for_arg = stmt.values[idx]
                # propagate inside for-loop
                self.propagate_transient_child(for_arg, layout)
                for_ret = stmt.let_vars[idx]
                # propagate to for return
                self.propagate_transient_child(for_ret, layout)
            
            elif isinstance(stmt, YieldStmt):                
                # assert all(isinstance(v, Var) for v in stmt.values), f"PropagateLayoutAnalysis only works on SSA form, found non-var arg {stmt.values} of yield stmt {stmt}"
                assert var in stmt.values, f"var {var} is not in yield stmt {stmt}"
                assoc_let = self.yield_analysis.yield_arg_to_for_return(stmt, var)
                print("assoc_let:", assoc_let)
                self.propagate_transient_child(assoc_let, layout)
                for_arg = self.yield_analysis.yield_arg_to_for_arg(stmt, var)
                print("for arg:", for_arg)
                self.propagate_transient_child(for_arg, layout)
                for_value = self.yield_analysis.for_arg_to_for_value(self.yield_analysis.parent_for(stmt), for_arg)
                print("for_value:", for_value)
                self.propagate_transient_parent(for_value, layout)
            else:
                raise NotImplementedError("PropagateLayoutAnalysis found unsupported stmt type, {}".format(type(stmt)))

