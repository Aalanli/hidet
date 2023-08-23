from typing import List, Dict, Optional, Type
from collections import defaultdict
from hidet.ir.expr import Var, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.type import TileLayout
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store
from hidet.ir.tile.ops import Construct, Assign, convert_layout, ConvertLayout
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.convert_tile_expr_to_let import convert_to_let
from hidet.transforms.tile_generic.pattern_transform import apply_transforms, PatternTransform, TilePattern, Pattern
from hidet.transforms.tile_generic.dead_code_elimination import DeadCodeEliminationRewriter
from hidet.transforms.tile_generic.utils.usage_analyzer import VarUsage, UsageAnalyzer
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


class ChangeForArgLayoutRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.usages: Dict[Var, VarUsage] = dict()

    def anchor_priority(self, op: Type[TileOp]):
        pass

    def visit_Function(self, func: Function):
        usage_analyzer = UsageAnalyzer()
        usage_analyzer.visit(func)
        self.usages = usage_analyzer.usages
        updated_func = super().visit_Function(func)
        if updated_func is not func:
            updated_func = canonicalize_to_ssa(updated_func)
        return updated_func

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg in stmt.args:
            if not isinstance(arg.type, TileType):
                continue
            layout = arg.type.layout

            usage: VarUsage = self.usages[arg]

            # the mapping from the layout to the list of tile operators that require the arg to have the layout
            layout2anchor: Dict[TileLayout, List[TileOp]] = defaultdict(list)

            for let_usage in usage.call_op_let_usages():
                op = let_usage.bind_value.op
                if isinstance(op, ConvertLayout):
                    bind_var = let_usage.bind_var
                    for cvt_usage in self.usages[bind_var].call_op_let_usages():
                        layout2anchor[op.layout].append(cvt_usage.bind_value.op)
                else:
                    layout2anchor[layout].append(op)

            # find the layout that has the anchor with highest priority
            def anchor2priority(anchor_op):
                anchor_list = [

                ]

        return super().visit_PureForStmt(stmt)



class RemoveLayoutConvertPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_transforms(
            func, [
                IdentityConvertLayoutTransform(),
                ConvertConstructLayoutTransform(),
                DeadCodeEliminationRewriter()
            ]
        )


def remove_layout_convert_pass() -> TileFunctionPass:
    return RemoveLayoutConvertPass()
