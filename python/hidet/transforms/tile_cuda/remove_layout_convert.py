from typing import List, Dict, Optional
from hidet.ir.expr import Var, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store
from hidet.ir.tile.ops import Construct, Assign, convert_layout, ConvertLayout
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.convert_tile_expr_to_let import convert_to_let
from hidet.transforms.tile_generic.pattern_transform import apply_transforms, Transform, TilePattern, Pattern
from hidet.transforms.tile_generic.dead_code_elimination import DeadCodeEliminationRewriter
from hidet.utils import same_list


class ConvertConstructLayoutTransform(Transform):
    def __init__(self):
        self.cst = self.construct()
        self.cvt = self.convert_layout(self.cst)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[CallTileOp]:
        cst: Construct = self.get_tile_op(self.cst, matched)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched)

        updated_cst = Construct(value=cst.value, shape=cst.shape, axes=cst.axes, layout=cvt.layout)
        return updated_cst.make_call()


class RemoveLayoutConvertWithTransformsRewriter(IRRewriter):
    def __call__(self, func: Function) -> Function:
        transforms = [
            ConvertConstructLayoutTransform(),
        ]
        return apply_transforms(func, transforms)


class RemoveLayoutConvertPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_rewriter_list(func, [
            RemoveLayoutConvertWithTransformsRewriter(),
            DeadCodeEliminationRewriter()
        ])


def remove_layout_convert_pass() -> TileFunctionPass:
    return RemoveLayoutConvertPass()
