from typing import List

from hidet.ir.expr import var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp
from hidet.ir.tile.ops import ExpandDims, ConvertLayout, convert_layout
from hidet.ir.tile.layout import BlockLayout, TileLayout, SharedLayout, FlattenBlockLayout
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.layout import row_major
from hidet.ir.tools import TypeInfer
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from .base import TileFunctionPass
from .convert_tile_expr_to_let import convert_to_let


class CanonicalizeConvertLayerRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self._type_infer = TypeInfer()

    def infer_type(self, e):
        if isinstance(e, TileOp):
            e = CallTileOp(e)
        return self._type_infer(e)

    def visit_ConvertLayout(self, e: ConvertLayout):
        x_type: TileType = self.infer_type(e.x)
        y_type: TileType = self.infer_type(e)
        la: TileLayout = x_type.layout
        lb: TileLayout = y_type.layout

        if (
            isinstance(la, SharedLayout)
            or isinstance(lb, SharedLayout)
            or (isinstance(la, BlockLayout) and isinstance(lb, FlattenBlockLayout) and lb.parent == la)
            or (isinstance(la, FlattenBlockLayout) and isinstance(lb, BlockLayout) and la.parent == lb)
        ):
            # do not need use another shared memory to relay the data
            return super().visit_ConvertLayout(e)
        else:
            lc: TileLayout = SharedLayout(row_major(*x_type.shape))
            return ConvertLayout(convert_layout(e.x, lc), e.layout)


class CanonicalizeConvertLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = CanonicalizeConvertLayerRewriter()
        return convert_to_let(rewriter.visit(func))

def canonicalize_convert_layout_pass() -> TileFunctionPass:
    return CanonicalizeConvertLayoutPass()
