from hidet.ir.expr import Var, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store
from hidet.ir.tile.ops import Construct, Assign, convert_layout
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.convert_tile_expr_to_let import convert_to_let
from hidet.utils import same_list





class RemoveLayoutConvertPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        pass


def remove_layout_convert_pass() -> TileFunctionPass:
    return RemoveLayoutConvertPass()

