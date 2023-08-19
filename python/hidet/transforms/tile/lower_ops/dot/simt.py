from typing import List, Union

from hidet.ir.expr import Expr, var, left_shift
from hidet.ir.mapping import auto_map
from hidet.ir.mapping import spatial_map
from hidet.ir.primitives.cuda import shfl_down_sync, shfl_up_sync, threadIdx
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.ops.dot import Dot
from hidet.utils import prod, is_power_of_two, log_two
from ..registry import TileOpImpl, register_impl
from ..buffer import Buffer


@register_impl(Dot)
class ReduceImpl(TileOpImpl):
    def implement(self, op: TileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        raise NotImplementedError()
