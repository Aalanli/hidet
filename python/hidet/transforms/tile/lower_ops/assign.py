from typing import List, Union, Optional

from hidet.ir.expr import Expr, var, left_shift
from hidet.ir.mapping import auto_map
from hidet.ir.mapping import spatial_map
from hidet.ir.primitives.cuda import shfl_down_sync, shfl_up_sync, threadIdx
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.ops.assign import Assign
from hidet.utils import prod, is_power_of_two, log_two
from .registry import TileOpImpl, register_impl
from .buffer import Buffer


@register_impl(Assign)
class AssignImpl(TileOpImpl):
    def implement(self, op: Assign, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        dst: Buffer = args[0]
        src: Buffer = args[1]

        assert dst.layout == src.layout and src.is_distributed()

        def f_compute(local_indices, global_indices, not_duplicated):
            return src[local_indices]

        self.iterate_dist_buffer_and_compute(dst, f_compute)
