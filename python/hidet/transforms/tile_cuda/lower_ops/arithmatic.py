from typing import List, Union

from hidet.ir.expr import Expr, var, left_shift
from hidet.ir.mapping import auto_map
from hidet.ir.mapping import spatial_map
from hidet.ir.primitives.cuda import shfl_down_sync, shfl_up_sync, threadIdx
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.ops.arthimatic import UnaryTileOp, BinaryTileOp
from hidet.utils import prod, is_power_of_two, log_two
from .registry import TileOpImpl, register_impl
from .buffer import Buffer


@register_impl(UnaryTileOp)
class UnaryTileOpImpl(TileOpImpl):
    def implement(self, op: UnaryTileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]

        if src.is_distributed():
            self.iterate_dist_buffer_and_compute(
                output, lambda local_indices, global_indices, not_duplicated: op.apply_scalar(src[local_indices])
            )


@register_impl(BinaryTileOp)
class BinaryTileOpImpl(TileOpImpl):
    def implement(self, op: BinaryTileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        lhs: Buffer = args[0]
        rhs: Buffer = args[1]

        if lhs.is_distributed() and rhs.is_distributed() and lhs.layout == rhs.layout:
            self.iterate_dist_buffer_and_compute(
                output,
                lambda local_indices, global_indices, not_duplicated: op.apply_scalar(
                    lhs[local_indices], rhs[local_indices]
                ),
            )
        else:
            raise NotImplementedError()
