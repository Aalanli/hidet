from typing import List, Union

from hidet.ir.expr import Expr
from hidet.ir.tile.ops.creation import Full, Arange
from .buffer import Buffer
from .registry import TileOpImpl, register_impl


@register_impl(Arange)
class ArangeImpl(TileOpImpl):
    def implement(self, op: Arange, args: List[Union[Buffer, Expr]], output: Buffer):
        self.iterate_dist_buffer_and_compute(
            output, lambda local_indices, global_indices, not_duplicated: global_indices[0] + op.begin
        )


@register_impl(Full)
class FullImpl(TileOpImpl):
    def implement(self, op: Full, args: List[Union[Buffer, Expr]], output: Buffer):
        self.iterate_dist_buffer_and_compute(
            output, lambda local_indices, global_indices, not_duplicated: args[0]
        )
