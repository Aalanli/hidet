from typing import List, Union

from hidet.ir.expr import Expr
from hidet.ir.tile.ops.creation import Full, Arange, Construct
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(Arange)
class ArangeImpl(TileOpImpl):
    def implement(self, op: Arange, args: List[Union[Buffer, Expr]], output: Buffer):
        self.iterate_dist_buffer_and_compute(
            output, lambda local_indices, global_indices, not_duplicated: global_indices[0] + op.begin
        )


@register_impl(Full)
class FullImpl(TileOpImpl):
    def implement(self, op: Full, args: List[Union[Buffer, Expr]], output: Buffer):
        self.iterate_dist_buffer_and_compute(output, lambda local_indices, global_indices, not_duplicated: args[0])


@register_impl(Construct)
class ConstructImpl(TileOpImpl):
    def implement(self, op: Construct, args: List[Union[Buffer, Expr]], output: Buffer):
        self.iterate_dist_buffer_and_compute(
            output, lambda local_indices, global_indices, not_duplicated: op[global_indices]
        )
