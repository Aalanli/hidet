from typing import List, Union

from hidet.ir.expr import Expr
from hidet.ir.tile.ops.creation import Create
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(Create)
class ConstructImpl(TileOpImpl):
    def implement(self, op: Create, args: List[Union[Buffer, Expr]], output: Buffer):
        self.iterate_dist_buffer_and_compute(
            output, lambda local_indices, global_indices, not_duplicated: op[global_indices]
        )
