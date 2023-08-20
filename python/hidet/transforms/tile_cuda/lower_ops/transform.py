from typing import List, Union, Optional

from hidet.ir.expr import Expr, Var, logical_and
from hidet.ir.tile.layout import DistributedLayout
from hidet.ir.tile.ops.transform import ExpandDims, Broadcast
from .buffer import Buffer
from .registry import TileOpImpl, register_impl


@register_impl(ExpandDims)
class ExpandDimsImpl(TileOpImpl):
    def implement(self, op: ExpandDims, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.is_flatten_block() and dst.is_block() and src.flatten_block_layout.parent == dst.layout:
            assert src.flatten_block_layout.axis == op.axis

            def f_compute(local_indices, global_indices, not_duplicated):
                return src[local_indices]

            self.iterate_dist_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()


@register_impl(Broadcast)
class BroadcastImpl(TileOpImpl):
    def implement(self, op: Broadcast, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        dst: Buffer = output

        broadcast_dims = [i for i in range(len(dst.shape)) if dst.shape[i] != src.shape[i]]

        if src.is_distributed() and dst.is_distributed() and src.layout == dst.layout:

            def f_compute(local_indices, global_indices, not_duplicated):
                local_indices = [idx if i not in broadcast_dims else 0 for i, idx in enumerate(local_indices)]
                return src[local_indices]

            self.iterate_dist_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()
