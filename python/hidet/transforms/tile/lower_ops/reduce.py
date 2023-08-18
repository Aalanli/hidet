from typing import List, Union
from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.builders import StmtBuilder
from hidet.ir.tile.ops.reduce import ReduceOp, ReduceKind
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.mapping import repeat_map
from .base import TileOpImpl, register_impl
from .buffer import Buffer


@register_impl(ReduceOp)
class ReduceOpImpl(TileOpImpl):

    def intra_thread_reduce(self, sb: StmtBuilder, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        layout: BlockLayout = src.block_layout
        shape: List[int] = src.shape

        spatial_shape: List[int] = layout.local_shape[:axis] + layout.local_shape[axis + 1:]
        default_value: Expr = rk.default_value(dst.dtype)

        with sb.for_grid(spatial_shape) as spatial_indices:
            dst_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
            sb += BufferStoreStmt(dst.var, dst_indices, default_value)
            if shape[axis] >= layout.size_per_thread[axis]:
                local_reduce_extent = layout.size_per_thread[axis]
            else:
                local_reduce_extent = shape[axis]
            with sb.for_range(local_reduce_extent) as reduce_index:
                src_indices = spatial_indices[:axis] + [reduce_index] + spatial_indices[axis:]
                sb += BufferStoreStmt(
                    dst.var,
                    dst_indices,
                    rk.combine(dst[dst_indices], src[src_indices])
                )

    def intra_warp_reduce(self, sb: StmtBuilder, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        layout: BlockLayout = src.block_layout
        shape: List[int] = src.shape

        spatial_shape: List[int] = layout.local_shape[:axis] + layout.local_shape[axis + 1:]
        default_value: Expr = rk.default_value(dst.dtype)

    def intra_block_reduce(self, sb: StmtBuilder, src: Buffer, dst: Buffer, axis: int):
        pass

    def implement(self, sb: StmtBuilder, op: ReduceOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.is_block() and dst.is_flatten_block() and dst.flatten_block_layout.parent == src:
            # in-thread reduce
            self.intra_thread_reduce(sb, src, dst, op.axis, op.kind)
        else:
            raise NotImplementedError()
