from typing import List, Union
from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.cuda import shfl_down_sync, shfl_sync, threadIdx
from hidet.ir.primitives.cuda.tile import alloc_shared
from hidet.ir.tile.ops.reduce import ReduceOp, ReduceKind
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.mapping import spatial_map
from hidet.ir.mapping import repeat_map
from hidet.utils import prod, is_power_of_two, log_two
from .base import TileOpImpl, register_impl
from .buffer import Buffer


@register_impl(ReduceOp)
class ReduceOpImpl(TileOpImpl):

    def intra_thread_reduce(self, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        layout: BlockLayout = src.block_layout
        shape: List[int] = src.shape

        spatial_shape: List[int] = layout.local_shape[:axis] + layout.local_shape[axis + 1:]
        default_value: Expr = rk.default_value(dst.dtype)

        # decompose layout_shape[axis]:
        # --                size_per_thread = 2
        # --------          thread_per_warp = 4
        # ----------------  warps_per_block = 2
        #
        # there are three cases:
        # case 1: shape[axis] <= size_per_thread
        #    reduce over the first shape[axis] elements
        # case 2: size_per_thread < shape[axis] <= size_per_thread * thread_per_warp
        #    reduce over the whole size_per_thread elements
        # case 3: shape[axis] > size_per_thread * thread_per_warp
        #    reduce over all elements store in the thread along axis dimension
        if shape[axis] <= layout.size_per_thread[axis]:
            # case 1
            reduce_extent = shape[axis]
        elif shape[axis] <= layout.size_per_thread[axis] * layout.thread_per_warp[axis]:
            # case 2
            reduce_extent = layout.size_per_thread[axis]
        else:
            # case 3
            reduce_extent = layout.thread_per_warp[axis] * (shape[axis] // layout.layout_shape[axis])

        with self.for_grid(spatial_shape) as spatial_indices:
            dst_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
            self.buffer_store(dst.var, dst_indices, default_value)
            with self.for_range(reduce_extent) as reduce_index:
                src_indices = spatial_indices[:axis] + [reduce_index] + spatial_indices[axis:]
                self.buffer_store(
                    dst.var,
                    dst_indices,
                    rk.combine(dst[dst_indices], src[src_indices])
                )

    def intra_warp_reduce(self, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        layout: BlockLayout = src.block_layout
        shape: List[int] = src.shape

        # decompose layout_shape[axis]:
        # --                size_per_thread = 2
        # --------          thread_per_warp = 4
        # ----------------  warps_per_block = 2
        #
        # there are three cases:
        # case 1: shape[axis] <= size_per_thread (e.g., shape[axis] = 1, 2)
        #    we do not need to do intra-warp reduce as there is only one element
        # case 2: size_per_thread < shape[axis] <= size_per_thread * thread_per_warp (e.g., shape[axis] = 4, 8)
        #    we need to do the intra-thread reduce over size_per_thread * thread_per_warp / shape[axis] elements
        # case 3: shape[axis] > size_per_thread * thread_per_warp (e.g., shape[axis] = 16, 32, ...)
        #    we need to do the intra-thread reduce over thread_per_warp elements
        if shape[axis] <= layout.size_per_thread[axis]:
            # case 1
            num_elements: int = 1
        elif shape[axis] <= layout.size_per_thread[axis] * layout.thread_per_warp[axis]:
            # case 2
            num_elements: int = shape[axis] // layout.size_per_thread[axis]
        else:
            # case 3
            num_elements: int = layout.thread_per_warp[axis]
        assert is_power_of_two(num_elements)

        delta: int = prod(layout.thread_per_warp[axis + 1:])
        width: int = prod(layout.thread_per_warp[axis:])
        num_rounds: int = log_two(num_elements)

        spatial_shape: List[int] = layout.local_shape[:axis] + layout.local_shape[axis + 1:]
        with self.for_grid(spatial_shape) as spatial_indices:
            mask = 0xffffffff
            with self.for_range(num_rounds) as i:
                dst_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                origin_value = dst[dst_indices]
                neighbor_value = shfl_down_sync(mask, origin_value, delta << i, width)
                value = rk.combine(origin_value, neighbor_value)
                self.buffer_store(dst.var, dst_indices, value)

    def intra_block_reduce(self, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        layout: BlockLayout = src.block_layout
        shape: List[int] = src.shape

        # decompose layout_shape[axis]:
        # --                size_per_thread = 2
        # --------          thread_per_warp = 4
        # ----------------  warps_per_block = 2
        # there are two cases:
        # case 1: shape[axis] <= size_per_thread * thread_per_warp (e.g., shape[axis] = 1, 2, 4, 8)
        #    we do not need to do inter-warp reduce
        # case 2: shape[axis] > size_per_thread * thread_per_warp (e.g., shape[axis] = 16, 32, ...)
        #    we follow the following steps:
        #    1) regs -> smem
        #    2) reduce over smem
        #    3) smem -> regs
        if (
            shape[axis] <= layout.size_per_thread[axis] * layout.thread_per_warp[axis]
            or layout.warps_per_block[axis] == 1
        ):
            # case 1
            return

        # case 2
        spatial_shape: List[int] = layout.local_shape[:axis] + layout.local_shape[axis + 1:]
        smem_shape: List[int] = shape[:axis] + [layout.warps_per_block[axis]] + shape[axis + 1:]
        smem_buf = self.alloc_shared_buffer(dst.dtype, shape=smem_shape, hint='reduce_{}'.format(rk.name))
        # 1) regs -> smem
        lane_id = threadIdx.x % 32
        warp_indices: List[Expr] = layout.warp_indices()
        with self.for_grid(spatial_shape) as spatial_indices:
            with self.if_then(lane_id == 0):
                src_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                dst_indices = layout.local_to_global(src_indices, global_shape=shape)[0]
                dst_indices[axis] = warp_indices[axis]
                self.buffer_store(smem_buf.var, dst_indices, src[src_indices])
        self.sync_threads()
        # 2) reduce over smem
        global_spatial_shape: List[int] = shape[:axis] + shape[axis + 1:]
        if axis == len(shape) - 1:
            mapping = spatial_map(spatial_shape)
        else:
            pass
        # 3) smem -> regs

    def broadcast_back(self, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        pass

    def implement(self, op: ReduceOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.is_block() and dst.is_flatten_block() and dst.flatten_block_layout.parent == src:
            # in-thread reduce
            self.intra_thread_reduce(src, dst, op.axis, op.kind)
            self.intra_warp_reduce(dst, dst, op.axis, op.kind)
            self.intra_block_reduce(dst, dst, op.axis, op.kind)
            self.broadcast_back(dst, dst, op.axis, op.kind)
        else:
            raise NotImplementedError()
