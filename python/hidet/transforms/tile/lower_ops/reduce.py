from typing import List, Union
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, logical_not, var, left_shift
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.cuda import shfl_down_sync, shfl_up_sync, shfl_sync, threadIdx
from hidet.ir.primitives.cuda.tile import alloc_shared
from hidet.ir.tile.ops.reduce import ReduceOp, ReduceKind
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.mapping import spatial_map
from hidet.ir.mapping import repeat_map, auto_map
from hidet.utils import prod, is_power_of_two, log_two
from .base import TileOpImpl, register_impl
from .buffer import Buffer


@register_impl(ReduceOp)
class ReduceOpImpl(TileOpImpl):

    def intra_thread_reduce(self, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        local_shape: List[int] = src.local_shape
        layout: BlockLayout = src.block_layout
        shape: List[int] = src.shape

        spatial_shape: List[int] = local_shape[:axis] + local_shape[axis + 1:]
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
        local_shape: List[int] = src.local_shape
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

        spatial_shape: List[int] = local_shape[:axis] + local_shape[axis + 1:]
        with self.for_grid(spatial_shape) as spatial_indices:
            mask = 0xffffffff
            with self.for_range(num_rounds) as i:
                dst_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                origin_value = dst[dst_indices]
                neighbor_value = shfl_down_sync(mask, origin_value, left_shift(delta, i), width)
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
        # case 1: warps_per_block == 1
        #    we do not need to do inter-wrap reduce as there is only one warp
        # case 2: shape[axis] <= size_per_thread * thread_per_warp and warps_per_block > 1
        #    we do not need to do inter-warp reduce as each warp already has the final result in the first thread
        # case 3: shape[axis] > size_per_thread * thread_per_warp
        #    we follow the following steps:
        #    1) regs -> smem
        #    2) reduce over smem
        if (
            shape[axis] <= layout.size_per_thread[axis] * layout.thread_per_warp[axis]
            or layout.warps_per_block[axis] == 1
        ):
            # case 1 and 2
            return
        else:
            # case 3
            spatial_shape: List[int] = local_shape[:axis] + local_shape[axis + 1:]
            smem_shape: List[int] = shape[:axis] + [layout.warps_per_block[axis]] + shape[axis + 1:]
            smem_buf = self.alloc_shared_buffer(dst.dtype, shape=smem_shape, hint='reduce_{}'.format(rk.name))
            # 1) regs -> smem
            lane_id = threadIdx.x % 32
            warp_indices: List[Expr] = layout.warp_indices()
            with self.for_grid(spatial_shape) as spatial_indices:
                with self.if_then(lane_id == 0):
                    src_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                    dst_indices, not_duplicated = layout.local_to_global(src_indices, global_shape=shape)[0]
                    dst_indices[axis] = warp_indices[axis]
                    with self.if_then(not_duplicated):
                        self.buffer_store(smem_buf.var, dst_indices, src[src_indices])
            self.sync_threads()
            # 2) reduce over smem
            global_spatial_shape: List[int] = shape[:axis] + shape[axis + 1:]
            num_warps: int = prod(layout.warps_per_block)
            num_threads: int = num_warps * 32
            spatial_size: int = prod(global_spatial_shape)
            if num_threads < spatial_size:
                mapping = auto_map(*global_spatial_shape, workers=num_threads)
            else:
                mapping = spatial_map(*global_spatial_shape)
            default_value: Expr = rk.default_value(dst.dtype)
            reduction_extent = layout.warps_per_block[axis]
            with self.if_then(threadIdx.x < spatial_size):
                with self.for_mapping(mapping) as global_spatial_indices:
                    v = self.declare(var('acc', dst.dtype), default_value)
                    with self.for_range(reduction_extent) as k:
                        indices = global_spatial_indices[:axis] + [k] + global_spatial_indices[axis:]
                        self.assign(v, rk.combine(v, smem_buf[indices]))
                    indices = global_spatial_indices[:axis] + [0] + global_spatial_indices[axis + 1:]
                    self.buffer_store(dst.var, indices, v)
            self.sync_threads()
            # 3) smem -> regs
            with self.for_grid(spatial_shape) as spatial_indices:
                with self.if_then(lane_id == 0):
                    dst_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                    src_indices, _ = layout.local_to_global(dst_indices, global_shape=shape)[0]
                    src_indices[axis] = 0
                    self.buffer_store(dst.var, dst_indices, smem_buf[src_indices])

    def broadcast_back(self, src: Buffer, dst: Buffer, axis: int, rk: ReduceKind):
        # decompose layout_shape[axis]:
        # --                size_per_thread = 2
        # --------          thread_per_warp = 4
        # ----------------  warps_per_block = 2
        # broadcast the result in the first local element of the first thread to all the other threads
        # and their corresponding local elements
        # step1: use warp shuffle to broadcast the result to all the threads in the warp
        # step2: assign the result to all the local elements in the thread

        layout: BlockLayout = src.block_layout
        local_shape: List[int] = src.local_shape
        spatial_shape: List[int] = local_shape[:axis] + local_shape[axis + 1:]

        delta: int = prod(layout.thread_per_warp[axis + 1:])
        width: int = prod(layout.thread_per_warp[axis:])
        num_rounds: int = log_two(layout.thread_per_warp[axis])

        with self.for_grid(spatial_shape) as spatial_indices:
            # step 1
            mask = 0xffffffff
            with self.for_range(num_rounds) as i:
                indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                value = dst[indices]
                value = shfl_up_sync(mask, value, left_shift(delta, num_rounds - i - 1), width)
                self.buffer_store(dst.var, indices, value)

            # step 2
            with self.for_range(dst.local_shape[axis]) as i:
                from_indices = spatial_indices[:axis] + [0] + spatial_indices[axis:]
                to_indices = spatial_indices[:axis] + [i] + spatial_indices[axis:]
                self.buffer_store(dst.var, to_indices, dst[from_indices])

    def implement(self, op: ReduceOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.is_block() and dst.is_flatten_block() and dst.flatten_block_layout.parent == src.layout:
            # in-thread reduce
            self.intra_thread_reduce(src, dst, op.axis, op.kind)
            self.intra_warp_reduce(src, dst, op.axis, op.kind)
            self.intra_block_reduce(src, dst, op.axis, op.kind)
            self.broadcast_back(src, dst, op.axis, op.kind)
        else:
            raise NotImplementedError()
