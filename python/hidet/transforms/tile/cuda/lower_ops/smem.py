from typing import List, Union, Optional

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, if_then_else
from hidet.ir.primitives.cuda.cp_async import cp_async_commit_group, cp_async_wait_group
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.ops.smem import AllocTensor, InsertSliceAsync, AsyncCommitGroup, AsyncWait, ExtractSlice
from hidet.ir.tile.ops.smem import StoreShared, LoadShared
from hidet.ir.type import DataType, void_p
from hidet.transforms.tile import annotations
from hidet.utils import is_power_of_two
from .registry import TileOpImpl, Buffer, register_impl
from .utils import get_type_erased_dtype


@register_impl(AllocTensor)
class AllocTensorImpl(TileOpImpl):
    def implement(self, op: AllocTensor, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
        if annotations.global_offset not in op.annotations:
            op.annotations[annotations.global_offset] = 0
        self.assign(output.var, dynamic_shared_memory(op.annotations[annotations.global_offset], void_p))


@register_impl(InsertSliceAsync)
class InsertSliceAsyncImpl(TileOpImpl):
    def implement(self, op: InsertSliceAsync, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.cuda.cp_async import cp_async
        from hidet.ir.primitives.cuda.ldst import load

        ptr: Buffer = args[0]
        dst: Buffer = args[1]
        index: Expr = args[2]
        mask: Optional[Buffer] = args[3] if len(args) > 3 else None
        other: Optional[Buffer] = args[4] if len(args) > 4 else None
        insert_axis: int = op.axis
        layout = ptr.layout

        dtype: DataType = get_type_erased_dtype(ptr.dtype)

        if isinstance(layout, BlockLayout) and other is None:
            axis: int = len(layout.shape) - 1  # the last axis is the innermost axis

            # the number of elements loaded by each thread
            vec_size: int = layout.size_per_thread[axis]

            # we need to make sure that the ptr and optional mask/other are contiguous in the axis
            vec_size = min(vec_size, ptr.info.continuity[axis], ptr.info.divisibility[axis])
            if mask:
                vec_size = min(vec_size, mask.info.constancy[axis])
            if other:
                vec_size = min(vec_size, other.info.constancy[axis])

            # each thread can load at most 16 bytes (128 bits) at a time
            vec_size = min(vec_size, 16 // dtype.nbytes)

            # get the cp size per thread in the unit of bytes
            cp_size = vec_size * dtype.nbytes

            assert is_power_of_two(cp_size)

            if cp_size in [4, 8, 16]:
                # cp.async requires cp_size in [4, 8, 16] bytes
                local_shape: List[int] = layout.local_shape()
                assert local_shape[axis] % vec_size == 0
                local_shape = [e if i != axis else e // vec_size for i, e in enumerate(local_shape)]

                with self.for_grid(local_shape) as local_indices:
                    local_indices[axis] = local_indices[axis] * vec_size
                    logical_indices, not_duplicated = layout.local2logical(local_indices)
                    logical_indices = logical_indices[:insert_axis] + [index] + logical_indices[insert_axis:]
                    with self.if_then(not_duplicated):
                        self.append(
                            cp_async(
                                dst=~dst[logical_indices],
                                src=ptr[local_indices],
                                cp_size=cp_size,
                                src_size=if_then_else(mask[local_indices], cp_size, 0) if mask else None,
                            )
                        )
                self.assign(output.var, dst.var)
                return

        if mask:
            assert mask.layout == ptr.layout
        if other:
            assert other.layout == ptr.layout

        def f_apply(local_indices, global_indices, not_duplicated):
            global_indices = global_indices[:insert_axis] + [index] + global_indices[insert_axis:]
            with self.if_then(not_duplicated):
                if dtype.nbytes < 4 or (mask is not None and other is not None):
                    if mask is None:
                        self.append(load(dtype, addr=ptr[local_indices], dst_addrs=[~dst[global_indices]]))
                    else:
                        if other is None:
                            other_value: Expr = dtype.zero
                        else:
                            other_value: Expr = other[local_indices]
                        with self.if_then(mask[local_indices]):
                            self.append(load(dtype, addr=ptr[local_indices], dst_addrs=[~dst[global_indices]]))
                        with self.otherwise():
                            self.buffer_store(dst, global_indices, other_value)
                else:
                    if mask is not None:
                        src_size = if_then_else(mask[local_indices], dtype.nbytes, 0)
                    else:
                        src_size = None
                    self.append(
                        cp_async(
                            dst=~dst[global_indices], src=ptr[local_indices], cp_size=dtype.nbytes, src_size=src_size
                        )
                    )

        self.iterate_dist_buffer_and_apply(ptr, f_apply)
        self.assign(output.var, dst.var)


@register_impl(AsyncCommitGroup)
class AsyncCommitGroupImpl(TileOpImpl):
    def implement(self, op: AsyncCommitGroup, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(cp_async_commit_group())


@register_impl(AsyncWait)
class AsyncWaitImpl(TileOpImpl):
    def implement(self, op: AsyncWait, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(cp_async_wait_group(op.n))
        self.append(syncthreads())


@register_impl(ExtractSlice)
class ExtractSliceImpl(TileOpImpl):
    def implement(self, op: ExtractSlice, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        index: Expr = args[1]
        if op.extent == 1:
            indices: List[Expr] = [int32(0) for _ in range(len(output.shape))]
            indices = indices[: op.axis] + [index] + indices[op.axis :]
        else:
            indices: List[Expr] = [int32(0) for _ in range(len(output.shape))]
            indices[op.axis] = index
        self.assign(output.var, ~src[indices])


@register_impl(LoadShared)
class LoadSharedImpl(TileOpImpl):
    def implement(self, op: LoadShared, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        assert src.scope.is_shared() and output.scope.is_register()

        def f_compute(local_indices, global_indices, not_duplicated):
            return src[global_indices]

        self.iterate_dist_buffer_and_compute(output, f_compute)


@register_impl(StoreShared)
class StoreSharedImpl(TileOpImpl):
    def implement(self, op: StoreShared, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        dst: Buffer = args[1]

        assert src.scope.is_register() and dst.scope.is_shared()

        def f_apply(local_indices, global_indices, not_duplicated):
            with self.if_then(not_duplicated):
                self.buffer_store(dst, global_indices, src[local_indices])

        self.iterate_dist_buffer_and_apply(src, f_apply)
        self.assign(output.var, dst.var)
