from typing import List, Union, Optional

from hidet.ir.expr import Expr, if_then_else, logical_and, cast
from hidet.ir.tile.expr import TileOp
from hidet.ir.mapping import repeat_map
from hidet.ir.tile.layout import DistributedLayout, BlockLayout
from hidet.ir.tile.ops.smem import AllocTensor, InsertSliceAsync, AsyncCommitGroup, AsyncWait, ExtractSlice
from hidet.ir.type import PointerType, DataType, void_p, sizeof, BaseType
from hidet.ir.primitives.cuda.cp_async import cp_async_commit_group, cp_async, cp_async_wait_group
from hidet.ir.dtypes import uint8, uint16, uint32, uint64, int32
from hidet.utils import prod
from .registry import TileOpImpl, Buffer, register_impl
from .utils import get_type_erased_dtype


@register_impl(AllocTensor)
class AllocTensorImpl(TileOpImpl):
    def implement(self, op: AllocTensor, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.cuda.tile import alloc_shared
        nbytes: int = prod(op.shape) * sizeof(op.dtype)
        self.assign(output.var, alloc_shared(nbytes=nbytes))


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

        if isinstance(layout, BlockLayout):
            axis = len(layout.layout_shape) - 1  # the last axis is the innermost axis
            cp_size = min(layout.size_per_thread[axis] * dtype.nbytes, 16)
            vec_size = cp_size // dtype.nbytes
            if cp_size > 4 and mask is None:
                # todo: check the ptr contiguity and dst layout contiguity
                local_shape: List[int] = layout.calc_local_shape(ptr.shape)
                mapping_shape: List[int] = [d if i != axis else d // vec_size for i, d in enumerate(local_shape)]

                with self.for_mapping(repeat_map(mapping_shape)) as indices:
                    local_indices = [idx if dim != axis else idx * vec_size for dim, idx in enumerate(indices)]
                    global_indices, not_duplicated = layout.local_to_global(local_indices, global_shape=ptr.shape)
                    global_indices = global_indices[:insert_axis] + [index] + global_indices[insert_axis:]
                    with self.if_then(not_duplicated):
                        self.append(
                            cp_async(
                                dst=~dst[global_indices],
                                src=ptr[local_indices],
                                cp_size=cp_size,
                            )
                        )
                self.assign(output.var, dst.var)
                return

        if isinstance(ptr.layout, DistributedLayout):
            if mask:
                assert mask.layout == ptr.layout
            if other:
                assert other.layout == ptr.layout

            def f_apply(local_indices, global_indices, not_duplicated):
                if mask is None:
                    with self.if_then(not_duplicated):
                        if dtype.nbytes < 4:
                            self.append(load(dtype, addr=ptr[local_indices], dst_addrs=[~dst[global_indices]]))
                        else:
                            self.append(
                                cp_async(dst=~dst[global_indices], src=ptr[local_indices], cp_size=dtype.nbytes)
                            )
                else:
                    raise NotImplementedError()

            self.iterate_dist_buffer_and_apply(ptr, f_apply)
            self.assign(output.var, dst.var)
        else:
            raise NotImplementedError()


@register_impl(AsyncCommitGroup)
class AsyncCommitGroupImpl(TileOpImpl):
    def implement(self, op: AsyncCommitGroup, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(cp_async_commit_group())


@register_impl(AsyncWait)
class AsyncWaitImpl(TileOpImpl):
    def implement(self, op: AsyncWait, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(cp_async_wait_group(op.n))


@register_impl(ExtractSlice)
class ExtractSliceImpl(TileOpImpl):
    def implement(self, op: ExtractSlice, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        index: Expr = args[1]
        indices: List[Expr] = [int32(0) for _ in range(len(output.shape))]
        indices = indices[:op.axis] + [index] + indices[op.axis:]
        self.assign(output.var, ~src[indices])
