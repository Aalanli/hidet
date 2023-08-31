from typing import List, Union, Optional

from hidet.ir.expr import Expr, logical_and
from hidet.ir.tile.layout import DistributedLayout, BlockLayout
from hidet.ir.tile.ops.memory import Load, Store
from hidet.ir.type import PointerType, DataType, void_p
from .registry import TileOpImpl, Buffer, register_impl
from .utils import get_type_erased_dtype


@register_impl(Load)
class LoadImpl(TileOpImpl):
    def implement(self, op: Load, args: List[Union[Buffer, Expr]], output: Buffer):
        from hidet.ir.primitives.cuda.ldst import load
        from hidet.ir.mapping import repeat_map

        ptr: Buffer = args[0]
        mask: Optional[Buffer] = args[1] if len(args) > 1 else None
        other: Optional[Buffer] = args[2] if len(args) > 2 else None
        layout = ptr.layout

        dtype: DataType = get_type_erased_dtype(ptr.dtype)

        if isinstance(layout, BlockLayout):
            axis = len(layout.layout_shape) - 1  # the last axis is the innermost axis
            vec_size = min(layout.size_per_thread[axis] * dtype.nbytes, 16) // dtype.nbytes
            if vec_size > 1 and mask is None:
                local_shape: List[int] = layout.calc_local_shape(output.shape)
                mapping_shape: List[int] = [d if i != axis else d // vec_size for i, d in enumerate(local_shape)]

                with self.for_mapping(repeat_map(mapping_shape)) as indices:
                    local_indices = [idx if dim != axis else idx * vec_size for dim, idx in enumerate(indices)]
                    dst_addrs = []
                    local_indices_iter = local_indices.copy()
                    for i in range(vec_size):
                        dst_addrs.append(~output.var[local_indices_iter])
                        local_indices_iter[axis] += 1
                    self.append(
                        load(dtype, addr=ptr[local_indices], dst_addrs=dst_addrs, space='global', nc_cache=True)
                    )
                return

        if isinstance(ptr.layout, DistributedLayout):
            if mask:
                assert mask.layout == ptr.layout
            if other:
                assert other.layout == ptr.layout

            def f_apply(local_indices, global_indices, not_duplicated):
                if mask is None:
                    self.append(load(dtype, addr=ptr[local_indices], dst_addrs=[~output.var[local_indices]]))
                else:
                    if other is None:
                        assert isinstance(ptr.dtype, PointerType)
                        value_type = ptr.dtype.base_type
                        if isinstance(value_type, PointerType):
                            other_value = void_p(0)
                        elif isinstance(value_type, DataType):
                            other_value = value_type.zero
                        else:
                            raise NotImplementedError()
                    else:
                        other_value = other[local_indices]
                    with self.if_then(mask[local_indices]):
                        self.append(load(dtype, addr=ptr[local_indices], dst_addrs=[~output.var[local_indices]]))
                    with self.otherwise():
                        self.buffer_store(output.var, local_indices, other_value)

            self.iterate_dist_buffer_and_apply(output, f_apply)
        else:
            raise NotImplementedError()


@register_impl(Store)
class StoreImpl(TileOpImpl):
    def implement(self, op: Store, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.mapping import repeat_map
        from hidet.ir.primitives.cuda.ldst import store
        ptr: Buffer = args[0]
        value: Buffer = args[1]
        mask: Optional[Buffer] = args[2] if len(args) > 2 else None

        dtype: DataType = get_type_erased_dtype(ptr.dtype)
        layout = ptr.layout

        if isinstance(layout, BlockLayout):
            axis = len(layout.layout_shape) - 1  # the last axis is the innermost axis
            vec_size = min(layout.size_per_thread[axis] * dtype.nbytes, 16) // dtype.nbytes
            # TODO: use u32 if vec_size > 4
            vec_size = min(vec_size, 4)
            if vec_size > 1 and mask is None:
                local_shape: List[int] = layout.calc_local_shape(ptr.shape)
                mapping_shape: List[int] = [d if i != axis else d // vec_size for i, d in enumerate(local_shape)]

                with self.for_mapping(repeat_map(mapping_shape)) as indices:
                    local_indices = [idx if dim != axis else idx * vec_size for dim, idx in enumerate(indices)]
                    src_addrs = []
                    local_indices_iter = local_indices.copy()
                    for i in range(vec_size):
                        src_addrs.append(~value.var[local_indices_iter])
                        local_indices_iter[axis] += 1
                    self.append(
                        store(dtype, addr=ptr[local_indices], src_addrs=src_addrs)
                    )
                return

        if isinstance(ptr.layout, DistributedLayout):
            assert value.layout == ptr.layout
            if mask:
                assert mask.layout == ptr.layout

            def f_apply(local_indices, global_indices, not_duplicated):

                if mask:
                    assert isinstance(mask, Buffer) and ptr.layout == mask.layout
                    mask_value = mask[local_indices]
                else:
                    mask_value = True

                with self.if_then(logical_and(not_duplicated, mask_value)):
                    # the same element in the tile might be stored in multiple threads, this if statement
                    # ensures that only one thread stores the value
                    self.append(store(dtype, addr=ptr.var[local_indices], src_addrs=[~value.var[local_indices]]))

            self.iterate_dist_buffer_and_apply(ptr, f_apply)
