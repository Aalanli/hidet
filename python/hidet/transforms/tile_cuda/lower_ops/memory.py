from typing import List, Union, Optional

from hidet.ir.expr import Expr, if_then_else, logical_and
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.layout import DistributedLayout, BlockLayout
from hidet.ir.tile.ops.memory import Load, Store
from hidet.ir.type import PointerType, DataType, void_p, sizeof, BaseType
from hidet.ir.dtypes import uint8, uint16, uint32, uint64
from .buffer import Buffer
from .registry import TileOpImpl, register_impl

def get_type_erased_dtype(ptr_type: PointerType) -> DataType:
    # get the type-erased data type of the loaded element
    assert isinstance(ptr_type, PointerType)
    nbits: int = sizeof(ptr_type.base_type) * 8
    nbits2dtype = {
        8: uint8,
        16: uint16,
        32: uint32,
        64: uint64
    }
    return nbits2dtype[nbits]


@register_impl(Load)
class LoadImpl(TileOpImpl):
    def implement(self, op: Load, args: List[Union[Buffer, Expr]], output: Buffer):
        ptr: Buffer = args[0]
        mask: Optional[Buffer] = args[1] if len(args) > 1 else None
        other: Optional[Buffer] = args[2] if len(args) > 2 else None
        layout = ptr.layout

        dtype: DataType = get_type_erased_dtype(ptr.dtype)

        # if isinstance(layout, BlockLayout):
        #     axis = len(layout.layout_shape) - 1     # the last axis is the innermost axis
        #     pass

        if isinstance(ptr.layout, DistributedLayout):
            if mask:
                assert mask.layout == ptr.layout
            if other:
                assert other.layout == ptr.layout

            def f_apply(local_indices, global_indices, not_duplicated):
                from hidet.ir.primitives.cuda.ldst import load

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
        ptr: Buffer = args[0]
        value: Buffer = args[1]
        mask: Optional[Buffer] = args[2] if len(args) > 2 else None

        dtype: DataType = get_type_erased_dtype(ptr.dtype)

        if isinstance(ptr.layout, DistributedLayout):
            assert value.layout == ptr.layout
            if mask:
                assert mask.layout == ptr.layout

            def f_apply(local_indices, global_indices, not_duplicated):
                from hidet.ir.primitives.cuda.ldst import store

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
