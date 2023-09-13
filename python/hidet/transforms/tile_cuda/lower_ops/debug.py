from typing import List, Union, Optional

from hidet.ir.type import DataType
from hidet.ir.expr import Expr, Var, logical_and
from hidet.ir.tile.layout import DistributedLayout
from hidet.ir.tile.ops.debug import DebugPrint
from .registry import TileOpImpl, Buffer, register_impl, TileLayout


@register_impl(DebugPrint)
class DebugPrintImpl(TileOpImpl):
    def implement(self, op: DebugPrint, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.debug import printf, format_string_from_dtype
        from hidet.ir.dtypes import float32, float16, int32, int64

        buffer: Buffer = args[0]

        assert buffer.scope.is_register()

        layout: TileLayout = buffer.layout

        shape = buffer.shape
        with self.for_grid(buffer.shape) as indices:
            local_indices, is_valid = layout.logical2local(indices)
            logical_indices, not_duplicated = layout.local2logical(local_indices)
            with self.if_then(logical_and(is_valid, not_duplicated)):
                value = buffer[local_indices]
                if isinstance(buffer.dtype, DataType) and buffer.dtype.is_float():
                    value = float32(value)
                    fmt = format_string_from_dtype(float32)
                else:
                    fmt = format_string_from_dtype(buffer.dtype)

                if len(shape) == 0:
                    self.append(printf(f'{fmt}\n', value))
                else:
                    if len(shape) == 1:
                        with self.if_then(indices[0] == 0):
                            self.append(printf('['))
                    else:
                        with self.if_then(logical_and(indices[-2] == 0, indices[-1] == 0)):
                            self.append(printf('[['))
                        with self.otherwise():
                            with self.if_then(indices[-1] == 0):
                                self.append(printf(' ['))
                    self.append(printf(f'{fmt}', value))
                    with self.if_then(indices[-1] == shape[-1] - 1):
                        if len(shape) == 1:
                            self.append(printf(']\n'))
                        else:
                            with self.if_then(indices[-2] != shape[-2] - 1):
                                self.append(printf(']\n'))
                            with self.otherwise():
                                self.append(printf(']]\n'))
                                if len(shape) > 2:
                                    self.append(printf('\n'))
                    with self.otherwise():
                        self.append(printf(', '))
            self.sync_threads()
