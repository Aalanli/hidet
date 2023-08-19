from typing import List, Union, Optional

from hidet.ir.expr import Expr, Var, logical_and
from hidet.ir.tile.layout import DistributedLayout
from hidet.ir.tile.ops.debug import DebugPrint
from .buffer import Buffer
from .registry import TileOpImpl, register_impl


@register_impl(DebugPrint)
class DebugPrintImpl(TileOpImpl):
    def implement(self, op: DebugPrint, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.debug import printf
        from hidet.ir.dtypes import float32, int32

        buffer: Buffer = args[0]

        if buffer.is_distributed():
            assert isinstance(buffer.layout, DistributedLayout)
            layout: DistributedLayout = buffer.distributed_layout

            shape = buffer.shape
            with self.for_grid(buffer.shape) as indices:
                local_indices, is_valid = layout.global_to_local(indices, shape)
                dtype2fmt = {float32: '%.2f', int32: '%d'}
                with self.if_then(is_valid):
                    if len(shape) == 0:
                        self.append(printf(f'{dtype2fmt[buffer.dtype]}\n', buffer[local_indices]))
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
                        self.append(printf(f'{dtype2fmt[buffer.dtype]}', buffer[local_indices]))
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
        else:
            raise NotImplementedError()
