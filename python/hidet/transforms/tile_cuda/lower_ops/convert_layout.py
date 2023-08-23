from typing import List, Union

from hidet.ir.expr import Expr, var, left_shift
from hidet.ir.mapping import auto_map
from hidet.ir.mapping import spatial_map
from hidet.ir.primitives.cuda import shfl_down_sync, shfl_up_sync, threadIdx
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.utils import prod, is_power_of_two, log_two
from .registry import TileOpImpl, register_impl
from .buffer import Buffer


@register_impl(ConvertLayout)
class ConvertLayoutImpl(TileOpImpl):
    def implement(self, op: TileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.is_distributed() and dst.is_distributed():
            # handle the cases where the conversion can be done efficiently
            if src.is_block() and dst.is_block() and src.layout == dst.layout:
                return src
            elif src.is_block() and dst.is_flatten_block() and src.layout == dst.flatten_block_layout.parent:
                raise NotImplementedError()
            elif src.is_flatten_block() and dst.is_block() and src.flatten_block_layout.parent == dst.layout:
                raise NotImplementedError()
            elif src.is_flatten_block() and dst.is_flatten_block() and src.layout == dst.layout:
                raise NotImplementedError()
            else:
                # use shared memory to do the conversion from a general distributed layout to another
                smem_shape = [s + 1 for s in src.shape]  # add one extra dimension to avoid bank conflict
                smem = self.alloc_shared_buffer(src.dtype, smem_shape, 'cvt_smem')

                # src to smem
                def f_apply(local_indices, global_indices, not_duplicated):
                    with self.if_then(not_duplicated):
                        self.buffer_store(smem.var, global_indices, value=src[local_indices])

                self.iterate_dist_buffer_and_apply(src, f_apply)

                # sync
                self.sync_threads()

                # smem to dst
                def f_compute(local_indices, global_indices, not_duplicated):
                    return smem[global_indices]

                self.iterate_dist_buffer_and_compute(dst, f_compute)

                self.sync_threads()

        elif src.is_distributed() and dst.is_shared():
            def f_apply(local_indices, global_indices, not_duplicated):
                with self.if_then(not_duplicated):
                    self.buffer_store(dst.var, global_indices, value=src[local_indices])

            self.iterate_dist_buffer_and_apply(src, f_apply)
            self.sync_threads()
        elif src.is_shared() and dst.is_distributed():

            def f_compute(local_indices, global_indices, not_duplicated):
                return src[global_indices]

            self.iterate_dist_buffer_and_compute(dst, f_compute)
            self.sync_threads()
        elif src.is_shared() and dst.is_shared():
            raise NotImplementedError()
        else:
            assert False
        return dst
