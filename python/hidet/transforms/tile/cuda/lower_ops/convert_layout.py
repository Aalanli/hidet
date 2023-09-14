from typing import List, Union

from hidet.ir.type import sizeof
from hidet.ir.expr import Expr, Var, cast
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tools import infer_type
from hidet.utils import prod
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(ConvertLayout)
class ConvertLayoutImpl(TileOpImpl):
    def request_smem_nbytes(self, op: TileOp) -> int:
        src: TileType = infer_type(op.args[0])
        dst: TileType = infer_type(op.make_call())

        if src.scope.is_register() and dst.scope.is_register():
            # handle the cases where the conversion can be done efficiently
            if src.layout == dst.layout:
                return 0
            else:
                # use shared memory to do the conversion from a general distributed layout to another
                smem_shape = [s + 1 for s in src.shape]  # add one extra dimension to avoid bank conflict
                return prod(smem_shape) * sizeof(src.type)
        elif src.scope.is_register() and dst.scope.is_shared():
            return prod(dst.layout.local_shape()) * sizeof(dst.type)
        elif src.scope.is_shared() and dst.scope.is_register():
            return 0
        elif src.scope.is_shared() and dst.scope.is_shared():
            raise NotImplementedError()
        else:
            assert False

    def implement(self, op: TileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.scope.is_register() and dst.scope.is_register():
            # handle the cases where the conversion can be done efficiently
            if src.layout == dst.layout:
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
                smem_ptr = cast(self.get_smem_ptr(op, prod(smem_shape) * sizeof(src.dtype)), ~src.dtype)
                smem = self.make_shared_buffer(src.dtype, smem_shape, 'cvt_smem', ptr=smem_ptr)

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

        elif src.scope.is_register() and dst.scope.is_shared():

            def f_apply(local_indices, global_indices, not_duplicated):
                with self.if_then(not_duplicated):
                    self.buffer_store(dst.var, global_indices, value=src[local_indices])

            smem_ptr = self.get_smem_ptr(op, nbytes=prod(dst.layout.local_shape()) * sizeof(dst.dtype))
            self.assign(dst.var, cast(smem_ptr, ~dst.dtype))
            self.iterate_dist_buffer_and_apply(src, f_apply)
            self.sync_threads()
        elif src.scope.is_shared() and dst.scope.is_register():

            def f_compute(local_indices, global_indices, not_duplicated):
                return src[global_indices]

            self.iterate_dist_buffer_and_compute(dst, f_compute)
        elif src.scope.is_shared() and dst.scope.is_shared():
            raise NotImplementedError()
        else:
            assert False
        return dst
