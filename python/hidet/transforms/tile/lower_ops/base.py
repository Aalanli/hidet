from typing import List, Dict, Type, Union
from hidet.ir.type import DataType, PointerType, TensorPointerType, tensor_pointer_type, sizeof
from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import AssignStmt, DeclareStmt
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.layout import SharedLayout
from hidet.ir.primitives.cuda import syncthreads
from hidet.ir.layout import row_major
from hidet.ir.stmt import Stmt
from hidet.ir.builders import StmtBuilder
from hidet.utils import prod
from .buffer import Buffer


class TileOpImpl(StmtBuilder):

    def alloc_shared_buffer(
        self,
        dtype: Union[DataType, PointerType],
        shape: List[int],
        hint: str,
        data_layout=None
    ) -> Buffer:
        from hidet.ir.primitives.cuda.tile import alloc_shared
        buf_var = Var(hint=hint, type=tensor_pointer_type(dtype=dtype, shape=shape))
        self.declare(buf_var, init=alloc_shared(nbytes=sizeof(dtype) * prod(shape)))
        return Buffer(
            buf_var=buf_var,
            dtype=dtype,
            shape=shape,
            local_shape=shape,
            layout=SharedLayout(data_layout if data_layout else row_major(*shape)),
        )

    def sync_threads(self):
        self.append(syncthreads())

    def implement(self, op: TileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        raise NotImplementedError()


_registered_implementations: Dict[Type[TileOp], Type[TileOpImpl]] = {}


def register_impl(op_cls: Type[TileOp]):
    def decorator(impl_cls: Type[TileOpImpl]):
        _registered_implementations[op_cls] = impl_cls
        return impl_cls

    return decorator


def implement_tile_op(op: TileOp, args: List[Buffer], output: Buffer) -> Stmt:
    op_cls = type(op)
    if op_cls not in _registered_implementations:
        raise RuntimeError(f"Cannot implement tile op {op_cls}")
    impl_cls = _registered_implementations[op_cls]
    impl = impl_cls()
    impl.implement(op, args, output)
    return impl.finish()
