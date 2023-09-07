from typing import List, Dict, Type, Union, Callable, Tuple, Any, Optional
import inspect
from hidet.ir.type import DataType, PointerType, TensorPointerType, tensor_pointer_type, sizeof
from hidet.ir.expr import Expr, Var, tensor_var
from hidet.ir.stmt import AssignStmt, DeclareStmt, DeclareScope
from hidet.ir.mapping import spatial_map, repeat_map
from hidet.ir.tile.layout import TileLayout, BlockLayout, DistributedLayout, FlattenBlockLayout, SharedLayout
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tile.expr import TileOp
from hidet.ir.primitives.cuda import syncthreads, threadIdx
from hidet.ir.layout import row_major
from hidet.ir.stmt import Stmt
from hidet.ir.builders import StmtBuilder
from hidet.transforms.tile_generic.analyzers.value_analyzer import TensorInfo
from hidet.utils import prod


class Buffer:
    def __init__(
        self,
        buf_var: Var,
        dtype: Union[PointerType, DataType],
        shape: List[int],
        scope: TileScope,
        local_shape: List[int],
        layout: TileLayout,
        info=None,
    ):
        self.var: Var = buf_var
        self.dtype: Union[PointerType, DataType] = dtype
        self.shape: List[int] = shape
        self.scope: TileScope = scope
        self.local_shape: List[int] = local_shape
        self.layout: TileLayout = layout
        self.info: TensorInfo = info

        if scope.is_shared():
            assert isinstance(layout, SharedLayout)
        elif scope.is_register():
            assert not isinstance(layout, SharedLayout)

    def __getitem__(self, item):
        if self.scope == self.scope.Shared:
            local_indices, _ = self.layout.logical2local(item)
            return self.var[local_indices]
        elif self.scope == self.scope.Register:
            return self.var[item]
        else:
            raise NotImplementedError()

    @property
    def block_layout(self) -> BlockLayout:
        assert isinstance(self.layout, BlockLayout)
        return self.layout

    @property
    def shared_layout(self) -> SharedLayout:
        assert isinstance(self.layout, SharedLayout)
        return self.layout

    @property
    def flatten_block_layout(self) -> FlattenBlockLayout:
        assert isinstance(self.layout, FlattenBlockLayout)
        return self.layout

    def is_shared(self):
        return isinstance(self.layout, SharedLayout)

    def is_block(self):
        return isinstance(self.layout, BlockLayout)

    def is_flatten_block(self):
        return isinstance(self.layout, FlattenBlockLayout)


class TileOpImpl(StmtBuilder):
    def alloc_shared_buffer(
        self, dtype: Union[DataType, PointerType], shape: List[int], hint: str, data_layout=None
    ) -> Buffer:
        from hidet.ir.primitives.cuda.tile import alloc_shared

        buf_var = Var(hint=hint, type=tensor_pointer_type(dtype=dtype, shape=shape, layout=data_layout))
        self.declare(buf_var, init=alloc_shared(nbytes=sizeof(dtype) * prod(shape)))
        return Buffer(
            buf_var=buf_var,
            dtype=dtype,
            shape=shape,
            scope=TileScope.Shared,
            local_shape=shape,
            layout=SharedLayout(shape)
        )

    def sync_threads(self):
        self.append(syncthreads())

    def iterate_dist_buffer_and_apply(self, buf: Buffer, f_apply: Callable[[List[Expr], List[Expr], Expr], None]):
        assert buf.scope == TileScope.Register

        layout: TileLayout = buf.layout
        local_shape: List[int] = layout.local_shape()

        with self.for_grid(local_shape) as local_indices:
            global_indices, not_duplicated = layout.local2logical(local_indices, worker_index=threadIdx.x)
            f_apply(local_indices, global_indices, not_duplicated)

    def iterate_dist_buffer_and_compute(self, buf: Buffer, f_compute: Callable[[List[Expr], List[Expr], Expr], Expr]):
        def f_apply(local_indices, global_indices, not_duplicated):
            value = f_compute(local_indices, global_indices, not_duplicated)
            self.buffer_store(buf.var, indices=local_indices, value=value)

        self.iterate_dist_buffer_and_apply(buf, f_apply)

    def implement(self, op: TileOp, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
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
        parent_classes: Tuple = inspect.getmro(op_cls)
        for cls in parent_classes:
            if cls in _registered_implementations:
                _registered_implementations[op_cls] = _registered_implementations[cls]
                break
        else:
            raise RuntimeError(f"Cannot implement tile op:\n {op_cls.op_name()}")

    impl_cls = _registered_implementations[op_cls]
    impl = impl_cls()
    impl.implement(op, args, output)
    return impl.finish()
