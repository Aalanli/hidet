from typing import Optional, List, Union, Sequence, Callable
from hidet.ir.dtypes import int32
from hidet.ir.type import BaseType, DataType, data_type
from hidet.ir.expr import Var, convert, index_vars
from hidet.ir.tile.type import tile_type, TileLayout, TileScope
from hidet.ir.tile.expr import TileOp, Expr


class Create(TileOp):
    def __init__(self, value: Expr, shape: List[int], axes: List[Var], layout: Optional[TileLayout] = None):
        super().__init__(args=[value], attrs={"shape": shape, "axes": axes, "layout": layout})
        self.shape: List[int] = shape
        self.axes: List[Var] = axes
        self.value: Expr = value
        self.layout: Optional[TileLayout] = layout

    def __getitem__(self, actual_indices: List[Union[Expr, int]]) -> Expr:
        from hidet.ir.expr import convert
        from hidet.ir.tools import rewrite

        remap = {axis: convert(actual_index) for axis, actual_index in zip(self.axes, actual_indices)}
        return rewrite(self.value, remap)

    @staticmethod
    def from_compute(shape: List[int], f_compute: Callable[[List[Var]], Expr], layout: Optional[TileLayout] = None):
        axes: List[Var] = index_vars(num_vars=len(shape))
        value: Expr = f_compute(axes)
        return Create(value, shape, axes, layout)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        return tile_type(elem_type=x_type, shape=self.shape, layout=self.layout)


def arange(begin: int, end: int):
    return Create.from_compute(
        shape=[end - begin],
        f_compute=lambda axes: axes[0] + convert(begin)
    ).make_call()


def full(value: Union[Expr, int, bool, float], shape: Sequence[int]):
    return Create.from_compute(
        shape=list(shape),
        f_compute=lambda axes: convert(value)
    ).make_call()


def grid(shape: List[int], starts: List[Union[Expr, int]], strides: List[Union[Expr, int]]):
    from hidet.ir.expr import convert

    starts = [convert(start) for start in starts]
    strides = [convert(stride) for stride in strides]
    return Create.from_compute(
        shape=shape,
        f_compute=lambda axes: sum((axes[i] + starts[i]) * strides[i] for i in range(len(shape)))
    ).make_call()


def zeros(shape: List[int], dtype: Union[DataType, str] = 'float32'):
    dtype = data_type(dtype)
    return full(dtype.zero, shape)


def ones(shape: List[int], dtype: Union[DataType, str] = 'float32'):
    dtype = data_type(dtype)
    return full(dtype.one, shape)


def construct(shape: List[int], f_compute: Callable[[List[Var]], Expr]):
    return Create.from_compute(shape, f_compute).make_call()
