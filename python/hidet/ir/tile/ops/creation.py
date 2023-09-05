from typing import Optional, List, Union, Sequence, Callable
from hidet.ir.type import BaseType, DataType, data_type
from hidet.ir.expr import Var, convert, index_vars
from hidet.ir.tile.type import tile_type, TileLayout, TileScope
from hidet.ir.tile.expr import TileOp, Expr


class Arange(TileOp):
    def __init__(self, begin: int, end: int, layout: Optional[TileLayout] = None):
        super().__init__(args=[], attrs={"begin": begin, "end": end, "layout": layout})
        self.begin: int = begin
        self.end: int = end
        self.layout: Optional[TileLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        from hidet.ir.dtypes import int32

        extent = self.attrs["end"] - self.attrs["begin"]
        layout = self.attrs["layout"]
        return tile_type(elem_type=int32, shape=[extent], layout=layout)


class Full(TileOp):
    def __init__(self, value: Expr, shape: List[int], layout: Optional[TileLayout] = None):
        super().__init__(args=[value], attrs={"shape": shape, "layout": layout})
        self.value: Expr = value
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return tile_type(elem_type=arg_types[0], shape=self.shape, layout=self.layout)


class Construct(TileOp):
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
        return Construct(value, shape, axes, layout)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        return tile_type(elem_type=x_type, shape=self.shape, layout=self.layout)


def arange(begin: int, end: int):
    return Arange(begin, end).make_call()


def full(value: Union[Expr, int, bool, float], shape: Sequence[int]):
    shape = list(shape)
    value = convert(value)
    return Full(value, shape).make_call()


def zeros(shape: List[int], dtype: Union[DataType, str] = 'float32'):
    dtype = data_type(dtype)
    return full(dtype.zero, shape)


def ones(shape: List[int], dtype: Union[DataType, str] = 'float32'):
    dtype = data_type(dtype)
    return full(dtype.one, shape)


def construct(shape: List[int], f_compute: Callable[[List[Var]], Expr]):
    return Construct.from_compute(shape, f_compute).make_call()
