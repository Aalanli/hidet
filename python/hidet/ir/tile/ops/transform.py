from typing import Optional, List, Union
from hidet.ir.type import BaseType, DataType, PointerType, data_type
from hidet.ir.expr import Expr
from hidet.ir.tile.type import TileType, TileLayout, tile_type
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.expr import TileOp


class Broadcast(TileOp):
    def __init__(self, x: Expr, shape: List[int], layout: Optional[TileLayout] = None):
        super().__init__(args=[x], attrs={"shape": shape, "layout": layout})
        self.x: Expr = x
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        return tile_type(type_=x_type.type, shape=self.shape, layout=self.layout)


class Reshape(TileOp):
    def __init__(self, x: Expr, shape: List[int], layout: Optional[BlockLayout] = None):
        super().__init__(args=[x], attrs={"shape": shape, "layout": layout})
        self.x: Expr = x
        self.shape: List[int] = shape
        self.layout: Optional[BlockLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        return tile_type(type_=x_type.type, shape=self.shape, layout=self.layout)


class ExpandDims(TileOp):
    def __init__(self, x: Expr, axis: int, layout: Optional[BlockLayout] = None):
        super().__init__(args=[x], attrs={"axis": axis, "layout": layout})
        self.x: Expr = x
        self.axis: int = axis
        self.layout: Optional[BlockLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        x_shape = x_type.shape
        axis = self.axis if self.axis >= 0 else len(x_shape) + self.axis + 1
        y_shape = x_shape[:axis] + [1] + x_shape[axis:]
        return tile_type(type_=x_type.type, shape=y_shape, layout=self.layout)


class CastOp(TileOp):
    def __init__(self, x: Expr, dtype: Union[DataType, PointerType]):
        super().__init__(args=[x], attrs={"dtype": dtype})
        self.x: Expr = x
        self.dtype: Union[DataType, PointerType] = dtype

    @classmethod
    def op_name(cls):
        # we use CastOp as the class name to avoid conflict with hidet.ir.expr.Cast
        return 'cast'

    @property
    def var_name_hint(self):
        return 'cst'

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        return tile_type(type_=self.dtype, shape=x_type.shape, layout=x_type.layout)


def broadcast(x: Expr, shape: List[int]):
    return Broadcast(x, shape).make_call()


def reshape(x: Expr, shape: List[int]):
    return Reshape(x, shape).make_call()


def expand_dims(x: Expr, axis: int):
    return ExpandDims(x, axis).make_call()


def cast(x: Expr, dtype: Union[DataType, PointerType, str]):
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    return CastOp(x, dtype).make_call()
