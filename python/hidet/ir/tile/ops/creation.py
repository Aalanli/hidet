from typing import Optional, List
from hidet.ir.type import BaseType
from hidet.ir.tile.type import tile_type, void_layout, TileLayout
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
        if layout is None:
            layout = void_layout([extent])
        return tile_type(type_=int32, shape=[extent], layout=layout)


class Full(TileOp):
    def __init__(self, value: Expr, shape: List[int], layout: Optional[TileLayout] = None):
        super().__init__(args=[value], attrs={"shape": shape, "layout": layout})
        self.value: Expr = value
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return tile_type(type_=arg_types[0], shape=self.shape, layout=self.layout)


def arange(begin: int, end: int):
    return Arange(begin, end).make_call()


def full(value: Expr, shape: List[int]):
    return Full(value, shape).make_call()
