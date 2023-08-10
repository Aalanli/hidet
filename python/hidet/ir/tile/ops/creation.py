from typing import Union, Optional
from hidet.ir.type import BaseType
from hidet.ir.expr import Var
from hidet.ir.tile.type import tile_type, void_layout, TileLayout
from hidet.ir.tile.expr import TileOp


class Arange(TileOp):
    def __init__(self, begin: int, end: int, layout: Optional[TileLayout] = None):
        super().__init__(args=[], attrs={"begin": begin, "end": end, "layout": layout})

    def infer_type(self) -> BaseType:
        from hidet.ir.dtypes import int32

        extent = self.attrs["end"] - self.attrs["begin"]
        layout = self.attrs["layout"]
        if layout is None:
            layout = void_layout([extent])
        return tile_type(type_=int32, shape=[extent], layout=layout)


def arange(begin: int, end: int):
    return Arange(begin, end).make_call()
