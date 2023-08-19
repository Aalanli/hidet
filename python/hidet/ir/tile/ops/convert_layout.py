from typing import Union, Optional, List
from hidet.ir.type import BaseType
from hidet.ir.expr import Var, Expr
from hidet.ir.tile.type import tile_type, TileLayout, TileType
from hidet.ir.tile.expr import TileOp
from hidet.utils import same_list


class ConvertLayout(TileOp):
    def __init__(self, x: Expr, layout: TileLayout):
        super().__init__(args=[x], attrs={"layout": layout})
        self.x: Expr = x
        self.layout: TileLayout = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        a_type = arg_types[0]
        assert isinstance(a_type, TileType)
        return tile_type(a_type.type, a_type.shape, self.layout)


def convert_layout(x: Expr, layout: TileLayout):
    return ConvertLayout(x, layout).make_call()
