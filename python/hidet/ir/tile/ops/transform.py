from typing import Optional, List
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from hidet.ir.tile.type import TileType, BlockLayout, tile_type, block_layout, void_layout
from hidet.ir.tile.expr import TileOp


class Broadcast(TileOp):
    def __init__(self, x: Expr, shape: List[int]):
        super().__init__(args=[x], attrs={"shape": shape})
        self.x: Expr = x
        self.shape: List[int] = shape

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        xtype = arg_types[0]
        assert isinstance(xtype, TileType)
        layout = xtype.layout
        assert isinstance(layout, BlockLayout)
        return tile_type(
            type_=xtype.type,
            shape=self.shape,
            layout=block_layout(
                shape=self.shape,
                size_per_thread=layout.size_per_thread,
                thread_per_warp=layout.thread_per_warp,
                warps_per_block=layout.warps_per_block,
            ),
        )


class Reshape(TileOp):
    def __init__(self, x: Expr, shape: List[int], layout: Optional[BlockLayout] = None):
        super().__init__(args=[x], attrs={"shape": shape, "layout": layout})
        self.x: Expr = x
        self.shape: List[int] = shape
        self.layout: Optional[BlockLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        xtype = arg_types[0]
        assert isinstance(xtype, TileType)
        if self.layout is None:
            layout = void_layout(self.shape)
        else:
            layout = self.layout
        return tile_type(type_=xtype.type, shape=self.shape, layout=layout)


class Full(TileOp):
    def __init__(self, value: Expr, shape: List[int], layout: Optional[BlockLayout] = None):
        super().__init__(args=[value], attrs={"shape": shape, "layout": layout})
        self.value: Expr = value
        self.shape: List[int] = shape
        self.layout: Optional[BlockLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return tile_type(
            type_=arg_types[0],
            shape=self.shape,
            layout=self.layout
        )


def broadcast(x: Expr, shape: List[int]):
    return Broadcast(x, shape).make_call()


def reshape(x: Expr, shape: List[int]):
    return Reshape(x, shape).make_call()


def full(value: Expr, shape: List[int]):
    return Full(value, shape).make_call()
