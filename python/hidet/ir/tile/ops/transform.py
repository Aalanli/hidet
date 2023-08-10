from typing import Union, Optional, List
from hidet.ir.type import BaseType
from hidet.ir.expr import Var
from hidet.ir.tile.type import TileType, PointerType, BlockLayout, tile_type, block_layout, void_layout
from hidet.ir.tile.expr import TileOp


class Broadcast(TileOp):
    def __init__(self, x: Var, shape: List[int]):
        super().__init__(args=[x], attrs={"shape": shape})
        self.x: Var = x
        self.shape: List[int] = shape

    def infer_type(self) -> BaseType:
        xtype = self.x.type
        assert isinstance(xtype, TileType)
        layout = xtype.layout
        assert isinstance(layout, BlockLayout)
        return tile_type(
            type_=self.x.type,
            shape=self.shape,
            layout=block_layout(
                shape=self.shape,
                size_per_thread=layout.size_per_thread,
                thread_per_warp=layout.thread_per_warp,
                warps_per_block=layout.warps_per_block
            )
        )


class Reshape(TileOp):
    def __init__(self, x: Var, shape: List[int], layout: Optional[BlockLayout] = None):
        super().__init__(args=[x], attrs={"shape": shape, "layout": layout})
        self.x: Var = x
        self.shape: List[int] = shape
        self.layout: Optional[BlockLayout] = layout

    def infer_type(self) -> BaseType:
        xtype = self.x.type
        assert isinstance(xtype, TileType)
        if self.layout is None:
            layout = void_layout(self.shape)
        else:
            layout = self.layout
        return tile_type(
            type_=self.x.type,
            shape=self.shape,
            layout=layout
        )


class Full(TileOp):
    def __init__(self, value: Var, shape: List[int], layout: Optional[BlockLayout] = None):
        super().__init__(args=[value], attrs={"shape": shape, "layout": layout})
        self.value: Var = value
        self.shape: List[int] = shape
        self.layout: Optional[BlockLayout] = layout
