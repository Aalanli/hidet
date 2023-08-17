from __future__ import annotations

from typing import List, Optional, Union

from hidet.ir.tile.layout import TileLayout, void_layout
from hidet.ir.type import BaseType, PointerType, DataType


class TileType(BaseType):
    def __init__(self, type_: Union[PointerType, DataType], shape: List[int], layout: TileLayout):
        self.type: Union[PointerType, DataType] = type_
        self.shape: List[int] = shape
        self.layout: TileLayout = layout



def tile_type(type_, shape: List[int], layout: Optional[TileLayout] = None):
    assert isinstance(type_, (PointerType, DataType))
    if layout is None:
        layout = void_layout()
    return TileType(type_, shape, layout)


