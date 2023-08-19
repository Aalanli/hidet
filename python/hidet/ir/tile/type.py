from __future__ import annotations

from typing import List, Optional, Union

from hidet.ir.tile.layout import TileLayout
from hidet.ir.type import BaseType, PointerType, DataType


class TileType(BaseType):
    def __init__(self, type_: Union[PointerType, DataType], shape: List[int], layout: Optional[TileLayout] = None):
        self.type: Union[PointerType, DataType] = type_
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout


def tile_type(type_, shape: List[int], layout=None):
    assert isinstance(type_, (PointerType, DataType))
    return TileType(type_, shape, layout)
