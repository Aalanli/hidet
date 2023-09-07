from __future__ import annotations

from typing import List, Optional, Union

from hidet.ir.tile.layout import TileLayout
from hidet.ir.stmt import DeclareScope as TileScope
from hidet.ir.type import BaseType, PointerType, DataType


class TileType(BaseType):
    def __init__(
        self,
        elem_type: Union[PointerType, DataType],
        shape: List[int],
        layout: Optional[TileLayout] = None,
        scope: Optional[Union[TileScope, str]] = None,

    ):
        self.type: Union[PointerType, DataType] = elem_type
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout
        self.scope: TileScope = TileScope.make(scope) if scope is not None else TileScope.Register

        from hidet.ir.tile.layout import SharedLayout
        assert not (self.scope.is_shared() ^ isinstance(self.layout, SharedLayout))


def tile_type(
    elem_type,  # Union[PointerType, DataType]
    shape: List[int],
    layout: Optional[TileLayout] = None,
    scope: Optional[Union[TileScope, str]] = None

):
    assert isinstance(elem_type, (PointerType, DataType))
    return TileType(elem_type, shape, layout, scope)
