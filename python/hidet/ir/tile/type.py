from typing import List, Optional, Union
from hidet.ir.node import Node
from hidet.ir.type import BaseType, PointerType, DataType
from .expr import Attribute
from hidet.utils import same_list


class TileLayout(Attribute):
    def __init__(self, shape: List[int]):
        self.shape: List[int] = shape


class VoidLayout(TileLayout):
    """the layout has not been specified"""

    def __init__(self, shape: List[int]):
        super().__init__(shape)

    def __eq__(self, other):
        return isinstance(other, VoidLayout) and same_list(self.shape, other.shape)


class SharedLayout(TileLayout):
    def __init__(self, shape: List[int]):
        super().__init__(shape)

    def __eq__(self, other):
        return isinstance(other, SharedLayout) and same_list(self.shape, other.shape)


class BlockLayout(TileLayout):
    def __init__(
        self, shape: List[int], size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]
    ):
        super().__init__(shape)
        self.size_per_thread: List[int] = size_per_thread
        self.thread_per_warp: List[int] = thread_per_warp
        self.warps_per_block: List[int] = warps_per_block

    def __eq__(self, other):
        return (
            isinstance(other, BlockLayout) and same_list(self.shape, other.shape) and
            same_list(self.size_per_thread, other.size_per_thread) and
            same_list(self.thread_per_warp, other.thread_per_warp) and
            same_list(self.warps_per_block, other.warps_per_block)
        )


class TileType(BaseType):
    def __init__(self, type_: Union[PointerType, DataType], shape: List[int], layout: TileLayout):
        self.type: Union[PointerType, DataType] = type_
        self.shape: List[int] = shape
        self.layout: TileLayout = layout


def block_layout(shape: List[int], size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
    return BlockLayout(shape, size_per_thread, thread_per_warp, warps_per_block)


def tile_type(type_, shape: List[int], layout: Optional[TileLayout] = None):
    assert isinstance(type_, (PointerType, DataType))
    if layout is None:
        layout = void_layout(shape)
    return TileType(type_, shape, layout)


def void_layout(shape: List[int]):
    return VoidLayout(shape)
