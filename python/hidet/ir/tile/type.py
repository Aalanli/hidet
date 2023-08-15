from typing import List, Optional, Union
from hidet.ir.node import Node
from hidet.ir.type import BaseType, PointerType, DataType
from .expr import Attribute
from hidet.utils import same_list


class TileLayout(Attribute):
    pass


class VoidLayout(TileLayout):
    """the layout has not been specified"""
    def __eq__(self, other):
        return isinstance(other, VoidLayout)


class SharedLayout(TileLayout):
    def __init__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, SharedLayout)


class BlockLayout(TileLayout):
    def __init__(
        self, size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]
    ):
        super().__init__()
        self.size_per_thread: List[int] = size_per_thread
        self.thread_per_warp: List[int] = thread_per_warp
        self.warps_per_block: List[int] = warps_per_block

    def __eq__(self, other):
        return (
            isinstance(other, BlockLayout) and
            same_list(self.size_per_thread, other.size_per_thread) and
            same_list(self.thread_per_warp, other.thread_per_warp) and
            same_list(self.warps_per_block, other.warps_per_block)
        )


class FlattenBlockLayout(TileLayout):
    def __init__(self, parent: BlockLayout, axis: int):
        super().__init__()
        self.parent: BlockLayout = parent
        self.axis: int = axis

    def __eq__(self, other):
        return isinstance(other, FlattenBlockLayout) and self.parent == other.parent and self.axis == other.axis


class TileType(BaseType):
    def __init__(self, type_: Union[PointerType, DataType], shape: List[int], layout: TileLayout):
        self.type: Union[PointerType, DataType] = type_
        self.shape: List[int] = shape
        self.layout: TileLayout = layout


def block_layout(size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
    return BlockLayout(size_per_thread, thread_per_warp, warps_per_block)


def flatten_block_layout(parent: BlockLayout, axis: int):
    return FlattenBlockLayout(parent, axis)


def tile_type(type_, shape: List[int], layout: Optional[TileLayout] = None):
    assert isinstance(type_, (PointerType, DataType))
    if layout is None:
        layout = void_layout()
    return TileType(type_, shape, layout)


def void_layout():
    return VoidLayout()
