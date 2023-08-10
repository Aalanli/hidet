from typing import List, Optional
from hidet.ir.node import Node
from hidet.ir.type import BaseType, PointerType, DataType


class TileLayout(Node):
    def __init__(self, shape: List[int]):
        self.shape: List[int] = shape


class VoidLayout(TileLayout):
    """the layout has not been specified"""
    def __init__(self, shape: List[int]):
        super().__init__(shape)


class SharedLayout(TileLayout):
    def __init__(self, shape: List[int]):
        super().__init__(shape)


class BlockLayout(TileLayout):
    def __init__(self, shape: List[int], size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
        super().__init__(shape)
        self.size_per_thread: List[int] = size_per_thread
        self.thread_per_warp: List[int] = thread_per_warp
        self.warps_per_block: List[int] = warps_per_block


class TileType:
    def __init__(self, type_: BaseType, shape: List[int], layout: TileLayout):
        self.type: BaseType = type_
        self.shape: List[int] = shape
        self.layout: TileLayout = layout


def block_layout(shape: List[int], size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
    return BlockLayout(shape, size_per_thread, thread_per_warp, warps_per_block)


def tile_type(type_: BaseType, shape: List[int], layout: Optional[TileLayout] = None):
    if layout is None:
        layout = void_layout(shape)
    return TileType(type_, shape, layout)


def void_layout(shape: List[int]):
    return VoidLayout(shape)
