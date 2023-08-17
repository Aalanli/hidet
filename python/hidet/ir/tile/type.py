from __future__ import annotations
from typing import List, Optional, Union, Iterable, Tuple
from hidet.ir.node import Node
from hidet.ir.type import BaseType, PointerType, DataType
from hidet.ir.expr import Expr
from .expr import Attribute
from hidet.utils import same_list, iter_grid, prod, is_power_of_two, argmin


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
        self.layout_shape: List[int] = [
            a * b * c for a, b, c in zip(size_per_thread, thread_per_warp, warps_per_block)
        ]

    @staticmethod
    def from_shape(shape: List[int], num_warps: int) -> BlockLayout:
        num_elements = prod(shape)
        if not is_power_of_two(num_elements):
            raise ValueError(f"The tensor must have a power of 2 number of elements, got {num_elements}")
        size_per_thread = []
        thread_per_warp = []
        warps_per_block = []
        remaining_threads = 32
        remaining_warps = num_warps
        for extent in shape:
            size_per_thread.append(1)
            if extent <= remaining_threads:
                assert remaining_threads % extent == 0
                thread_per_warp.append(extent)
                warps_per_block.append(1)
                remaining_threads //= extent
            elif extent <= remaining_threads * remaining_warps:
                assert extent % remaining_threads == 0
                assert remaining_warps % (extent // remaining_threads) == 0
                allocated_warps = extent // remaining_threads
                thread_per_warp.append(remaining_threads)
                warps_per_block.append(allocated_warps)
                remaining_threads = 1
                remaining_warps //= allocated_warps
            else:
                thread_per_warp.append(remaining_threads)
                warps_per_block.append(remaining_warps)
                remaining_threads = 1
                remaining_warps = 1

        while remaining_threads > 1:
            assert remaining_threads % 2 == 0
            thread_per_warp[argmin(thread_per_warp)] *= 2
            remaining_threads //= 2

        while remaining_warps > 1:
            assert remaining_warps % 2 == 0
            warps_per_block[argmin(warps_per_block)] *= 2
            remaining_warps //= 2

        assert prod(warps_per_block) == num_warps
        assert prod(thread_per_warp) == 32

        return block_layout(size_per_thread, thread_per_warp, warps_per_block)

    def local_shape(self, shape: List[int]) -> List[int]:
        l_shape: List[int] = []
        for extent, size_per_thread, layout_extent in zip(shape, self.size_per_thread, self.layout_shape):
            assert extent % layout_extent == 0 or layout_extent % extent == 0
            if extent <= layout_extent:
                local_extent = size_per_thread
            else:
                local_extent = size_per_thread * (extent // layout_extent)
            l_shape.append(local_extent)
        return l_shape

    def local_to_global(self, local_indices: List[Expr], tid: Expr, global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean
        from hidet.ir.expr import logical_or
        from .utils import unflatten_indices
        assert len(global_shape) == len(self.layout_shape)

        lane_index = tid % 32
        warp_index = tid // 32
        lane_indices: List[Expr] = unflatten_indices(lane_index, self.thread_per_warp)
        warp_indices: List[Expr] = unflatten_indices(warp_index, self.warps_per_block)

        global_indices: List[Expr] = []
        # when the same element of the tensor is stored in multiple places, only one of them is not duplicated
        # (e.g., is_duplicated = false for the first element of the tensor)
        is_duplicated: Expr = boolean.false
        for i in range(len(global_shape)):
            local_index = local_indices[i]
            if global_shape[i] <= self.layout_shape[i]:
                layout_index = (
                    local_index
                    + lane_indices[i] * self.size_per_thread[i]
                    + warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
                global_index = layout_index % global_shape[i]
                is_duplicated = logical_or(is_duplicated, layout_index < global_shape[i])
            else:
                layout_index = (
                    local_index % self.size_per_thread[i]
                    + lane_indices[i] * self.size_per_thread[i]
                    + warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
                global_index = layout_index + self.layout_shape[i] * (local_index // self.size_per_thread[i])
            global_indices.append(global_index)
        return global_indices, is_duplicated

    def global_to_local(self, global_indices: List[Expr], tid: Expr, global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean
        from hidet.ir.expr import logical_and
        from .utils import unflatten_indices

        lane_index = tid % 32
        warp_index = tid // 32
        lane_indices: List[Expr] = unflatten_indices(lane_index, self.thread_per_warp)
        warp_indices: List[Expr] = unflatten_indices(warp_index, self.warps_per_block)

        local_shape = self.local_shape(global_shape)
        local_indices: List[Expr] = []
        is_valid: Expr = boolean.true   # whether the element is hold by the given tid
        is_repeated: Expr = boolean.false
        for i in range(len(global_shape)):
            global_index = global_indices[i]
            if global_shape[i] <= self.layout_shape[i]:
                layout_index = global_index
                local_index = (
                    layout_index
                    - lane_indices[i] * self.size_per_thread[i]
                    - warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
            else:
                layout_index = global_index % self.layout_shape[i]
                local_index = (
                    layout_index
                    - lane_indices[i] * self.size_per_thread[i]
                    - warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                ) + global_index // self.layout_shape[i] * self.size_per_thread[i]
            is_valid = logical_and(is_valid, 0 <= local_index, local_index < local_shape[i])
            local_indices.append(local_index)
        return local_indices, is_valid

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
