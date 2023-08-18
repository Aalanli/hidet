from __future__ import annotations

from typing import List, Tuple

from hidet.ir.expr import Expr
from hidet.ir.layout import DataLayout
from hidet.utils import same_list, prod, is_power_of_two, argmin
from .expr import Attribute


class TileLayout(Attribute):
    pass


class VoidLayout(TileLayout):
    """the layout has not been specified"""

    def __eq__(self, other):
        return isinstance(other, VoidLayout)


class SharedLayout(TileLayout):
    def __init__(self, data_layout: DataLayout):
        super().__init__()
        self.data_layout: DataLayout = data_layout

    def __eq__(self, other):
        # todo: compare data_layout
        return isinstance(other, SharedLayout)

    def local_shape(self, shape: List[int]) -> List[int]:
        return shape


class BlockLayout(TileLayout):
    def __init__(self, size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
        super().__init__()
        self.size_per_thread: List[int] = size_per_thread
        self.thread_per_warp: List[int] = thread_per_warp
        self.warps_per_block: List[int] = warps_per_block
        self.layout_shape: List[int] = [a * b * c for a, b, c in zip(size_per_thread, thread_per_warp, warps_per_block)]

    def __eq__(self, other):
        return (
            isinstance(other, BlockLayout)
            and same_list(self.size_per_thread, other.size_per_thread)
            and same_list(self.thread_per_warp, other.thread_per_warp)
            and same_list(self.warps_per_block, other.warps_per_block)
        )

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

    def warp_indices(self) -> List[Expr]:
        from hidet.ir.primitives.cuda import threadIdx
        from .utils import unflatten_indices
        return unflatten_indices(threadIdx.x // 32, self.warps_per_block)

    def lane_indices(self) -> List[Expr]:
        from hidet.ir.primitives.cuda import threadIdx
        from .utils import unflatten_indices
        return unflatten_indices(threadIdx.x % 32, self.thread_per_warp)

    def local_to_global(self, local_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean
        from hidet.ir.expr import logical_or

        assert len(global_shape) == len(self.layout_shape)

        lane_indices: List[Expr] = self.lane_indices()
        warp_indices: List[Expr] = self.warp_indices()

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

    def global_to_local(
        self, global_indices: List[Expr], global_shape: List[int]
    ) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean
        from hidet.ir.expr import logical_and

        lane_indices: List[Expr] = self.lane_indices()
        warp_indices: List[Expr] = self.warp_indices()

        local_shape = self.local_shape(global_shape)
        local_indices: List[Expr] = []
        is_valid: Expr = boolean.true  # whether the element is hold by the given tid
        for i in range(len(global_shape)):
            global_index = global_indices[i]
            if global_shape[i] <= self.layout_shape[i]:
                layout_index = global_index
                local_index = (
                    layout_index
                    - lane_indices[i] * self.size_per_thread[i]
                    - warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
                is_valid = logical_and(is_valid, 0 <= local_index, local_index < local_shape[i])
            else:
                layout_index = global_index % self.layout_shape[i]
                local_index_part0 = (
                    layout_index
                    - lane_indices[i] * self.size_per_thread[i]
                    - warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
                local_index_part1 = global_index // self.layout_shape[i] * self.size_per_thread[i]
                is_valid = logical_and(is_valid, 0 <= local_index_part0, local_index_part0 < self.size_per_thread[i])
                local_index = local_index_part0 + local_index_part1
            local_indices.append(local_index)
        return local_indices, is_valid


class FlattenBlockLayout(TileLayout):
    def __init__(self, parent: BlockLayout, axis: int):
        super().__init__()
        self.parent: BlockLayout = parent
        self.axis: int = axis

    def __eq__(self, other):
        return isinstance(other, FlattenBlockLayout) and self.parent == other.parent and self.axis == other.axis

    def expanded_shape(self, shape: List[int]):
        return shape[: self.axis] + [1] + shape[self.axis:]

    def local_shape(self, shape: List[int]) -> List[int]:
        return self.parent.local_shape(self.expanded_shape(shape))

    def warp_indices(self) -> List[Expr]:
        return self.parent.warp_indices()

    def lane_indices(self) -> List[Expr]:
        return self.parent.lane_indices()

    def local_to_global(self, local_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        global_indices, is_duplicated = self.parent.local_to_global(local_indices, self.expanded_shape(global_shape))
        global_indices = global_indices[: self.axis] + global_indices[self.axis + 1:]
        return global_indices, is_duplicated

    def global_to_local(self, global_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import int32

        global_indices = global_indices[: self.axis] + [int32.zero] + global_indices[self.axis:]
        return self.parent.global_to_local(global_indices, self.expanded_shape(global_shape))


def void_layout():
    return VoidLayout()


def block_layout(size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
    return BlockLayout(size_per_thread, thread_per_warp, warps_per_block)


def flatten_block_layout(parent: BlockLayout, axis: int):
    return FlattenBlockLayout(parent, axis)
