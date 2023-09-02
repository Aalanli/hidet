from __future__ import annotations

from typing import List, Tuple, Optional

from hidet.ir.expr import Expr
from hidet.ir.layout import DataLayout, row_major
from hidet.utils import same_list, prod, is_power_of_two, argmin
from .expr import Attribute


class TileLayout(Attribute):
    pass


class SharedLayout(TileLayout):
    def __init__(self, shape: List[int], data_layout: Optional[DataLayout] = None):
        super().__init__()
        self.shape: List[int] = shape
        self.data_layout: DataLayout = data_layout if data_layout is not None else row_major(*shape)

    def __str__(self):
        return 'shared({})'.format(self.data_layout)

    def __eq__(self, other):
        # todo: compare data_layout
        assert isinstance(other, TileLayout)
        return isinstance(other, SharedLayout)

    def __hash__(self):
        return hash(self.data_layout)

    def calc_local_shape(self, shape: List[int]) -> List[int]:
        return shape


class DistributedLayout(TileLayout):
    def __init__(self, layout_shape: List[int], num_warps: int):
        super().__init__()
        self.layout_shape: List[int] = layout_shape
        self.num_warps: int = num_warps

    def calc_local_shape(self, shape: List[int]) -> List[int]:
        raise NotImplementedError()

    def local_to_global(self, local_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        """
        Map the local indices to global indices.

        The local indices are the indices for the buffer defined by each thread.
        The global indices are the logical indices for the whole buffer in the view of the whole block.

        Parameters
        ----------
        local_indices: List[Expr]
            The local indices.

        global_shape: List[int]
            The global shape of the buffer.

        Returns
        -------
        global_indices, not_duplicated: List[Expr], Expr
            - global_indices: The global indices corresponding to the given local indices.
            - not_duplicated: a boolean expression indicating whether the global indices are duplicated. A global
            indices is duplicated if the data is stored in multiple threads and the threadIdx.x that holding the data is
            not the one with the smallest threadIdx.x and smallest local indices (if the data is stored multiple times
            in the same thread).
        """
        raise NotImplementedError()

    def global_to_local(self, global_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        """
        Map the global indices to local indices.

        Parameters
        ----------
        global_indices: List[Expr]
            The global indices.

        global_shape: List[int]
            The global shape of the buffer.

        Returns
        -------
        local_indices, is_valid: List[Expr], Expr
            - local_indices: The local indices corresponding to the given global indices.
            - is_valid: a boolean expression indicating whether the current thread is holding the data.
        """
        raise NotImplementedError()


class BlockLayout(DistributedLayout):
    def __init__(self, size_per_thread: List[int], thread_per_warp: List[int], warps_per_block: List[int]):
        super().__init__(
            layout_shape=[a * b * c for a, b, c in zip(size_per_thread, thread_per_warp, warps_per_block)],
            num_warps=prod(warps_per_block),
        )
        self.size_per_thread: List[int] = size_per_thread
        self.thread_per_warp: List[int] = thread_per_warp
        self.warps_per_block: List[int] = warps_per_block

    def __str__(self):
        return 'block(size_per_thread={}, thread_per_warp={}, warps_per_block={})'.format(
            self.size_per_thread, self.thread_per_warp, self.warps_per_block
        )

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, BlockLayout)
            and same_list(self.size_per_thread, other.size_per_thread)
            and same_list(self.thread_per_warp, other.thread_per_warp)
            and same_list(self.warps_per_block, other.warps_per_block)
        )

    def __hash__(self):
        return hash((tuple(self.size_per_thread), tuple(self.thread_per_warp), tuple(self.warps_per_block), 'block'))

    @staticmethod
    def from_shape(shape: List[int], num_warps: int, size_per_thread: Optional[List[int]] = None) -> BlockLayout:
        if not is_power_of_two(prod(shape)):
            raise ValueError(f"The tensor must have a power of 2 number of elements, got {prod(shape)}")
        if size_per_thread is not None and not is_power_of_two(prod(size_per_thread)):
            raise ValueError(f"size_per_thread must have a power of 2 number of elements, got {prod(size_per_thread)}")
        if size_per_thread is None:
            size_per_thread = [1] * len(shape)
        if len(size_per_thread) != len(shape):
            raise ValueError(f"size_per_thread must have the same length as shape, got {size_per_thread}")
        shape = [max(extent // size, 1) for extent, size in zip(shape, size_per_thread)]
        thread_per_warp = []
        warps_per_block = []
        remaining_threads = 32
        remaining_warps = num_warps
        for extent in reversed(shape):  # from innermost to outermost
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

        thread_per_warp = list(reversed(thread_per_warp))
        warps_per_block = list(reversed(warps_per_block))

        assert prod(warps_per_block) == num_warps
        assert prod(thread_per_warp) == 32

        return BlockLayout(size_per_thread, thread_per_warp, warps_per_block)

    def warp_indices(self) -> List[Expr]:
        from hidet.ir.primitives.cuda import threadIdx
        from .utils import unflatten_indices

        return unflatten_indices(threadIdx.x // 32, self.warps_per_block)

    def lane_indices(self) -> List[Expr]:
        from hidet.ir.primitives.cuda import threadIdx
        from .utils import unflatten_indices

        return unflatten_indices(threadIdx.x % 32, self.thread_per_warp)

    def calc_local_shape(self, shape: List[int]) -> List[int]:
        l_shape: List[int] = []
        for extent, size_per_thread, layout_extent in zip(shape, self.size_per_thread, self.layout_shape):
            assert extent % layout_extent == 0 or layout_extent % extent == 0
            if extent <= layout_extent:
                local_extent = size_per_thread
            else:
                local_extent = size_per_thread * (extent // layout_extent)
            l_shape.append(local_extent)
        return l_shape

    def local_to_global(self, local_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean
        from hidet.ir.expr import logical_and

        assert len(global_shape) == len(self.layout_shape)

        lane_indices: List[Expr] = self.lane_indices()
        warp_indices: List[Expr] = self.warp_indices()

        global_indices: List[Expr] = []
        # when the same element of the tensor is stored in multiple places, only one of them is not duplicated
        # (e.g., not_duplicated = true for the first element of the tensor)
        not_duplicated: Expr = boolean.true
        for i in range(len(global_shape)):
            local_index = local_indices[i]
            if global_shape[i] <= self.layout_shape[i]:
                layout_index = (
                    local_index
                    + lane_indices[i] * self.size_per_thread[i]
                    + warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
                global_index = layout_index % global_shape[i]
                not_duplicated = logical_and(not_duplicated, layout_index < global_shape[i])
            else:
                layout_index = (
                    local_index % self.size_per_thread[i]
                    + lane_indices[i] * self.size_per_thread[i]
                    + warp_indices[i] * self.size_per_thread[i] * self.thread_per_warp[i]
                )
                global_index = layout_index + self.layout_shape[i] * (local_index // self.size_per_thread[i])
            global_indices.append(global_index)
        return global_indices, not_duplicated

    def global_to_local(self, global_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean
        from hidet.ir.expr import logical_and

        lane_indices: List[Expr] = self.lane_indices()
        warp_indices: List[Expr] = self.warp_indices()

        local_shape = self.calc_local_shape(global_shape)
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


class FlattenBlockLayout(DistributedLayout):
    def __init__(self, parent: BlockLayout, axis: int):
        super().__init__(
            layout_shape=parent.layout_shape[:axis] + parent.layout_shape[axis + 1 :], num_warps=parent.num_warps
        )
        self.parent: BlockLayout = parent
        self.axis: int = axis

    def __str__(self):
        return 'flatten_block(parent={}, axis={})'.format(self.parent, self.axis)

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, FlattenBlockLayout) and self.parent == other.parent and self.axis == other.axis

    def __hash__(self):
        return hash((self.parent, self.axis, 'flatten_block'))

    def expanded_shape(self, shape: List[int]):
        return shape[: self.axis] + [1] + shape[self.axis :]

    def warp_indices(self) -> List[Expr]:
        return self.parent.warp_indices()

    def lane_indices(self) -> List[Expr]:
        return self.parent.lane_indices()

    def calc_local_shape(self, shape: List[int]) -> List[int]:
        return self.parent.calc_local_shape(self.expanded_shape(shape))

    def local_to_global(self, local_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        global_indices, not_duplicated = self.parent.local_to_global(local_indices, self.expanded_shape(global_shape))
        global_indices = global_indices[: self.axis] + global_indices[self.axis + 1 :]
        return global_indices, not_duplicated

    def global_to_local(self, global_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import int32

        global_indices = global_indices[: self.axis] + [int32.zero] + global_indices[self.axis :]
        return self.parent.global_to_local(global_indices, self.expanded_shape(global_shape))


class DotOperandLayout(DistributedLayout):
    pass


class BlockDotOperandLayout(DotOperandLayout):
    def __init__(self, parent: BlockLayout, op_idx: int):
        super().__init__(layout_shape=[], num_warps=parent.num_warps)  # initialize later
        self.parent: BlockLayout = parent
        self.op_idx: int = op_idx
        self.axis: int = op_idx

        if op_idx == 0:
            self.layout_shape = [parent.layout_shape[0], 1]
        else:
            self.layout_shape = [1, parent.layout_shape[1]]

    def __str__(self):
        return 'block_dot_operand(parent={}, op_idx={})'.format(self.parent, self.op_idx)

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, BlockDotOperandLayout) and self.parent == other.parent and self.op_idx == other.op_idx

    def __hash__(self):
        return hash((self.parent, self.op_idx, 'block_dot_operand'))

    def calc_local_shape(self, shape: List[int]) -> List[int]:
        assert len(shape) == 2
        if shape[self.axis] >= self.layout_shape[self.axis]:
            assert shape[self.axis] % self.layout_shape[self.axis] == 0
            repeated: int = shape[self.axis] // self.layout_shape[self.axis]
        else:
            repeated: int = 1
        if self.axis == 0:
            return [self.parent.size_per_thread[self.axis] * repeated, shape[1]]
        else:
            return [shape[0], self.parent.size_per_thread[self.axis] * repeated]

    def local_to_global(self, local_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.dtypes import boolean

        local_index = local_indices[self.axis]

        axis = self.op_idx
        size_per_thread: List[int] = self.parent.size_per_thread
        thread_per_warp: List[int] = self.parent.thread_per_warp
        layout_shape: List[int] = self.layout_shape
        warp_indices: List[Expr] = self.parent.warp_indices()
        lane_indices: List[Expr] = self.parent.lane_indices()

        if global_shape[axis] >= layout_shape[axis]:
            repeat_idx = local_index // size_per_thread[axis]
            inner_idx = local_index % size_per_thread[axis]
            global_index = (
                inner_idx
                + lane_indices[axis] * size_per_thread[axis]
                + warp_indices[axis] * size_per_thread[axis] * thread_per_warp[axis]
                + repeat_idx * layout_shape[axis]
            )
            not_duplicated = boolean.true
        else:
            global_index = (
                local_index
                + lane_indices[axis] * size_per_thread[axis]
                + warp_indices[axis] * size_per_thread[axis] * thread_per_warp[axis]
            )
            not_duplicated = global_index < global_shape[axis]
            global_index = global_index % global_shape[axis]
        if self.axis == 0:
            return [global_index, local_indices[1]], not_duplicated
        else:
            return [local_indices[0], global_index], not_duplicated

    def global_to_local(self, global_indices: List[Expr], global_shape: List[int]) -> Tuple[List[Expr], Expr]:
        from hidet.ir.expr import logical_and

        axis = self.op_idx
        size_per_thread: List[int] = self.parent.size_per_thread
        thread_per_warp: List[int] = self.parent.thread_per_warp
        layout_shape: List[int] = self.layout_shape
        warp_indices: List[Expr] = self.parent.warp_indices()
        lane_indices: List[Expr] = self.parent.lane_indices()

        if global_shape[0] >= layout_shape[0]:
            repeat_idx = global_indices[0] // layout_shape[axis]
            layout_idx = global_indices[0] % layout_shape[axis]
            local_index_part0 = (
                layout_idx
                - lane_indices[axis] * size_per_thread[axis]
                - warp_indices[axis] * size_per_thread[axis] * thread_per_warp[axis]
            )
            is_valid = logical_and(0 <= local_index_part0, local_index_part0 < size_per_thread[axis])
            local_index_part1 = repeat_idx * size_per_thread[axis]
            local_index = local_index_part0 + local_index_part1
        else:
            layout_idx = global_indices[0]
            local_index = (
                layout_idx
                - lane_indices[axis] * size_per_thread[axis]
                - warp_indices[axis] * size_per_thread[axis] * thread_per_warp[axis]
            )
            is_valid = logical_and(0 <= local_index, local_index < size_per_thread[axis])

        if self.axis == 0:
            return [local_index, global_indices[1]], is_valid
        else:
            return [global_indices[0], local_index], is_valid
