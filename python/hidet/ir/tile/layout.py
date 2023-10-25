from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import io
import itertools
from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Expr, logical_and, equal
from hidet.ir.layout import DataLayout, row_major
from hidet.utils import same_list, prod, is_power_of_two, argmin, argmax
from hidet.utils.vector import Vector


class TileLayout:
    def __str__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        return hash(str(self))

    def __mul__(self, other):
        assert isinstance(other, TileLayout)
        return ComposedLayout(outer=self, inner=other)

    def num_workers(self) -> int:
        raise NotImplementedError()

    def local_shape(self) -> List[int]:
        raise NotImplementedError()

    def logical_shape(self) -> List[int]:
        raise NotImplementedError()

    def local2logical(self, local_indices: List[Expr], worker_index: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        """
        ret: logical_indices, not_duplicated
        """
        raise NotImplementedError()

    def logical2local(
        self, logical_indices: List[Expr], worker_index: Optional[Expr] = None
    ) -> Tuple[List[Expr], Expr]:
        """
        ret: local_index, is_valid
        """
        raise NotImplementedError()

    def atom(
        self, *shape, workers: List[int], ranks: Optional[List[int]] = None, worker_ranks: Optional[List[int]] = None
    ):
        return ComposedLayout(outer=self, inner=atom(*shape, workers=workers, ranks=ranks, worker_ranks=worker_ranks))

    def spatial(self, *shape: int, ranks: Optional[List[int]] = None):
        return ComposedLayout(outer=self, inner=spatial(*shape, ranks=ranks))

    def repeat(self, *shape: int, ranks: Optional[List[int]] = None):
        return ComposedLayout(outer=self, inner=repeat(*shape, ranks=ranks))

    def visualize(self, verbose=False) -> str:
        shape = self.logical_shape()
        if len(shape) not in [1, 2]:
            raise ValueError('Cannot visualize layout with rank {} (shape={})'.format(len(shape), shape))
        grid: Dict[Tuple[int, ...], str] = {}
        for logical_indices in itertools.product(*map(range, shape)):
            workers: List[Tuple[int, List[int]]] = []
            for worker_index in range(self.num_workers()):
                local_indices, is_valid = self.logical2local(logical_indices, int32(worker_index))
                if is_valid:
                    workers.append((worker_index, [int(v) for v in local_indices]))
            if len(workers) == 0:
                r = '.'
            else:
                if verbose:
                    r = '{' + ', '.join('{}:{}'.format(a, b) for a, b in workers) + '}'
                else:
                    if len(workers) == 1:
                        r = str(workers[0][0])
                    else:
                        r = '{' + ', '.join([str(a) for a, b in workers]) + '}'
            grid[logical_indices] = r
        width = max(len(str(a)) for a in grid.values()) + 1
        fmt = '{:>' + str(width) + '}'
        f = io.StringIO()

        idx_width = max(len(str(d)) for d in self.logical_shape()) + 1
        idx_fmt = '{:>' + str(idx_width) + '}'

        # print the logical shape, num of workers, and local extent
        print('  logical shape: {}'.format(self.logical_shape()), file=f)
        print('    local shape: {}'.format(self.local_shape()), file=f)
        print(' num of workers: {}'.format(self.num_workers()), file=f)
        # print the first row of indices
        for j in range(shape[-1]):
            if j == 0:
                print(' ' * idx_width + ' |', file=f, end='')
            print(fmt.format(j), file=f, end='')
        print(file=f)
        # print the second row of separator
        for j in range(shape[-1]):
            if j == 0:
                sep = ' ' + '-' * (idx_width - 1) + ' +'
                print(sep, file=f, end='')
            sep = ' ' + '-' * (width - 1)
            print(sep, file=f, end='')
        print(file=f)
        # print each row of the layout
        for logical_indices in itertools.product(*map(range, shape)):
            if logical_indices[-1] == 0:
                print(idx_fmt.format(logical_indices[0]) + ' |', file=f, end='')
            print(fmt.format(grid[logical_indices]), file=f, end='')
            if logical_indices[-1] == shape[-1] - 1:
                print(file=f)
        return f.getvalue()


class AtomLayout(TileLayout):
    def __init__(
        self,
        shape: List[int],
        worker_shape: List[int],
        ranks: Optional[List[int]] = None,
        worker_ranks: Optional[List[int]] = None,
    ):
        self.shape: List[int] = shape
        self.worker_shape: List[int] = worker_shape
        self.ranks: List[int] = ranks if ranks is not None else list(range(len(shape)))
        self.worker_ranks: List[int] = worker_ranks if worker_ranks is not None else list(range(len(worker_shape)))
        self._local_shape: List[int] = [max(a // b, 1) for a, b in zip(shape, worker_shape)]

        assert all(a % b == 0 or b % a == 0 for a, b in zip(self.shape, self.worker_shape))

    def __str__(self):
        items = []
        items.append(', '.join(str(a) for a in self.shape))
        if same_list(self.shape, self.worker_shape):
            name = 'spatial'
        elif all(a == 1 for a in self.worker_shape):
            name = 'repeat'
        else:
            name = 'atom'
            items.append('workers={}'.format(self.worker_shape))
        if self.ranks != list(range(len(self.shape))):
            items.append('ranks={}'.format(self.ranks))
        if self.worker_ranks != list(range(len(self.worker_shape))):
            items.append('worker_ranks={}'.format(self.worker_ranks))
        return name + '(' + ', '.join(items) + ')'

    def num_workers(self) -> int:
        return prod(self.worker_shape)

    def local_shape(self) -> List[int]:
        return self._local_shape

    def logical_shape(self) -> List[int]:
        return self.shape

    def local2logical(self, local_indices: List[Expr], worker_index: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        from hidet.ir.utils.index_transform import index_deserialize

        if worker_index is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker_index = threadIdx.x

        worker_indices = index_deserialize(worker_index, self.worker_shape, ranks=self.worker_ranks)
        logical_indices = []
        not_duplicated = boolean.true
        for i, (a, b) in enumerate(zip(self.shape, self.worker_shape)):
            if a < b:
                c = worker_indices[i] % a
                not_duplicated = logical_and(not_duplicated, worker_indices[i] < a)
            elif a > b:
                c = worker_indices[i] + local_indices[i] * b
            else:
                c = worker_indices[i]
            logical_indices.append(c)
        return logical_indices, not_duplicated

    def logical2local(
        self, logical_indices: List[Expr], worker_index: Optional[Expr] = None
    ) -> Tuple[List[Expr], Expr]:
        from hidet.ir.utils.index_transform import index_deserialize

        if worker_index is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker_index = threadIdx.x

        worker_indices = index_deserialize(worker_index, self.worker_shape, ranks=self.worker_ranks)
        local_indices = []
        is_valid = boolean.true
        for i, (a, b) in enumerate(zip(self.shape, self.worker_shape)):
            if a < b:
                # logical extent: ----
                #  worker extent: --------
                local_indices.append(int32.zero)
                is_valid = logical_and(is_valid, equal(worker_indices[i] % a, logical_indices[i]))
            elif a > b:
                # logical extent: --------
                #  worker extent: ----
                local_indices.append(logical_indices[i] // b)
                is_valid = logical_and(is_valid, equal(logical_indices[i] % b, worker_indices[i]))
            else:
                # logical extent: --------
                #  worker extent: --------
                local_indices.append(int32.zero)
                is_valid = logical_and(is_valid, equal(logical_indices[i], worker_indices[i]))
        return local_indices, is_valid


class RepeatLayout(AtomLayout):
    def __init__(self, shape: List[int], ranks: Optional[List[int]] = None):
        super().__init__(shape=shape, worker_shape=[1 for _ in range(len(shape))], ranks=ranks)


class SpatialLayout(AtomLayout):
    def __init__(self, shape: List[int], ranks: Optional[List[int]] = None):
        super().__init__(shape=shape, worker_shape=shape, ranks=ranks, worker_ranks=ranks)


class ComposedLayout(TileLayout):
    def __init__(self, outer: TileLayout, inner: TileLayout):
        self.outer: TileLayout = outer
        self.inner: TileLayout = inner

        assert len(self.outer.logical_shape()) == len(self.inner.logical_shape())

    def __str__(self):
        return '{}.{}'.format(self.outer, self.inner)

    def num_workers(self) -> int:
        return self.outer.num_workers() * self.inner.num_workers()

    def local_shape(self) -> List[int]:
        return [a * b for a, b in zip(self.outer.local_shape(), self.inner.local_shape())]

    def logical_shape(self) -> List[int]:
        return [a * b for a, b in zip(self.outer.logical_shape(), self.inner.logical_shape())]

    def local2logical(self, local_indices: List[Expr], worker_index: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        if worker_index is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker_index = threadIdx.x
        outer_worker = worker_index // self.inner.num_workers()
        inner_worker = worker_index % self.inner.num_workers()
        outer_local_indices = [a // b for a, b in zip(local_indices, self.inner.local_shape())]
        inner_local_indices = [a % b for a, b in zip(local_indices, self.inner.local_shape())]
        outer_logical, outer_not_duplicated = self.outer.local2logical(outer_local_indices, outer_worker)
        inner_logical, inner_not_duplicated = self.inner.local2logical(inner_local_indices, inner_worker)
        logical_indices = [a * s + b for a, s, b in zip(outer_logical, self.inner.logical_shape(), inner_logical)]
        not_duplicated = logical_and(outer_not_duplicated, inner_not_duplicated)
        return logical_indices, not_duplicated

    def logical2local(
        self, logical_indices: List[Expr], worker_index: Optional[Expr] = None
    ) -> Tuple[List[Expr], Expr]:
        if worker_index is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker_index = threadIdx.x
        outer_logical = [a // b for a, b in zip(logical_indices, self.inner.logical_shape())]
        inner_logical = [a % b for a, b in zip(logical_indices, self.inner.logical_shape())]
        outer_worker = worker_index // self.inner.num_workers()
        inner_worker = worker_index % self.inner.num_workers()
        outer_local, outer_is_valid = self.outer.logical2local(outer_logical, outer_worker)
        inner_local, inner_is_valid = self.inner.logical2local(inner_logical, inner_worker)
        local_indices = [a * b + c for a, b, c in zip(outer_local, self.inner.local_shape(), inner_local)]
        is_valid = logical_and(outer_is_valid, inner_is_valid)
        return local_indices, is_valid


class ParameterizedTileLayout(TileLayout):
    def __init__(self, layout: TileLayout = None):
        self.layout: TileLayout = layout

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return str(self.layout)

    def num_workers(self) -> int:
        return self.layout.num_workers()

    def local_shape(self) -> List[int]:
        return self.layout.local_shape()

    def logical_shape(self) -> List[int]:
        return self.layout.logical_shape()

    def local2logical(self, local_indices: List[Expr], worker_index: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        return self.layout.local2logical(local_indices, worker_index)

    def logical2local(
        self, logical_indices: List[Expr], worker_index: Optional[Expr] = None
    ) -> Tuple[List[Expr], Expr]:
        return self.layout.logical2local(logical_indices, worker_index)


def repeat(*shape: int, ranks: Optional[List[int]] = None) -> TileLayout:
    return RepeatLayout(list(shape), ranks=ranks)


def spatial(*shape: int, ranks: Optional[List[int]] = None) -> TileLayout:
    return SpatialLayout(list(shape), ranks=ranks)


def atom(*shape, workers: List[int], ranks: Optional[List[int]] = None, worker_ranks: Optional[List[int]] = None):
    if workers is None:
        workers = list(range(len(shape)))
    return AtomLayout(list(shape), workers, ranks, worker_ranks)


class SharedLayout(ParameterizedTileLayout):
    def __init__(self, shape: List[int]):
        super().__init__(layout=repeat(*shape))
        self.shape: List[int] = shape

    def __str__(self):
        return 'shared({})'.format(self.shape)

    def __eq__(self, other):
        return isinstance(other, SharedLayout) and same_list(self.shape, other.shape)


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


class BlockLayout(ParameterizedTileLayout):
    def __init__(
        self, shape: List[int], warps_per_block: List[int], thread_per_warp: List[int], size_per_thread: List[int]
    ):
        self.shape: List[int] = shape
        self.warps_per_block: List[int] = warps_per_block
        self.thread_per_warp: List[int] = thread_per_warp
        self.size_per_thread: List[int] = size_per_thread
        self.thread_shape: List[int] = [min(a, b) for a, b in zip(self.shape, self.size_per_thread)]
        self.warp_shape: List[int] = [
            min(a, b) for a, b in zip(self.shape, Vector(self.size_per_thread) * self.thread_per_warp)
        ]
        self.block_shape: List[int] = [
            min(a, b)
            for a, b in zip(self.shape, Vector(self.size_per_thread) * self.thread_per_warp * self.warps_per_block)
        ]
        self.layout_shape: List[int] = list(Vector(self.warps_per_block) * self.thread_per_warp * self.size_per_thread)
        super().__init__(
            layout=(
                atom(*Vector(self.shape) // self.block_shape, workers=[1 for _ in range(len(self.shape))])
                * atom(*Vector(self.block_shape) // self.warp_shape, workers=self.warps_per_block)
                * atom(*Vector(self.warp_shape) // self.thread_shape, workers=self.thread_per_warp)
                * atom(*self.thread_shape, workers=[1 for _ in range(len(self.shape))])
            )
        )

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, BlockLayout)
            and same_list(self.shape, other.shape)
            and same_list(self.size_per_thread, other.size_per_thread)
            and same_list(self.thread_per_warp, other.thread_per_warp)
            and same_list(self.warps_per_block, other.warps_per_block)
        )

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        fmt = 'block(shape={}, warps_per_block={}, thread_per_warp={}, size_per_thread={})'
        return fmt.format(self.shape, self.warps_per_block, self.thread_per_warp, self.size_per_thread)

    @staticmethod
    def from_shape(shape: List[int], num_warps: int, size_per_thread: Optional[List[int]] = None):
        if not is_power_of_two(prod(shape)):
            raise ValueError(f"The tensor must have a power of 2 number of elements, got {prod(shape)}")
        if size_per_thread is not None and not is_power_of_two(prod(size_per_thread)):
            raise ValueError(f"size_per_thread must have a power of 2 number of elements, got {prod(size_per_thread)}")
        if size_per_thread is None:
            size_per_thread = [1] * len(shape)
        if len(size_per_thread) != len(shape):
            raise ValueError(f"size_per_thread must have the same length as shape, got {size_per_thread}")
        orig_shape = shape
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

        return BlockLayout(orig_shape, warps_per_block, thread_per_warp, size_per_thread)


class FlattenBlockLayout(TileLayout):
    def __init__(self, parent: BlockLayout, axis: int):
        self.parent: BlockLayout = parent
        self.axis: int = axis

        shape = list(self.parent.shape)
        shape[axis] = 1
        self.flat_layout = BlockLayout(shape, parent.warps_per_block, parent.thread_per_warp, parent.size_per_thread)

    def __str__(self):
        return 'flatten_block(parent={}, axis={})'.format(self.parent, self.axis)
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, FlattenBlockLayout) and self.parent == other.parent and self.axis == other.axis

    def num_workers(self) -> int:
        return self.flat_layout.num_workers()

    def local_shape(self) -> List[int]:
        return self.flat_layout.local_shape()

    def logical_shape(self) -> List[int]:
        shape = self.flat_layout.logical_shape()
        assert shape[self.axis] == 1
        return shape[: self.axis] + shape[self.axis + 1 :]

    def local2logical(self, local_indices: List[Expr], worker_index: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices, not_duplicated = self.flat_layout.local2logical(local_indices, worker_index)
        logical_indices = logical_indices[: self.axis] + logical_indices[self.axis + 1 :]
        return logical_indices, not_duplicated

    def logical2local(
        self, logical_indices: List[Expr], worker_index: Optional[Expr] = None
    ) -> Tuple[List[Expr], Expr]:
        logical_indices = logical_indices[: self.axis] + [int32.zero] + logical_indices[self.axis :]
        return self.flat_layout.logical2local(logical_indices, worker_index)


class DotOperandLayout(ParameterizedTileLayout):
    pass


class BlockDotOperandLayout(DotOperandLayout):
    def __init__(self, parent: BlockLayout, k_size: int, op_idx: int):
        self.parent: BlockLayout = parent
        self.op_idx: int = op_idx

        shape: List[int] = parent.shape

        if op_idx == 0:
            layout = repeat(1, k_size) * BlockLayout(
                shape=[shape[0], 1],
                warps_per_block=parent.warps_per_block,
                thread_per_warp=parent.thread_per_warp,
                size_per_thread=[parent.size_per_thread[0], 1],
            )
        else:
            layout = repeat(k_size, 1) * BlockLayout(
                shape=[1, shape[1]],
                warps_per_block=parent.warps_per_block,
                thread_per_warp=parent.thread_per_warp,
                size_per_thread=[1, parent.size_per_thread[1]],
            )
        super().__init__(layout)

    def __str__(self):
        return 'block_dot_operand(parent={}, op_idx={})'.format(self.parent, self.op_idx)

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, BlockDotOperandLayout) and self.parent == other.parent and self.op_idx == other.op_idx
    
    def __hash__(self):
        return hash(str(self))


if __name__ == '__main__':
    # layout = BlockLayout(shape=[64, 32], warps_per_block=[2, 2], thread_per_warp=[8, 4], size_per_thread=[4, 4])
    # dd = {}
    # hash(layout)
    # dd[layout] = 1
    # print(layout.layout)
    # print(layout.visualize())
    # a_layout = BlockDotOperandLayout(layout, 4, 0)
    # print(a_layout.layout)
    # print(a_layout.visualize())
    # b_layout = BlockDotOperandLayout(layout, 4, 1)
    # print(b_layout.layout)
    # print(b_layout.visualize())

    # a = BlockLayout(shape=[16], warps_per_block=[1], thread_per_warp=[32], size_per_thread=[1])
    # print(a.visualize())
    # for w in range(32):
    #     print(w)
    #     for i in range(16):
    #         local_indices, is_valid = a.logical2local([int32(i)], worker_index=int32(w))
    #         print('({}, {}, {}) '.format(w, local_indices[0], is_valid), end='')
    #     print()

    a = BlockLayout(shape=[16, 16], warps_per_block=[1, 1], thread_per_warp=[4, 8], size_per_thread=[2, 2])
    print(a.visualize())
    for i in range(16):
        for j in range(16):
            for w in range(32):
                local_indices, is_valid = a.logical2local([int32(i), int32(j)], int32(w))
                _, not_duplicated = a.local2logical(local_indices, int32(w))
                if is_valid and not_duplicated:
                    print('{}'.format(w), end=' ')
        print()
