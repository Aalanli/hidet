from typing import List, Union

from hidet.ir.expr import Var
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, TileLayout, SharedLayout, DistributedLayout
from hidet.ir.type import DataType, PointerType


class Buffer:
    def __init__(
        self,
        buf_var: Var,
        dtype: Union[PointerType, DataType],
        shape: List[int],
        local_shape: List[int],
        layout: TileLayout
    ):
        self.var: Var = buf_var
        self.dtype: Union[PointerType, DataType] = dtype
        self.shape: List[int] = shape
        self.local_shape: List[int] = local_shape
        self.layout: TileLayout = layout

    def __getitem__(self, item):
        return self.var[item]

    @property
    def block_layout(self) -> BlockLayout:
        assert isinstance(self.layout, BlockLayout)
        return self.layout

    @property
    def shared_layout(self) -> SharedLayout:
        assert isinstance(self.layout, SharedLayout)
        return self.layout

    @property
    def flatten_block_layout(self) -> FlattenBlockLayout:
        assert isinstance(self.layout, FlattenBlockLayout)
        return self.layout

    @property
    def distributed_layout(self) -> DistributedLayout:
        assert isinstance(self.layout, DistributedLayout)
        return self.layout

    def is_shared(self):
        return isinstance(self.layout, SharedLayout)

    def is_block(self):
        return isinstance(self.layout, BlockLayout)

    def is_flatten_block(self):
        return isinstance(self.layout, FlattenBlockLayout)

    def is_distributed(self):
        return isinstance(self.layout, DistributedLayout)
