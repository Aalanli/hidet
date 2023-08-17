from typing import List, Union, Optional

from hidet.ir.expr import Var, tensor_var
from hidet.ir.layout import DataLayout
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, TileLayout, SharedLayout
from hidet.ir.type import DataType, PointerType


class Buffer:
    def __init__(
        self,
        hint: str,
        dtype: Union[PointerType, DataType],
        shape: List[int],
        local_shape: List[int],
        layout: TileLayout,
        var_data_layout: Optional[DataLayout] = None,
    ):
        self.var: Var = tensor_var(hint=hint, shape=local_shape, dtype=dtype, layout=var_data_layout)
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

    def is_shared(self):
        return isinstance(self.layout, SharedLayout)

    def is_block(self):
        return isinstance(self.layout, BlockLayout)

    def is_flatten_block(self):
        return isinstance(self.layout, FlattenBlockLayout)

    def is_block_like(self):
        return isinstance(self.layout, (BlockLayout, FlattenBlockLayout))
