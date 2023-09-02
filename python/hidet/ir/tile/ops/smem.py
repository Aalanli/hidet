from typing import Union, Optional, List, Dict, Any
from enum import Enum
from hidet.ir.expr import Expr
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.layout import TileLayout, SharedLayout
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp, call_tile_op


class ProcedureOp(TileOp):
    def __init__(self, attrs: Dict[str, Any]):
        super().__init__(args=[], attrs=attrs)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return void


class AllocTensor(TileOp):
    def __init__(self, dtype: Union[DataType, PointerType], shape: List[int]):
        super().__init__(args=[], attrs={"dtype": dtype, "shape": shape})
        self.dtype: Union[PointerType, DataType] = dtype
        self.shape: List[int] = shape

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        assert len(arg_types) == 0
        return TileType(type_=self.dtype, shape=self.shape, layout=SharedLayout(shape=self.shape))


class InsertSliceAsync(TileOp):
    def __init__(
        self,
        ptr: Expr,
        dst: Expr,
        index: Expr,
        mask: Optional[Expr] = None,
        other: Optional[Expr] = None,
        axis: int = 0,
    ):
        super().__init__(args=[ptr, dst, index], attrs={"axis": axis})
        if mask is not None:
            self.args.append(mask)
        if other is not None:
            assert mask is not None
            self.args.append(other)
        self.ptr: Expr = ptr
        self.dst: Expr = dst
        self.index: Expr = index
        self.mask: Optional[Expr] = mask
        self.other: Optional[Expr] = other
        self.axis: int = axis

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        dst_type = arg_types[1]
        return dst_type


class AsyncCommitGroup(ProcedureOp):
    def __init__(self):
        super().__init__(attrs={})


class AsyncWait(ProcedureOp):
    def __init__(self, n: int):
        super().__init__(attrs={"n": n})
        self.n: int = n


class ExtractSlice(TileOp):
    def __init__(self, src: Expr, start: Expr, axis: int, extent: int, layout: Optional[TileLayout] = None):
        super().__init__(args=[src, start], attrs={"axis": axis, "extent": extent, "layout": layout})
        self.axis: int = axis
        self.extent: int = extent
        self.layout: Optional[TileLayout] = layout

    @property
    def var_name_hint(self):
        return "ext_slice"

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        src_type = arg_types[0]
        assert isinstance(src_type, TileType)
        src_shape: List[int] = src_type.shape
        if self.extent == 1:
            shape = src_shape[: self.axis] + src_shape[self.axis + 1 :]
        else:
            shape = src_shape[: self.axis] + [self.extent] + src_shape[self.axis + 1 :]
        return TileType(type_=src_type.type, shape=shape, layout=self.layout)


def extract_slice(src: Expr, start_index: Expr, axis: int, extent: int, layout: Optional[TileLayout] = None):
    return ExtractSlice(src, start_index, axis, extent, layout).make_call()
