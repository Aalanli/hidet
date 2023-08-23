from typing import Union, Optional, List
from enum import Enum
from hidet.ir.expr import Expr
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp, call_tile_op


class AllocTensor(TileOp):
    def __init__(self, shape: List[int]):
        super().__init__(args=[], attrs={"shape": shape})


class InsertSliceAsync(TileOp):
    def __init__(self, ptr: Expr, dst: Expr, axis: int, index: int):
        super().__init__(args=[ptr, dst], attrs={"axis": axis, "index": index})


class AsyncCommitGroup(TileOp):
    def __init__(self):
        super().__init__()


class AsyncWait(TileOp):
    def __init__(self):
        super().__init__()


class ExtractSlice(TileOp):
    def __init__(self, src: Expr, axis: int, index: int, layout: Optional[TileLayout] = None):
        super().__init__(args=[src], attrs={"axis": axis, "index": index, "layout": layout})
