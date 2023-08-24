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
    def __init__(self, ptr: Expr, dst: Expr, index: Expr, mask: Optional[Expr], other: Optional[Expr], axis: int):
        super().__init__(args=[ptr, dst, index], attrs={"axis": axis})
        if mask is not None:
            self.args.append(mask)
        if other is not None:
            assert mask is not None
            self.args.append(other)


class AsyncCommitGroup(TileOp):
    def __init__(self):
        super().__init__()


class AsyncWait(TileOp):
    def __init__(self, n: int):
        super().__init__(attrs={"n": n})


class ExtractSlice(TileOp):
    def __init__(self, src: Expr, index: Expr, axis: int, layout: Optional[TileLayout] = None):
        super().__init__(args=[src, index], attrs={"axis": axis, "layout": layout})
