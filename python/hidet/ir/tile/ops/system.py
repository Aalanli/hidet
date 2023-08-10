from typing import Union, Optional
from hidet.ir.expr import Var
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp


class ProgramId(TileOp):
    def __init__(self):
        super().__init__()

    def infer_type(self) -> BaseType:
        from hidet.ir.dtypes import int32

        return int32


class NumPrograms(TileOp):
    def __init__(self):
        super().__init__()

    def infer_type(self) -> BaseType:
        from hidet.ir.dtypes import int32

        return int32


def program_id():
    return ProgramId().make_call()


def num_programs():
    return NumPrograms().make_call()
