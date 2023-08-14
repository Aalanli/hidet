from typing import List
from hidet.ir.type import BaseType
from hidet.ir.tile.expr import TileOp
from hidet.ir import dtypes


class ProgramId(TileOp):
    def __init__(self):
        super().__init__()

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return dtypes.int32


class NumPrograms(TileOp):
    def __init__(self):
        super().__init__()

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return dtypes.int32


def program_id():
    return ProgramId().make_call()


def num_programs():
    return NumPrograms().make_call()
