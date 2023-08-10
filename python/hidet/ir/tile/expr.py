from typing import List, Dict, Union
from enum import Enum
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr

_ScalarConst = Union[str, int, float, bool]
CConst = Union[_ScalarConst, List[_ScalarConst]]  # compile-time constant


class TileOp:
    def __init__(self, args: List[Expr] = None, attrs: Dict[str, CConst] = None):
        self.args: List[Expr] = args if args is not None else []
        self.attrs: Dict[str, CConst] = attrs if attrs is not None else {}

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def reforward(self, args: List[Expr] = None, attrs: Dict[str, CConst] = None):
        return self.__class__(*args, **attrs)

    def make_call(self):
        return CallTileOp(self)

    def infer_type(self) -> BaseType:
        raise NotImplementedError()


class CallTileOp(Expr):
    def __init__(self, op: TileOp):
        self.op: TileOp = op


def call_tile_op(top: TileOp):
    return CallTileOp(top)
