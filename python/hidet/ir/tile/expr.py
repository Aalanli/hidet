from typing import List, Dict, Union
from functools import lru_cache
from hidet.ir.node import Node
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr


class Attribute:
    pass


_ScalarConst = Union[str, int, float, bool, Attribute]
CConst = Union[_ScalarConst, List[_ScalarConst]]  # compile-time constant


class TileOp(Node):
    def __init__(self, args: List[Expr] = None, attrs: Dict[str, CConst] = None):
        self.args: List[Expr] = args if args is not None else []
        self.attrs: Dict[str, CConst] = attrs if attrs is not None else {}

    @classmethod
    def op_name(cls):
        # camel to snake (e.g., CamelName -> camel_name)
        camel_name = cls.__name__
        snake_name = "".join(["_" + c.lower() if c.isupper() else c for c in camel_name]).lstrip("_")
        return snake_name

    @property
    def name(self):
        return self.op_name()

    def reforward(self, args: List[Expr] = None, attrs: Dict[str, CConst] = None):
        if attrs is None:
            attrs = self.attrs
        return self.__class__(*args, **attrs)

    def make_call(self):
        return CallTileOp(self)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        raise NotImplementedError(
            "'infer_type' method has not been implemented for '{}' tile operator".format(type(self).__name__)
        )


class CallTileOp(Expr):
    def __init__(self, op: TileOp):
        self.op: TileOp = op


def call_tile_op(top: TileOp):
    return CallTileOp(top)
