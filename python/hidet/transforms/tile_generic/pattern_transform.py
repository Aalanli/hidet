from typing import List, Dict, Optional
import functools

from hidet.ir.expr import Expr, Var, Constant
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.ops import Full, Construct, Dot, dot
from hidet.ir.func import Function
from hidet.transforms.base import TileFunctionPass
from hidet.utils import repeat_until_converge
from .utils.pattern import PatternTransform, Pattern, TilePattern, apply_transforms
from .dead_code_elimination import DeadCodeEliminationRewriter


class DotAddTransform(PatternTransform):
    """
    add(dot(a, b, 0), c) => dot(a, b, c)
    """

    def __init__(self):
        super().__init__()
        self.a = self.any_tile()
        self.b = self.any_tile()
        self.c = self.any_tile()
        self.zero = self.any_tile()

        self.pattern = self.add(self.dot(self.a, self.b, self.zero), self.c)

    def source(self) -> Pattern:
        return self.pattern

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[CallTileOp]:
        a = matched[self.a]
        b = matched[self.b]
        c = matched[self.c]
        zero = matched[self.zero]
        if not self.is_zero(zero, var2call):
            return None
        return Dot(a, b, c).make_call()


class PatternTransformPass(TileFunctionPass):
    def __init__(self, transforms: List[PatternTransform]):
        super().__init__()
        self.transforms: List[PatternTransform] = transforms

    def process_tile_func(self, func: Function) -> Function:
        rewriter = DeadCodeEliminationRewriter()
        func = self.apply_transforms(func, self.transforms, repeat_limit=-1)
        func = rewriter(func)
        return func


def pattern_transform_pass() -> TileFunctionPass:
    transforms = [DotAddTransform()]
    return PatternTransformPass(transforms)
