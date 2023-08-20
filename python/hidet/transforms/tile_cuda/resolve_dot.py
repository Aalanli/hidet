from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.ops.dot import Dot, SimtDot
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType
from hidet.transforms.base import TileFunctionPass


class ResolveDotRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Dot(self, e: Dot):
        a = self.visit(e.a)
        b = self.visit(e.b)
        c = self.visit(e.c)
        return SimtDot(a, b, c)


class ResolveDotPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = ResolveDotRewriter()
        return rewriter.visit(func)


def resolve_dot_pass() -> TileFunctionPass:
    return ResolveDotPass()
