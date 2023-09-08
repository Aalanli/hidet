from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.dtypes import float32, float16, bfloat16
from hidet.ir.tile.ops.dot import Dot, SimtDot, MmaDot
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
        c_type = self.type_infer(c)
        assert isinstance(c_type, TileType)
        dtype = c_type.type

        # if dtype in [float16, bfloat16]:
        #     return MmaDot(a, b, c)
        if dtype in [float16, float32]:
            return SimtDot(a, b, c)
        raise NotImplementedError(dtype)


class ResolveDotPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = ResolveDotRewriter()
        return rewriter.visit(func)


def resolve_dot_pass() -> TileFunctionPass:
    return ResolveDotPass()
