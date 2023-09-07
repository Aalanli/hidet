from typing import List
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, Let
from hidet.ir.tile.ops.dot import Dot, SimtDot
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tile.ops.smem import extract_slice
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.ops import convert_layout, dot
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tile.layout import TileLayout, SharedLayout, DotOperandLayout
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.transforms.tile_cuda.remove_layout_convert import FoldConvertLayoutTransform, IdentityConvertLayoutTransform
from hidet.transforms.tile_generic.dead_code_elimination import DeadCodeEliminationRewriter

"""
d = dot(a, b, c)    # a: [m, ks], b: [ks, n], c: [m, n] where ks = s * k

=>

a = convert_layout(a, smem)
b = convert_layout(b, smem)
a0 = convert_layout(extract_slice(a, start=0, extent=k, axis=1), dot_operand_layout)
b0 = convert_layout(extract_slice(b, start=0, extent=k, axis=0), dot_operand_layout)
d0 = dot(a0, b0, c)
a1 = convert_layout(extract_slice(a, start=k, extent=k, axis=1), dot_operand_layout)
b1 = convert_layout(extract_slice(b, start=k, extent=k, axis=0), dot_operand_layout)
d = dot(a1, b1, d0)
"""


class LetChain:
    def __init__(self):
        self.lets: List[Let] = []
        self.type_infer = TypeInfer()

    def let(self, hint, value: Expr) -> Var:
        tp = self.type_infer(value)
        v = Var(hint, tp)
        self.lets.append(Let(v, value, None))
        return v

    def make_expr(self, body: Expr):
        for let in reversed(self.lets):
            body = Let(let.var, let.value, body)
        return body


class SplitDotKRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Dot(self, e: Dot):
        if isinstance(e, SimtDot):
            a = self.visit(e.a)
            b = self.visit(e.b)
            c = self.visit(e.c)
            a_type: TileType = self.type_infer(a)
            b_type: TileType = self.type_infer(b)
            c_type: TileType = self.type_infer(c)
            m, n, ks = a_type.shape[0], b_type.shape[1], a_type.shape[1]
            if ks <= 4:
                return super().visit_Dot(e)
            s = ks // 4
            k = 4

            a_layout = a_type.layout
            b_layout = b_type.layout

            let_chain: List[Let] = []
            cb = LetChain()  # chain builder

            a = cb.let('a', convert_layout(a, SharedLayout([m, ks]), scope=TileScope.Shared))
            b = cb.let('b', convert_layout(b, SharedLayout([ks, n]), scope=TileScope.Shared))
            d = c
            for i in range(s):
                start = int32(i * k)
                ai = cb.let(
                    'a',
                    convert_layout(extract_slice(a, start, extent=k, axis=1, layout=SharedLayout([m, ks])), a_layout),
                )
                bi = cb.let(
                    'b',
                    convert_layout(extract_slice(b, start, extent=k, axis=0, layout=SharedLayout([ks, n])), b_layout),
                )
                d = cb.let('c', SimtDot(ai, bi, d).make_call())
            return cb.make_expr(d)
        else:
            raise NotImplementedError()


class SplitDotKPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_transforms(
            func,
            [
                SplitDotKRewriter(),
                canonicalize_to_ssa,
                FoldConvertLayoutTransform(),
                IdentityConvertLayoutTransform(),
                DeadCodeEliminationRewriter(),
            ],
        )


def split_dot_k_pass() -> TileFunctionPass:
    return SplitDotKPass()
