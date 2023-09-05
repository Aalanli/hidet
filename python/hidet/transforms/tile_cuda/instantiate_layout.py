from typing import List, Optional, Tuple
from hidet.ir.expr import Var, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import TileLayout, spatial, repeat
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store
from hidet.ir.tile.ops import Construct, Assign, convert_layout
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.utils import same_list, is_power_of_two, prod, argmin, argmax, gcd


class LayoutPlan:
    def __init__(
        self,
        repeat_shape: List[int],
        repeat_workers: int,
        warps_per_block: List[int],
        threads_per_warp: List[int],
        size_per_thread: List[int],
        num_warps: int
    ):
        self.repeat_shape: List[int] = repeat_shape
        self.repeat_workers: int = repeat_workers
        self.warps_per_block: List[int] = warps_per_block
        self.threads_per_warp: List[int] = threads_per_warp
        self.size_per_thread: List[int] = size_per_thread
        self.num_warps: int = num_warps

    @staticmethod
    def from_shape(shape: List[int], num_warps: int, size_per_thread: Optional[List[int]] = None):
        if not is_power_of_two(prod(shape)):
            raise ValueError(f"The tensor must have a power of 2 number of elements, got {prod(shape)}")
        if size_per_thread is not None and not is_power_of_two(prod(size_per_thread)):
            raise ValueError(f"size_per_thread must have a power of 2 number of elements, got {prod(size_per_thread)}")
        if size_per_thread is None:
            size_per_thread = [1] * len(shape)
        if len(size_per_thread) != len(shape):
            raise ValueError(f"size_per_thread must have the same length as shape, got {size_per_thread}")
        if any(a > b for a, b in zip(size_per_thread, shape)):
            raise ValueError(f"size_per_thread must be smaller than shape, got {size_per_thread} and {shape}")
        shape = [max(extent // size, 1) for extent, size in zip(shape, size_per_thread)]
        threads_per_warp = []
        remaining_threads = 32
        for i in reversed(range(len(shape))):
            s = gcd(shape[i], remaining_threads)
            threads_per_warp.append(s)
            shape[i] //= s
            remaining_threads //= s

        warps_per_block = []
        remaining_warps = num_warps
        for i in reversed(range(len(shape))):
            s = gcd(shape[i], remaining_warps)
            warps_per_block.append(s)
            shape[i] //= s
            remaining_warps //= s

        threads_per_warp = list(reversed(threads_per_warp))
        warps_per_block = list(reversed(warps_per_block))
        return LayoutPlan(
            repeat_shape=shape,
            repeat_workers=remaining_warps * remaining_threads,
            warps_per_block=warps_per_block,
            threads_per_warp=threads_per_warp,
            size_per_thread=size_per_thread,
            num_warps=num_warps
        )

    def make_layout(self) -> TileLayout:
        return (
            repeat(*self.repeat_shape, workers=self.repeat_workers)
            * spatial(*self.warps_per_block)
            * spatial(*self.threads_per_warp)
            * repeat(*self.size_per_thread)
        )

    def make_expand_dim_input_layout(self, axis: int) -> TileLayout:
        assert (
            self.repeat_shape[axis]
            == self.warps_per_block[axis]
            == self.threads_per_warp[axis]
            == self.size_per_thread[axis]
            == 1
        )
        return LayoutPlan(
            repeat_shape=self.repeat_shape[:axis] + self.repeat_shape[axis + 1:],
            repeat_workers=self.repeat_workers,
            warps_per_block=self.warps_per_block[:axis] + self.repeat_shape[axis + 1:],
            threads_per_warp=self.threads_per_warp[:axis] + self.repeat_shape[axis + 1:],
            size_per_thread=self.size_per_thread[:axis] + self.repeat_shape[axis + 1:],
            num_warps=self.num_warps
        ).make_layout()

    def make_reduce_output_layout(self, axis: int, keepdims: bool) -> TileLayout:
        rank = len(self.repeat_shape)
        if keepdims:
            repeat_shape = self.repeat_shape[:axis] + self.repeat_shape[axis + 1:],
            warps_per_block = self.warps_per_block[:axis] + self.repeat_shape[axis + 1:],
            threads_per_warp = self.threads_per_warp[:axis] + self.repeat_shape[axis + 1:],
            size_per_thread = self.size_per_thread[:axis] + self.repeat_shape[axis + 1:],
            layout = repeat(*repeat_shape, workers=self.repeat_workers)
            layout *= spatial(*warps_per_block)
            if self.warps_per_block[axis] > 1:
                layout *= repeat(*[1 for _ in range(rank)], workers=self.warps_per_block[axis])
            layout *= spatial(*threads_per_warp)
            if self.threads_per_warp[axis] > 1:
                layout *= repeat(*[1 for _ in range(rank)], workers=self.threads_per_warp[axis])
            layout *= repeat(*size_per_thread)
            return layout
        else:
            pass

    def make_simt_dot_input_layouts(self) -> Tuple[TileLayout, TileLayout]:
        pass


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.num_warps: int = 0
        self.type_infer = TypeInfer()

    def layout_from_shape(self, shape: List[int], size_per_thread: Optional[List[int]] = None) -> TileLayout:
        layout = LayoutPlan.from_shape(shape, self.num_warps, size_per_thread).make_layout()
        # check the layout is correct
        assert same_list(shape, layout.logical_shape()) and layout.num_workers() == self.num_warps * 32
        return layout

    def visit_Function(self, func: Function):
        if 'cuda.block_dim' not in func.attrs:
            raise ValueError("cuda.block_dim must be specified for 'cuda_tile' function")
        block_dim = func.attrs['cuda.block_dim']
        try:
            block_dim = int(block_dim)
        except ValueError:
            raise ValueError(f"cuda.block_dim must be a constant integer, got {block_dim}")
        self.num_warps = block_dim // 32
        if block_dim % 32 != 0:
            raise ValueError(f"cuda.block_dim must be a multiple of 32, got {block_dim}")
        return super().visit_Function(func)

    def visit_CallTileOp(self, call: CallTileOp):
        op = self.visit(call.op)
        if op is call.op:
            ret = call
        else:
            ret = op.make_call()
        ttype = self.type_infer.visit(ret)
        if isinstance(ttype, TileType) and ttype.layout is None:
            raise NotImplementedError(
                'The layout of the following tile op has not been instantiated:\n'
                + '  {}\n'.format(type(call.op).__name__)
            )
        return ret

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for orig_var, orig_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(orig_value)
            bind_var = var(orig_var.hint, self.type_infer(bind_value))
            self.memo[orig_var] = bind_var
            bind_vars.append(bind_var)
            bind_values.append(bind_value)
        body = self.visit(stmt.body)
        return LetStmt(bind_vars, bind_values, body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.pure_for_stmts.append(stmt)
        extent: Expr = self.visit(stmt.extent)
        values: List[Expr] = self.visit(stmt.values)
        args: List[Var] = [Var(arg.hint, self.type_infer(value)) for arg, value in zip(stmt.args, values)]
        for orig_arg, arg in zip(stmt.args, args):
            self.memo[orig_arg] = arg
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()
        let_vars = [Var(let_var.hint, arg.type) for let_var, arg in zip(stmt.let_vars, args)]
        for orig_let_var, let_var in zip(stmt.let_vars, let_vars):
            self.memo[orig_let_var] = let_var
        let_body = self.visit(stmt.let_body)
        return PureForStmt(args, values, stmt.loop_var, extent, body, let_vars, let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        args: List[Var] = self.visit(for_stmt.args)
        yields: List[Expr] = self.visit(stmt.values)
        updated_values = []
        for arg, yie in zip(args, yields):
            yield_type = self.type_infer(yie)
            if isinstance(arg.type, TileType):
                assert isinstance(yield_type, TileType)
                if arg.type.layout != yield_type.layout:
                    updated_values.append(convert_layout(yie, arg.type.layout))
                else:
                    updated_values.append(yie)
            else:
                updated_values.append(yie)
        return YieldStmt(updated_values)

    def visit_Arange(self, e: Arange):
        assert e.layout is None
        layout = self.layout_from_shape([e.end - e.begin])
        return Arange(e.begin, e.end, layout)

    def visit_Full(self, e: Full):
        assert e.layout is None
        layout = self.layout_from_shape(e.shape)
        value = self.visit(e.value)
        return Full(value, e.shape, layout)

    def visit_Construct(self, e: Construct):
        assert e.layout is None
        layout = self.layout_from_shape(e.shape)
        value = self.visit(e.value)
        return Construct(value, e.shape, e.axes, layout)

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        x = self.visit(e.x)
        y = self.visit(e.y)
        x_type = self.type_infer.visit(x)
        y_type = self.type_infer.visit(y)
        assert isinstance(x_type, TileType) and isinstance(y_type, TileType)
        assert same_list(x_type.shape, y_type.shape)

        if x_type.layout != y_type.layout:
            y = convert_layout(y, x_type.layout)
            return e.reforward([x, y])
        else:
            return super().visit_BinaryTileOp(e)

    def visit_Broadcast(self, e: Broadcast):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        assert x_type.scope == TileScope.Register
        y_layout = repeat(*[a // b for a, b in zip(e.shape, x_type.shape)]) * x_type.layout
        return Broadcast(x, e.shape, y_layout)

    def visit_ExpandDims(self, e: ExpandDims):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)

        y_shape = x_type.shape[: e.axis] + [1] + x_type.shape[e.axis:]
        plan = LayoutPlan.from_shape(y_shape, num_warps=self.num_warps)

        x_layout = plan.make_expand_dim_input_layout(axis=e.axis)
        y_layout = plan.make_layout()

        return ExpandDims(x=convert_layout(x, layout=x_layout), axis=e.axis, layout=y_layout)

    def visit_ReduceOp(self, e: ReduceOp):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType) and x_type.scope == TileScope.Register

        plan = LayoutPlan.from_shape(x_type.shape, num_warps=self.num_warps)
        y_layout = plan.make_reduce_output_layout(axis=e.axis, keepdims=e.keepdims)
        return ReduceOp(x=x, axis=e.axis, keepdims=e.keepdims, kind=e.kind, layout=y_layout)

    def visit_Dot(self, e: Dot):
        if isinstance(e, SimtDot):
            a = self.visit(e.a)
            b = self.visit(e.b)
            c = self.visit(e.c)
            a_type: TileType = self.type_infer.visit(a)
            b_type: TileType = self.type_infer.visit(b)
            c_type: TileType = self.type_infer.visit(c)
            m, n = c_type.shape
            num_threads = self.num_warps * 32
            if m * n >= num_threads * 16:
                size_per_thread = [4, 4]
            elif m * n >= num_threads * 4:
                size_per_thread = [2, 2]
            else:
                size_per_thread = [1, 1]
            layout = BlockLayout.from_shape([m, n], num_warps=self.num_warps, size_per_thread=size_per_thread)
            if not (isinstance(a_type.layout, BlockDotOperandLayout) and a_type.layout.parent == layout):
                a = convert_layout(a, BlockDotOperandLayout(parent=layout, op_idx=0))
            if not (isinstance(b_type.layout, BlockDotOperandLayout) and b_type.layout.parent == layout):
                b = convert_layout(b, BlockDotOperandLayout(parent=layout, op_idx=1))
            if c_type.layout != layout:
                c = convert_layout(c, layout)
            return SimtDot(a, b, c)
        else:
            raise NotImplementedError()

    def visit_Store(self, e: Store):
        ptr = self.visit(e.ptr)
        value = self.visit(e.value)
        mask = self.visit(e.mask) if e.mask is not None else None

        ptr_type: TileType = self.type_infer.visit(ptr)
        value_type: TileType = self.type_infer.visit(value)
        mask_type: TileType = self.type_infer.visit(mask) if mask is not None else None

        # we use the layout of the value to determine the layout of the ptr and mask
        layout = value_type.layout

        if ptr_type.layout != layout:
            ptr = convert_layout(ptr, layout)

        if mask is not None and mask_type.layout != layout:
            mask = convert_layout(mask, layout)

        return Store(ptr, value, mask)

    def visit_Assign(self, e: Assign):
        dst = self.visit(e.dst)
        src = self.visit(e.src)
        dst_type = self.type_infer.visit(dst)
        src_type = self.type_infer.visit(src)
        assert isinstance(dst_type, TileType) and isinstance(src_type, TileType)
        if dst_type.layout != src_type.layout:
            src = convert_layout(src, dst_type.layout)
        return Assign(dst, src)


class InstantiateLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InstantiateLayoutRewriter()
        func = rewriter(func)
        func = canonicalize_to_ssa(func)
        return func


def instantiate_layout(func: Function) -> Function:
    rewriter = InstantiateLayoutRewriter()
    return canonicalize_to_ssa(rewriter.visit(func))


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()

# if __name__ == '__main__':
#     rewriter = InstantiateLayoutRewriter()
#     rewriter.num_warps = 2
#     a = rewriter.layout_from_shape([64, 32], size_per_thread=[4, 4])
#     print(a)
#     print(a.visualize())
#     b = spatial(2, 1).spatial(2, 1).spatial(4, 1).repeat(1, 16, workers=8).repeat(4, 1)
#     print(b)
#     print(b.visualize())
#
#     # b = rewriter.layout_from_shape([1, 1])
#     # print(b)
#     # c = rewriter.layout_from_shape([4, 4])
#     # print(c)
#     # print(c.visualize())
