from hidet.ir.expr import Var, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store
from hidet.ir.tile.ops import Construct, Assign, convert_layout
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.convert_tile_expr_to_let import convert_to_let
from hidet.utils import same_list


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps
        self.type_infer = TypeInfer()

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

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt.init is None:
            return super().visit_DeclareStmt(stmt)
        else:
            init = self.visit(stmt.init)
            stmt_var = var(stmt.var.hint, self.type_infer(init))
            self.memo[stmt.var] = stmt_var
            return DeclareStmt(stmt_var, init)

    def visit_AssignStmt(self, stmt: AssignStmt):
        value = self.visit(stmt.value)
        stmt_var: Var = self.visit(stmt.var)
        value_type = self.type_infer.visit(value)
        if isinstance(value_type, TileType):
            assert isinstance(stmt_var.type, TileType)
            if value_type.layout != stmt_var.type.layout:
                value = convert_layout(value, stmt_var.type.layout)
            return AssignStmt(stmt_var, value)
        else:
            assert not isinstance(stmt_var.type, TileType)
            super().visit_AssignStmt(stmt)

    def visit_Arange(self, e: Arange):
        layout = BlockLayout.from_shape([e.end - e.begin], self.num_warps)
        return Arange(e.begin, e.end, layout)

    def visit_Full(self, e: Full):
        layout = BlockLayout.from_shape(e.shape, self.num_warps)
        value = self.visit(e.value)
        return Full(value, e.shape, layout)

    def visit_Construct(self, e: Construct):
        value = self.visit(e.value)
        layout = BlockLayout.from_shape(e.shape, self.num_warps)
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
        if isinstance(x_type.layout, BlockLayout):
            y_layout = BlockLayout(
                size_per_thread=x_type.layout.size_per_thread,
                thread_per_warp=x_type.layout.thread_per_warp,
                warps_per_block=x_type.layout.warps_per_block,
            )
        else:
            raise NotImplementedError()
        return Broadcast(x, e.shape, y_layout)

    def visit_ExpandDims(self, e: ExpandDims):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        if isinstance(x_type.layout, BlockLayout):
            y_shape = x_type.shape[: e.axis] + [1] + x_type.shape[e.axis :]
            y_layout = BlockLayout.from_shape(y_shape, self.num_warps)
            return ExpandDims(
                x=convert_layout(x, layout=FlattenBlockLayout(y_layout, axis=e.axis)), axis=e.axis, layout=y_layout
            )
        else:
            raise NotImplementedError()

    def visit_ReduceOp(self, e: ReduceOp):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        if isinstance(x_type.layout, BlockLayout):
            y_type = e.infer_type([x_type])
            if e.keepdims:
                layout = x_type.layout
            else:
                layout = FlattenBlockLayout(x_type.layout, axis=e.axis)
            assert isinstance(y_type, TileType)
            return ReduceOp(x=x, axis=e.axis, keepdims=e.keepdims, kind=e.kind, layout=layout)
        else:
            raise NotImplementedError()

    def visit_Dot(self, e: Dot):
        if isinstance(e, SimtDot):
            a = self.visit(e.a)
            b = self.visit(e.b)
            c = self.visit(e.c)
            c_type = self.type_infer.visit(c)
            m, n = c_type.shape
            num_threads = self.num_warps * 32
            if m * n >= num_threads * 16:
                size_per_thread = [4, 4]
            elif m * n >= num_threads * 4:
                size_per_thread = [2, 2]
            else:
                size_per_thread = [1, 1]
            layout = BlockLayout.from_shape([m, n], num_warps=self.num_warps, size_per_thread=size_per_thread)
            a = convert_layout(a, BlockDotOperandLayout(parent=layout, op_idx=0))
            b = convert_layout(b, BlockDotOperandLayout(parent=layout, op_idx=1))
            c = convert_layout(c, layout)
            return SimtDot(a, b, c, layout)
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
        if 'cuda.block_dim' not in func.attrs:
            raise ValueError("cuda.block_dim must be specified for 'cuda_tile' function")
        block_dim = func.attrs['cuda.block_dim']
        try:
            block_dim = int(block_dim)
        except ValueError:
            raise ValueError(f"cuda.block_dim must be a constant integer, got {block_dim}")
        num_warps = block_dim // 32
        if block_dim % 32 != 0:
            raise ValueError(f"cuda.block_dim must be a multiple of 32, got {block_dim}")
        rewriter = InstantiateLayoutRewriter(num_warps)
        return convert_to_let(rewriter.visit(func))


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()
