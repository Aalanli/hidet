from typing import Dict, Optional, List, Type, Callable
import operator

import hidet.ir.tools
from hidet.ir.type import DataType, PointerType, sizeof
from hidet.ir.expr import Var, Constant, Div, Mod, Add, Sub, Multiply, Expr
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor, IRFunctor
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt, EvaluateStmt
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt, PureYieldStmt
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store, Load
from hidet.ir.tile.ops import Construct, Assign, convert_layout
from hidet.ir.tile.layout import BlockLayout, TileLayout, DistributedLayout
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.convert_tile_expr_to_let import convert_to_let
from hidet.transforms.tile_cuda.instantiate_layout import instantiate_layout
from hidet.utils import same_list, gcd
from .utils.affine import affine_decompose
from .utils.type_updater import update_type


class Value:
    def merge(self, other):
        raise NotImplementedError()

    def as_tile_value(self):
        assert isinstance(self, TileValue)
        return self

class Constancy:
    """
    The constancy of a variable indicates whether the variable is a constant.
    """

    def __init__(self, v: Optional[int] = None):
        self.v: Optional[int] = v

    def __eq__(self, other):
        return self.v == other.v

    def __add__(self, other):
        if self.v is not None and other.v is not None:
            return Constancy(self.v + other.v)
        else:
            return Constancy()

    def __mul__(self, other):
        if self.v is not None and other.v is not None:
            return Constancy(self.v * other.v)
        else:
            return Constancy()

    def merge(self, other):
        if self.v == other.v:
            return self
        else:
            return Constancy()


class Divisibility:
    """
    The divisibility of a variable indicates the greatest common divisor of its all possible values.
    """

    def __init__(self, v: int = 1):
        self.v: int = v

    def __eq__(self, other):
        return self.v == other.v

    def __add__(self, other):
        return Divisibility(gcd(self.v, other.v))

    def __mul__(self, other):
        return Divisibility(self.v * other.v)

    def merge(self, other):
        return Divisibility(gcd(self.v, other.v))


class ScalarValue(Value):
    """
    The constancy and divisibility of a scalar variable.
    """

    def __init__(self, constant: Constancy = Constancy(), divisibility: Divisibility = Divisibility()):
        self.constant: Constancy = constant
        self.divisibility: Divisibility = divisibility

    def __repr__(self):
        if self.constant.v is not None:
            return str(self.constant.v)
        else:
            return 'div({})'.format(self.divisibility.v)

    def __eq__(self, other):
        return self.constant == other.constant and self.divisibility == other.divisibility

    def __add__(self, other):
        return ScalarValue(self.constant + other.constant, self.divisibility + other.divisibility)

    def __mul__(self, other):
        return ScalarValue(self.constant * other.constant, self.divisibility * other.divisibility)

    def merge(self, other):
        return ScalarValue(self.constant.merge(other.constant), self.divisibility.merge(other.divisibility))


class TileValue(Value):
    """
    The constancy and divisibility of a tile variable.

    Example:
        Given a tile a with shape [m, n]. If the value at position (i, j) can be linearly expressed as
          a(i, j) = w0 * i + w1 * j + w2
        where w0, w1, w2 are independent of i and j, then the tile value of a is
          [w0, w1, w2]
        this class does not store the actual (w0, w1, w2), but store their constancy and divisibility.
    We can use this information to know the properties of the tile and help us to determine the coalescing strategy.

    > For tile
    [1    2   3   4]
    [5    6   7   8]
    [9   10  11  12]
    [13  14  15  16]
    we have TileValue [4, 1, 1] because the value at position (i, j) is i * 4 + j * 1 + 1.

    """

    def __init__(self, weights: List[ScalarValue]):
        self.weights: List[ScalarValue] = weights

    def __repr__(self):
        return '[' + ', '.join(str(w) for w in self.weights) + ']'

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.weights, other.weights))

    def __add__(self, other):
        return TileValue([a + b for a, b in zip(self.weights, other.weights)])

    def __mul__(self, other):
        return TileValue([a * b for a, b in zip(self.weights, other.weights)])

    def merge(self, other):
        return TileValue([a.merge(b) for a, b in zip(self.weights, other.weights)])


class CoalesceAnalyzer(IRVisitor):
    """
    Given a variable, it may have multiple potential values
    a := 8
    b := 16
       | a + 4
    c := b
       | a + b
    Let value(x) be the Value (with constancy and divisibility) of x.
    Then we have the following equations:
    value(a) = value(8)
    value(b) = merge(value(16), value(a) + value(4))
    value(c) = merge(value(b), value(a) + value(b))

    This class implements an iterative algorithm to solve the above equations.
    """

    def __init__(self):
        super().__init__()
        self.var2value: Dict[Var, Value] = {}
        self.updated: bool = False

    def analyze(self, func: Function):
        # repeat the update until it converges to a fixed point, which is the solution of the coalesce analysis
        while True:
            self.updated = False
            self.visit(func)
            if not self.updated:
                break

    def merge(self, v: Var, value: Optional[Value]):
        if value is None:
            return
        if v not in self.var2value:
            self.var2value[v] = value
            self.updated = True
        else:
            new_value = self.var2value[v].merge(value)
            if new_value != self.var2value[v]:
                self.var2value[v] = new_value
                self.updated = True

    def visit_Function(self, func: Function):
        for arg in func.params:
            if arg.type.is_pointer():
                # we assume that the pointer is aligned to 16 bytes
                self.merge(arg, ScalarValue(divisibility=Divisibility(16)))
            elif arg.type.is_data_type() and arg.type.as_data_type().is_integer():
                self.merge(arg, ScalarValue())
        self.visit(func.body)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            assert isinstance(bind_var, Var)
            if isinstance(bind_var.type, TileType):
                self.merge(bind_var, self.visit(bind_value))
            elif (
                isinstance(bind_var.type, DataType) and bind_var.type.is_integer()
                or isinstance(bind_var.type, PointerType)
            ):
                self.merge(bind_var, self.visit(bind_value))
            else:
                # do nothing for other types
                pass
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, value in zip(stmt.args, stmt.values):
            if isinstance(arg.type, TileType):
                self.merge(arg, self.visit(value))
        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        for arg, let_var in zip(stmt.args, stmt.let_vars):
            if isinstance(arg.type, TileType):
                self.merge(let_var, self.var2value.get(arg, None))
        self.visit(stmt.let_body)

    def visit_PureYieldStmt(self, stmt: PureYieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        for arg, yield_value in zip(for_stmt.args, stmt.yields):
            if isinstance(arg.type, TileType):
                self.merge(arg, self.visit(yield_value))

    def visit_CallTileOp(self, call: CallTileOp):
        return self.visit(call.op)

    def visit_Var(self, e: Var):
        if e in self.var2value:
            return self.var2value[e]

    # scalar expressions

    def visit_Constant(self, e: Constant):
        if e.type.is_data_type() and e.type.is_integer() or e.type.is_pointer():
            return ScalarValue(constant=Constancy(e.value), divisibility=Divisibility(e.value))

    def visit_Add(self, e: Add):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None:
            a = ScalarValue()
        if b is None:
            b = ScalarValue()
        return a + b

    def visit_Multiply(self, e: Multiply):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None:
            a = ScalarValue()
        if b is None:
            b = ScalarValue()
        return a * b

    # tile operators

    def visit_Construct(self, e: Construct):
        affine: Optional[List[Expr]] = affine_decompose(e.value, e.axes)
        if affine is not None:
            weights: List[Optional[ScalarValue]] = [self.visit(w) for w in affine]
            return TileValue([w if w is not None else ScalarValue() for w in weights])

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        from hidet.ir.tile.ops.arthimatic import Add, Sub, Multiply, Mod
        op_dict: Dict[Type[TileOp], Callable] = {
            Add: operator.add,
            Sub: operator.sub,
            Multiply: operator.mul,
            Mod: operator.mod,
        }
        if type(e) in op_dict:
            op = op_dict[type(e)]
            x = self.visit(e.x)
            y = self.visit(e.y)
            if x is None or y is None:
                return None
            else:
                return op(x, y)


class CoalesceMemoryAccessRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.analyzer = CoalesceAnalyzer()

    def __call__(self, func: Function) -> Function:
        self.analyzer.analyze(func)
        return self.visit(func)

    def try_to_get_vectorized_layout(self, ptr: Expr) -> Optional[TileLayout]:
        ptr = self.visit(ptr)
        assert isinstance(ptr, Var)

        if ptr not in self.analyzer.var2value:
            # does not know the constancy and divisibility information for ptr
            return None

        if not ptr.type.is_tile_type():
            # accessing a scalar value
            return None

        tv: TileValue = self.analyzer.var2value[ptr].as_tile_value()
        if len(tv.weights) == 1:
            # scalar tile (e.g., len(shape) == 0)
            return None

        base_divisibility: int = tv.weights[-1].divisibility.v
        offset_constant: Optional[int] = tv.weights[-2].constant.v
        elem_type: PointerType = ptr.type.as_tile_type().type.base_type
        dtype_bytes: int = sizeof(elem_type)
        if offset_constant != 1:
            # it is not contiguous in the last dimension
            return None

        # calculate the largest number of valid vectorized elements
        # in cuda, we can load at most 16 bytes per thread
        vector_elements: int = min(base_divisibility * dtype_bytes, 16) // dtype_bytes
        if vector_elements == 1:
            return None

        ttype: TileType = ptr.type.as_tile_type()
        orig_layout = ttype.layout
        assert isinstance(orig_layout, DistributedLayout)
        return BlockLayout.from_shape(
            shape=ttype.shape,
            num_warps=orig_layout.num_warps,
            size_per_thread=[
                vector_elements if i == len(ttype.shape) - 1 else 1 for i in range(len(ttype.shape))
            ]
        )

    def visit_Load(self, e: Load):
        ptr = self.visit(e.ptr)
        new_layout: Optional[BlockLayout] = self.try_to_get_vectorized_layout(ptr)
        if new_layout is None:
            return super().visit_Load(e)
        else:
            ptr = convert_layout(ptr, new_layout)
            mask: Optional[Expr] = convert_layout(self.visit(e.mask), new_layout) if e.mask is not None else None
            other: Optional[Expr] = convert_layout(self.visit(e.other), new_layout) if e.other is not None else None
            return Load(ptr=ptr, mask=mask, other=other)

    def visit_Store(self, e: Store):
        ptr = self.visit(e.ptr)
        new_layout: Optional[BlockLayout] = self.try_to_get_vectorized_layout(ptr)
        if new_layout is None:
            return super().visit_Store(e)
        else:
            ptr = convert_layout(ptr, new_layout)
            value: Expr = convert_layout(self.visit(e.value), new_layout) if e.value is not None else None
            mask: Optional[Expr] = convert_layout(self.visit(e.mask), new_layout) if e.mask is not None else None
            return Store(ptr=ptr, value=value, mask=mask)


class CoalesceMemoryAccessPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = CoalesceMemoryAccessRewriter()
        func = rewriter(func)
        func = instantiate_layout(func)
        return convert_to_let(func)


def coalesce_memory_access_pass() -> TileFunctionPass:
    return CoalesceMemoryAccessPass()
