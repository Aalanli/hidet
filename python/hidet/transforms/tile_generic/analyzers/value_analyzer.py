from typing import Any, Optional, List, Dict
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.tile.type import TileType
from hidet.ir.tile.ops import Construct
from hidet.ir.tile.ops.arthimatic import Add
from hidet.ir.functors import IRVisitor


class Constancy:
    def __init__(self, is_constant: bool, known_value: Optional[Any]):
        self.is_constant: bool = is_constant
        self.known_value: Optional[Any] = known_value


class Divisibility:
    def __init__(self, v: int = 1):
        self.v: int = v


class Continuity:
    def __init__(self, v: int = 1):
        self.v: int = v


class ValueInfo:
    def as_tensor_value(self):
        assert isinstance(self, TensorValueInfo)
        return self

    def as_scalar_value(self):
        assert isinstance(self, ScalarValueInfo)
        return self


class ScalarValueInfo:
    def __init__(self, constancy: Constancy, divisibility: Divisibility):
        self.constancy: Constancy = constancy
        self.divisibility: Divisibility = divisibility


class TensorValueInfo:
    def __init__(self, constancy: List[Constancy], divisibility: List[Divisibility], continuity: List[Continuity]):
        self.constancy: List[Constancy] = constancy
        self.divisibility: List[Divisibility] = divisibility
        self.continuity: List[Continuity] = continuity


class ValueAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.var2value: Dict[Var, ValueInfo] = {}
        self.updated: bool = False

    def merge(self, v: Var, value: Optional[ValueInfo]):
        pass

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

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        for arg, yield_value in zip(for_stmt.args, stmt.values):
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




