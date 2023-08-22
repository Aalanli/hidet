from typing import Type, Dict, Union, List, Tuple, Optional
from hidet.ir.expr import Expr, Var, Constant
from hidet.ir.stmt import LetStmt, DeclareStmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tile.ops import Full, Construct, Arange, BinaryTileOp, Dot, ConvertLayout
from hidet.ir.tile.ops.arthimatic import Add, Multiply, Equal, NotEqual
from hidet.utils import same_list


class Pattern:
    pass


class ScalarPlaceholder(Pattern):
    pass


class TilePattern(Pattern):
    pass


class TilePlaceholder(TilePattern):
    pass


class TileOpPattern(TilePattern):
    def __init__(self, op_cls: Type[TileOp], *args: Pattern):
        self.op_cls: Type[TileOp] = op_cls
        self.args: List[Pattern] = list(args)


class Transform:
    @staticmethod
    def any_scalar_expr() -> ScalarPlaceholder:
        return ScalarPlaceholder()

    @staticmethod
    def any_tile() -> TilePattern:
        return TilePlaceholder()

    @staticmethod
    def full():
        return TileOpPattern(Full)

    @staticmethod
    def arange():
        return TileOpPattern(Arange)

    @staticmethod
    def construct(value=None):
        if value is None:
            value = Transform.any_scalar_expr()
        return TileOpPattern(Construct, value)

    @staticmethod
    def convert_layout(x: TilePattern):
        return TileOpPattern(ConvertLayout, x)

    @staticmethod
    def add(lhs: TilePattern, rhs: TilePattern):
        return TileOpPattern(Add, lhs, rhs)

    @staticmethod
    def dot(a: TilePattern, b: TilePattern, c: TilePattern):
        return TileOpPattern(Dot, a, b, c)

    @staticmethod
    def is_zero(e: Expr, var2call: Dict[Var, CallTileOp]) -> bool:
        if isinstance(e, Var) and e in var2call:
            e = var2call[e]
        if isinstance(e, CallTileOp):
            if isinstance(e.op, Full):
                v = e.op.value
            elif isinstance(e.op, Construct):
                if isinstance(e.op.value, Constant):
                    v = e.op.value
                else:
                    return False
            else:
                return False
            return isinstance(v, Constant) and v.value == v.type.zero.value
        else:
            return False

    @staticmethod
    def get_tile_op(e: Pattern, matched: Dict[Pattern, Expr]):
        call_tile_op = matched[e]
        assert isinstance(call_tile_op, CallTileOp)
        return call_tile_op.op

    def source(self) -> TilePattern:
        raise NotImplementedError()

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[CallTileOp]:
        raise NotImplementedError()


class Matcher:
    def __init__(self, var2value: Dict[Var, Union[CallTileOp, Var]]):
        self.var2value: Dict[Var, Union[Var, CallTileOp]] = var2value
        self.match_dict: Dict[Pattern, Expr] = {}
        self.type_infer = TypeInfer()

    def match(self, pattern: Pattern, target: Expr) -> bool:
        if pattern in self.match_dict:
            if target is self.match_dict[pattern]:
                return True
            else:
                return False
        if isinstance(pattern, ScalarPlaceholder):
            return self.match_ScalarPlaceholder(pattern, target)
        elif isinstance(pattern, TilePlaceholder):
            return self.match_TilePlaceholder(pattern, target)
        elif isinstance(pattern, TileOpPattern):
            return self.match_TileOpPattern(pattern, target)
        else:
            raise NotImplementedError()

    def match_ScalarPlaceholder(self, pattern: ScalarPlaceholder, target: Expr) -> bool:
        tp = self.type_infer(target)
        if isinstance(tp, TileType):
            return False
        else:
            self.match_dict[pattern] = target
            return True

    def match_TilePlaceholder(self, pattern: TilePlaceholder, target: Expr) -> bool:
        if isinstance(target, Var) and isinstance(target.type, TileType) or isinstance(target, CallTileOp):
            self.match_dict[pattern] = target
            return True
        else:
            return False

    def match_TileOpPattern(self, pattern: TileOpPattern, target: Expr) -> bool:
        if isinstance(target, Var) and target in self.var2value:
            target = self.var2value[target]
            # fall through to next if
        if isinstance(target, CallTileOp):
            op_cls = type(target.op)
            if not issubclass(op_cls, pattern.op_cls):
                return False
            if len(target.op.args) != len(pattern.args):
                return False

            is_commutative_binary_op = (
                issubclass(op_cls, BinaryTileOp)
                and op_cls in [Add, Multiply, Equal, NotEqual]
            )
            if is_commutative_binary_op:
                saved_match_dict = self.match_dict.copy()
                pa, pb = pattern.args
                ta, tb = target.op.args
                if self.match(pa, ta) and self.match(pb, tb):
                    self.match_dict[pattern] = target
                    return True
                else:
                    self.match_dict = saved_match_dict.copy()
                    if self.match(pa, tb) and self.match(pb, ta):
                        self.match_dict[pattern] = target
                        return True
                    else:
                        return False
            else:
                for pattern_arg, target_arg in zip(pattern.args, target.op.args):
                    if not self.match(pattern_arg, target_arg):
                        return False
            self.match_dict[pattern] = target
            return True
        else:
            return False


class ApplyTransformRewriter(IRRewriter):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.var2value: Dict[Var, Union[CallTileOp, Var]] = {}
        self.transforms: List[Transform] = transforms
        self.type_infer = TypeInfer()

    def visit_Function(self, func: Function):
        self.var2value.clear()
        return super().visit_Function(func)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(bind_value)
            bind_values.append(bind_value)
            self.var2value[bind_var] = bind_value
        body = self.visit(stmt.body)
        if same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)

    def visit_CallTileOp(self, call: CallTileOp):
        call = super().visit_CallTileOp(call)

        for transform in self.transforms:
            matcher = Matcher(self.var2value)
            if matcher.match(transform.source(), call):
                updated_call = transform.target(matcher.match_dict, self.var2value)
                if updated_call is not None:
                    return updated_call
        return call


def apply_transforms(node: Union[IRModule, Function], transforms: List[Transform]):
    rewriter = ApplyTransformRewriter(transforms)
    return rewriter.visit(node)
