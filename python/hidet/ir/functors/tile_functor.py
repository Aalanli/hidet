from hidet.ir.node import Node
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.ops.creation import Arange, Full
from hidet.ir.tile.ops.memory import Load, Store
from hidet.ir.tile.ops.transform import Broadcast, ExpandDims
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tile.ops.arthimatic import UnaryTileOp, BinaryTileOp
from hidet.ir.tile.ops.reduce import ReduceOp
from hidet.ir.tile.ops.debug import DebugPrint
from hidet.ir.tile.ops.dot import Dot
from hidet.ir.tile.ops.assign import Assign
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class TileFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, TileType):
            return self.visit_TileType(node)
        elif isinstance(node, CallTileOp):
            return self.visit_CallTileOp(node)
        elif isinstance(node, UnaryTileOp):
            return self.visit_UnaryTileOp(node)
        elif isinstance(node, BinaryTileOp):
            return self.visit_BinaryTileOp(node)
        elif isinstance(node, Arange):
            return self.visit_Arange(node)
        elif isinstance(node, Load):
            return self.visit_Load(node)
        elif isinstance(node, Store):
            return self.visit_Store(node)
        elif isinstance(node, Broadcast):
            return self.visit_Broadcast(node)
        elif isinstance(node, Full):
            return self.visit_Full(node)
        elif isinstance(node, ConvertLayout):
            return self.visit_ConvertLayout(node)
        elif isinstance(node, ExpandDims):
            return self.visit_ExpandDims(node)
        elif isinstance(node, DebugPrint):
            return self.visit_DebugPrint(node)
        elif isinstance(node, ReduceOp):
            return self.visit_ReduceOp(node)
        elif isinstance(node, Dot):
            return self.visit_Dot(node)
        elif isinstance(node, Assign):
            return self.visit_Assign(node)
        elif isinstance(node, TileOp):
            raise NotImplementedError(
                'Rewriter for the following tile op is not implemented: \n{}'.format(node.op_name())
            )
        else:
            return NotImplemented

    def visit_CallTileOp(self, call: CallTileOp):
        raise NotImplementedError()

    def visit_TileType(self, t: TileType):
        raise NotImplementedError()

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        raise NotImplementedError()

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        raise NotImplementedError()

    def visit_Arange(self, e: Arange):
        raise NotImplementedError()

    def visit_Load(self, e: Load):
        raise NotImplementedError()

    def visit_Store(self, e: Store):
        raise NotImplementedError()

    def visit_Broadcast(self, e: Broadcast):
        raise NotImplementedError()

    def visit_ExpandDims(self, e: ExpandDims):
        raise NotImplementedError()

    def visit_Full(self, e: Full):
        raise NotImplementedError()

    def visit_ConvertLayout(self, e: ConvertLayout):
        raise NotImplementedError()

    def visit_ReduceOp(self, e: ReduceOp):
        raise NotImplementedError()

    def visit_Dot(self, e: Dot):
        raise NotImplementedError()

    def visit_Assign(self, e: Assign):
        raise NotImplementedError()

    def visit_DebugPrint(self, e: DebugPrint):
        raise NotImplementedError()


class TileVisitor(TileFunctor, BaseVisitor):
    def visit_TileType(self, t: TileType):
        self.visit(t.type)

    def visit_CallTileOp(self, call: CallTileOp):
        self.visit(call.op)

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        self.visit(e.x)

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        self.visit(e.x)
        self.visit(e.y)

    def visit_Arange(self, e: Arange):
        pass

    def visit_Load(self, e: Load):
        self.visit(e.ptr)
        self.visit(e.mask)

    def visit_Store(self, e: Store):
        self.visit(e.ptr)
        self.visit(e.mask)
        self.visit(e.value)

    def visit_Broadcast(self, e: Broadcast):
        self.visit(e.x)

    def visit_ExpandDims(self, e: ExpandDims):
        self.visit(e.x)

    def visit_Full(self, e: Full):
        pass

    def visit_ConvertLayout(self, e: ConvertLayout):
        self.visit(e.x)

    def visit_ReduceOp(self, e: ReduceOp):
        self.visit(e.x)

    def visit_Dot(self, e: Dot):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Assign(self, e: Assign):
        self.visit(e.src)
        self.visit(e.dst)

    def visit_DebugPrint(self, e: DebugPrint):
        self.visit(e.x)


class TileRewriter(TileFunctor, BaseRewriter):
    def visit_TileType(self, t: TileType):
        tp = self.visit(t.type)
        if tp is t.type:
            return t
        else:
            return TileType(tp, shape=t.shape, layout=t.layout)

    def visit_CallTileOp(self, call: CallTileOp):
        op = self.visit(call.op)
        if op is call.op:
            return call
        else:
            return op.make_call()

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        x = self.visit(e.x)
        y = self.visit(e.y)
        if x is e.x and y is e.y:
            return e
        else:
            return e.reforward([x, y])

    def visit_Arange(self, e: Arange):
        return e

    def visit_Load(self, e: Load):
        ptr = self.visit(e.ptr)
        mask = self.visit(e.mask)
        if ptr is e.ptr and mask is e.mask:
            return e
        else:
            return e.reforward([ptr, mask])

    def visit_Store(self, e: Store):
        ptr = self.visit(e.ptr)
        value = self.visit(e.value)
        mask = self.visit(e.mask)
        if ptr is e.ptr and mask is e.mask and value is e.value:
            return e
        else:
            return e.reforward([ptr, value, mask])

    def visit_Broadcast(self, e: Broadcast):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_ExpandDims(self, e: ExpandDims):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_Full(self, e: Full):
        return e

    def visit_ConvertLayout(self, e: ConvertLayout):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_ReduceOp(self, e: ReduceOp):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_Dot(self, e: Dot):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is e.a and b is e.b:
            return e
        else:
            return e.reforward([a, b])

    def visit_Assign(self, e: Assign):
        src = self.visit(e.src)
        dst = self.visit(e.dst)
        if src is e.src and dst is e.dst:
            return e
        else:
            return e.reforward([dst, src])

    def visit_DebugPrint(self, e: DebugPrint):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])
