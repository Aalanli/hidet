from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import CallTileOp
from hidet.utils import same_list
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class TileFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, TileType):
            return self.visit_TileType(node)
        elif isinstance(node, CallTileOp):
            return self.visit_CallTileOp(node)
        else:
            return NotImplemented

    def visit_TileType(self, t: TileType):
        raise NotImplementedError()

    def visit_CallTileOp(self, call: CallTileOp):
        raise NotImplementedError()


class TileVisitor(TileFunctor, BaseVisitor):
    def visit_TileType(self, t: TileType):
        self.visit(t.type)

    def visit_CallTileOp(self, call: CallTileOp):
        self.visit(call.op.args)


class TileRewriter(TileFunctor, BaseRewriter):
    def visit_TileType(self, t: TileType):
        tp = self.visit(t.type)
        if tp is t.type:
            return t
        else:
            return TileType(tp, shape=t.shape, layout=t.layout)

    def visit_CallTileOp(self, call: CallTileOp):
        args = self.visit(call.op.args)
        if same_list(args, call.op.args):
            return call
        else:
            return call.op.reforward(args, call.op.attrs).make_call()
