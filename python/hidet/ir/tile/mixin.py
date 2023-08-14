from .ops.arthimatic import UnaryTileOp, BinaryTileOp
from .ops.creation import Arange
from .ops.memory import Load, Store
from .ops.transform import Broadcast, Reshape, Full
from .ops.convert_layout import ConvertLayout
from .expr import CallTileOp, TileOp


class TileOpMixin:
    def dispatch_CallTileOp(self, e: CallTileOp):
        op: TileOp = e.op
        if isinstance(op, UnaryTileOp):
            op = self.visit_UnaryTileOp(op)
        elif isinstance(op, BinaryTileOp):
            op = self.visit_BinaryTileOp(op)
        elif isinstance(op, Arange):
            op = self.visit_Arange(op)
        elif isinstance(op, Load):
            op = self.visit_Load(op)
        elif isinstance(op, Store):
            op = self.visit_Store(op)
        elif isinstance(op, Broadcast):
            op = self.visit_Broadcast(op)
        elif isinstance(op, Reshape):
            op = self.visit_Reshape(op)
        elif isinstance(op, Full):
            op = self.visit_Full(op)
        elif isinstance(op, ConvertLayout):
            op = self.visit_ConvertLayout(op)
        else:
            raise NotImplementedError()
        if op is e.op:
            return e
        else:
            return op.make_call()

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

    def visit_Reshape(self, e: Reshape):
        raise NotImplementedError()

    def visit_Full(self, e: Full):
        raise NotImplementedError()

    def visit_ConvertLayout(self, e: ConvertLayout):
        raise NotImplementedError()
