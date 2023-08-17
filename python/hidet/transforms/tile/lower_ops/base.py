from typing import List, Dict, Type
from hidet.ir.tile.expr import TileOp
from hidet.ir.stmt import Stmt
from hidet.ir.builders import StmtBuilder
from .buffer import Buffer


class TileOpImpl:
    def implement(self, sb: StmtBuilder, args: List[Buffer], output: Buffer):
        raise NotImplementedError()


_registered_implementations: Dict[Type[TileOp], TileOpImpl] = {}


def implement_tile_op(op: TileOp, args: List[Buffer], output: Buffer) -> Stmt:
    op_cls = type(op)
    if op_cls not in _registered_implementations:
        raise RuntimeError(f"Cannot implement tile op {op_cls}")
    sb = StmtBuilder()
    _registered_implementations[op_cls].implement(sb, args, output)
    return sb.finish()
