from typing import List, Dict, Type, Union
from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.stmt import Stmt
from hidet.ir.builders import StmtBuilder
from .buffer import Buffer


class TileOpImpl:
    def implement(self, sb: StmtBuilder, op: TileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        raise NotImplementedError()


_registered_implementations: Dict[Type[TileOp], TileOpImpl] = {}


def register_impl(op_cls: Type[TileOp]):
    def decorator(impl: Type[TileOpImpl]):
        _registered_implementations[op_cls] = impl()
        return impl

    return decorator


def implement_tile_op(op: TileOp, args: List[Buffer], output: Buffer) -> Stmt:
    op_cls = type(op)
    if op_cls not in _registered_implementations:
        raise RuntimeError(f"Cannot implement tile op {op_cls}")
    sb = StmtBuilder()
    _registered_implementations[op_cls].implement(sb, op, args, output)
    return sb.finish()
