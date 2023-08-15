from typing import Union, Optional, List
from hidet.ir.expr import Expr
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp, call_tile_op


class Load(TileOp):
    def __init__(self, ptr: Expr, mask: Optional[Expr] = None):
        super().__init__()
        self.ptr: Expr = ptr
        self.mask: Optional[Expr] = mask

        self.args = [ptr] + ([mask] if mask is not None else [])

    @staticmethod
    def _get_loaded_type(ptr_type: PointerType):
        ret_type = ptr_type.base_type
        if isinstance(ret_type, (PointerType, DataType)):
            return ret_type
        else:
            raise RuntimeError(f"Invalid type of Load: {ret_type}")

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:

        ptr_type = arg_types[0]
        if isinstance(ptr_type, PointerType):
            return self._get_loaded_type(ptr_type)
        elif isinstance(ptr_type, TileType):
            assert isinstance(ptr_type.type, PointerType)
            elem_type = self._get_loaded_type(ptr_type.type)
            return tile_type(elem_type, ptr_type.shape, ptr_type.layout)
        else:
            assert False


class Store(TileOp):
    def __init__(self, ptr: Expr, value: Expr, mask: Optional[Expr] = None):
        super().__init__()
        self.ptr: Expr = ptr
        self.value: Expr = value
        self.mask: Optional[Expr] = mask

        self.args = [ptr, value] + ([mask] if mask is not None else [])

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:

        return void


def load(ptr: Expr, mask: Optional[Expr] = None):
    return Load(ptr, mask).make_call()


def store(ptr: Expr, value: Expr, mask: Optional[Expr] = None):
    return Store(ptr, value, mask).make_call()
