from typing import Union, Optional
from hidet.ir.expr import Var
from hidet.ir.type import BaseType, PointerType, DataType, void
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.tile.expr import TileOp


class Load(TileOp):
    def __init__(self, ptr: Var, mask: Optional[Var] = None):
        super().__init__()
        self.ptr: Var = ptr
        self.mask: Optional[Var] = mask

        self.args = [ptr] + ([mask] if mask is not None else [])

    @staticmethod
    def _get_loaded_type(ptr_type: PointerType):
        ret_type = ptr_type.base_type
        if isinstance(ret_type, (PointerType, DataType)):
            return ret_type
        else:
            raise RuntimeError(f"Invalid type of Load: {ret_type}")

    def infer_type(self) -> BaseType:
        ptr_type = self.ptr.type
        if isinstance(ptr_type, PointerType):
            return self._get_loaded_type(ptr_type)
        elif isinstance(ptr_type, TileType):
            assert isinstance(ptr_type.type, PointerType)
            elem_type = self._get_loaded_type(ptr_type.type)
            return tile_type(elem_type, ptr_type.shape, ptr_type.layout)
        else:
            assert False


class Store(TileOp):
    def __init__(self, ptr: Var, value: Var, mask: Optional[Var] = None):
        super().__init__()
        self.ptr: Var = ptr
        self.value: Var = value
        self.mask: Optional[Var] = mask

        self.args = [ptr, value] + ([mask] if mask is not None else [])

    def infer_type(self) -> BaseType:
        return void
