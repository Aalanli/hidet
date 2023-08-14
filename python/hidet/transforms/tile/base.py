from hidet.ir.module import IRModule
from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass


class TileFunctionPass(FunctionPass):
    def predicate(self, ir_module: IRModule) -> bool:
        # only apply to ir module with cuda tile functions
        return any(func.kind == 'cuda_tile' for func in ir_module.functions.values())

    def process_func(self, func: Function) -> Function:
        if func.kind != 'cuda_tile':
            return func
        return self.process_tile_func(func)

    def process_tile_func(self, func: Function) -> Function:
        raise NotImplementedError()
