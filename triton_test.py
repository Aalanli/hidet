# %%
import triton
from triton import language as tl
from triton import jit

import torch

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=1, num_stages=1)
#     ], key=[]
# )
@jit
def mul(x, y):
    ix = tl.arange(0, 32)
    ix = ix[:, None] * 32 + ix[None, :]
    x = tl.load(x + ix)
    xh = x * 2
    tl.store(y + ix, xh)

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=1, num_stages=1)
#     ], key=[]
# )
@jit
def mul2(x, y, k):
    for i in range(32):
        ix = tl.arange(0, 32) + i * 32
        xh = tl.load(x + ix)
        xh = xh * 2
        tl.store(y + ix, xh)


# signature = '*fp32,*fp32'

# from triton.compiler import *
# import triton._C.libtriton.triton as _triton
# kwargs = dict()
# capability = kwargs.get("cc", None)
# if capability is None:
#     device = torch.cuda.current_device()
#     capability = torch.cuda.get_device_capability(device)
#     capability = capability[0] * 10 + capability[1]
# # we get the kernel, i.e. the first function generated in the module
# # if fn is not a JITFunction, then it
# # has to be a path to a file
# context = _triton.ir.context()
# asm = dict()
# constants = kwargs.get("constants", dict())
# num_warps = kwargs.get("num_warps", 1)
# num_stages = kwargs.get("num_stages", 3 if capability >= 75 else 2)
# extern_libs = kwargs.get("extern_libs", dict())
# fn = mul
# configs = [instance_descriptor()]

# # build compilation stages
# stages = {
#     "ast": (lambda path: fn, None),
#     "ttir": (lambda path: parse_mlir_module(path, context),
#                 lambda src: ast_to_ttir(src, signature, configs[0], constants)),
#     "ttgir": (lambda path: parse_mlir_module(path, context),
#                 lambda src: ttir_to_ttgir(src, num_warps, num_stages, capability)),
#     "llir": (lambda path: Path(path).read_text(),
#                 lambda src: ttgir_to_llir(src, extern_libs, capability)),
#     "ptx": (lambda path: Path(path).read_text(),
#             lambda src: llir_to_ptx(src, capability)),
#     "cubin": (lambda path: Path(path).read_bytes(),
#                 lambda src: ptx_to_cubin(src, capability))
# }

# ttir = stages['ttir'][1](fn)
# print(ttir)
# ttgir = stages['ttgir'][1](ttir)
# print(ttgir)
# llir = stages['llir'][1](ttgir)
# ptx = stages['ptx'][1](llir)
# print(ptx)

# %%

import hidet
from hidet.lang import attrs
from hidet.lang.cuda import blockIdx, threadIdx

from hidet.ir.module import IRModule
from hidet.ir import primitives as prim
from hidet.ir.expr import is_constant
from hidet.ir.stmt import Stmt, AssignStmt
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from hidet.graph.ops.utils import Task, TensorNode, compute, reduce

# class TestTask(Task):
#     def __init__(self, x: TensorNode):

with hidet.script_module() as module:
    @hidet.script
    def mul_kernel(xs: hidet.float32[32, 32], ys: hidet.float32[32, 32]):
        attrs.cuda.block_dim = 32
        attrs.cuda.grid_dim = 1
        for i in range(32):
            x = xs[i, threadIdx.x]
            y = x * 2
            ys[i, threadIdx.x] = y

ir_module = module.ir_module()
f1 = ir_module.build()

with hidet.script_module() as module:
    @hidet.script
    def mul_kernel2(xs: hidet.float32[32, 32], ys: hidet.float32[32, 32], k: hidet.int32):
        attrs.cuda.block_dim = 32
        attrs.cuda.grid_dim = 1
        for i in range(k):
            x = xs[i, threadIdx.x]
            y = x * 2
            ys[i, threadIdx.x] = y

ir_module = module.ir_module()
f2 = ir_module.build()

x = hidet.ones((32, 32), dtype='float32', device='cuda')
y = hidet.zeros((32, 32), dtype='float32', device='cuda')

f1(x, y)
f2(x, y)

x = torch.ones((32, 32), dtype=torch.float32, device='cuda')
y = torch.zeros((32, 32), dtype=torch.float32, device='cuda')
mul.run(x, y, grid=(1,), num_warps=1, num_stages=1)
mul2.run(x, y, grid=(1,), num_warps=1, num_stages=1)