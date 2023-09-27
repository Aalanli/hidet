# %%
import torch
import hidet
from hidet.utils.benchmark import Bench
from hidet.graph.ops.quant.matmul import symmetric_quant_matmul as symmetric_quant_matmul_ref
from quant_matmul import symmetric_quant_matmul
from triton_matmul_experiment import triton_matmul

def get_args(s, i):
    a = hidet.from_torch(torch.randn(s, i, dtype=torch.float16, device='cuda'))
    b = hidet.from_torch((torch.randn(i, i, dtype=torch.float16, device='cuda') * 10).to(torch.int8))
    scale = hidet.from_torch(torch.randn(i, dtype=torch.float16, device='cuda'))
    return a, b, scale


def bench_packed_quant(i, **kwargs):
    a, b, scale = get_args(kwargs['s'], i)

    sa = hidet.symbol(['s', i], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([i, i], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul(sa, sb, sscale, parallel_k_parts=1)
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

def bench_ref(i, **kwargs):
    a, b, scale = get_args(kwargs['s'], i)

    sa = hidet.symbol(['s', i], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([i, i], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul_ref(sa, sb, sscale, parallel_k_parts=1)
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

s = 128
d = 256

a, b, scale = get_args(s, d)

bench_packed_quant(d, space=2, s=s)()
bench_ref(d, space=2, s=s)()
y3 = triton_matmul(a.torch(), b.torch(), scale.torch())

s = 4
a, b, scale = get_args(s, d)

bench_packed_quant(d, space=2, s=s)()
bench_ref(d, space=2, s=s)()
y3 = triton_matmul(a.torch(), b.torch(), scale.torch())

