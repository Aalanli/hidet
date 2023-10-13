# %%
import torch
import hidet
from hidet.utils.benchmark import Bench
from hidet.graph.ops.quant.matmul import symmetric_quant_matmul as symmetric_quant_matmul_ref
from quant_matmul import symmetric_quant_matmul
from triton_matmul_experiment import triton_matmul

hidet.option.debug_cache_tuning()
def get_args(s, i):
    a = hidet.from_torch(torch.randn(s, i, dtype=torch.float16, device='cuda'))
    b = hidet.from_torch((torch.randn(i, i, dtype=torch.float16, device='cuda') * 10).to(torch.int8))
    scale = hidet.from_torch(torch.randn(i, dtype=torch.float16, device='cuda'))
    return a, b, scale

a, b, scale = get_args(128, 128)

y1 = symmetric_quant_matmul(a, b, scale, parallel_k_parts=1)
y2 = symmetric_quant_matmul_ref(a, b, scale, parallel_k_parts=1)
y3 = triton_matmul(a.torch(), b.torch(), scale.torch())

print(torch.allclose(y1.torch(), y2.torch()))
print(torch.allclose(y1.torch(), y3, atol=1e-2, rtol=1e-2))
print((y1.torch() - y3).abs().max())

from hidet.utils.benchmark import Bench

def bench_ref(i, **kwargs):
    a, b, scale = get_args(kwargs['s'], i)

    sa = hidet.symbol(['s', i], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([i, i], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul_ref(sa, sb, sscale, parallel_k_parts=kwargs['pk_parts'])
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

def bench_packed_quant(i, **kwargs):
    a, b, scale = get_args(kwargs['s'], i)

    sa = hidet.symbol(['s', i], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([i, i], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul(sa, sb, sscale, parallel_k_parts=kwargs['pk_parts'])
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

def bench_triton(i, **kwargs):
    a, b, scale = get_args(kwargs['s'], i)
    a = a.torch()
    b = b.torch()
    scale = scale.torch()
    return lambda: triton_matmul(a, b, scale)

def torch_orig(i, **kwargs):
    a = torch.randn(kwargs['s'], i, dtype=torch.float16, device='cuda')
    b = torch.randn(i, i, dtype=torch.float16, device='cuda')

    return lambda: a @ b

S = 128
bn = Bench(x_name='C', x_vals=[1024, 2048, 4096, 4096 * 2, 4096 * 4], space=2, s=S, pk_parts=8)
bn.bench(bench_ref)
bn.bench(bench_packed_quant)
# bn.bench(bench_triton)
# bn.bench(torch_orig)
# bn.measure_flops(lambda config, c: torch.finfo(config.dtype).bits // 8 * c**2)
data = bn.run()
data.show_plot(title=f'f16[{S}, C] x i8[C, C]')
data.print_data()

# %%
from triton_matmul_experiment import _kernel

for k, v in _kernel.cache.items():
    print(k)
    print(v)


# %%
from hidet.utils.benchmark import Bench

def get_args2(m, k, n):
    a = hidet.from_torch(torch.zeros(m, k, dtype=torch.float16, device='cuda'))
    b = hidet.from_torch((torch.zeros(k, n, dtype=torch.float16, device='cuda') * 10).to(torch.int8))
    scale = hidet.from_torch(torch.zeros(n, dtype=torch.float16, device='cuda'))
    return a, b, scale

def bench_ref(i, **kwargs):
    m, k, n = i
    a, b, scale = get_args2(m, k, n)

    sa = hidet.symbol(['s', k], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([k, n], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul_ref(sa, sb, sscale, parallel_k_parts=4)
    ys = ys.sum(0)
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

def bench_packed_quant(i, **kwargs):
    m, k, n = i
    a, b, scale = get_args2(m, k, n)

    sa = hidet.symbol(['s', k], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([k, n], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    # if k <= 4096:
    #     ys = symmetric_quant_matmul(sa, sb, sscale, parallel_k_parts=1)
    # else:
    ys = symmetric_quant_matmul(sa, sb, sscale, parallel_k_parts=4)
    ys = ys.sum(0)
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

def bench_triton(i, **kwargs):
    m, k, n = i
    a, b, scale = get_args2(m, k, n)
    a = a.torch()
    b = b.torch()
    scale = scale.torch()
    return lambda: triton_matmul(a, b, scale)

def torch_orig(i, **kwargs):
    m, k, n = i
    a = torch.randn(m, k, dtype=torch.float16, device='cuda')
    b = torch.randn(k, n, dtype=torch.float16, device='cuda')

    return lambda: a @ b

S = 128
bn = Bench(x_name='C', x_vals=[(S, 4096, 11008), (S, 11008, 4096), (S, 4096, 4096), (S, 4096, 32000)], space=2)
bn.bench(bench_ref)
bn.bench(bench_packed_quant)
# bn.bench(bench_triton)
# bn.bench(torch_orig)
# bn.measure_flops(lambda config, c: torch.finfo(config.dtype).bits // 8 * c**2)
data = bn.run()
data.show_plot(title=f'f16 x i8')
data.print_data()
