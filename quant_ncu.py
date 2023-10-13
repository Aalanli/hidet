# %%
import torch
import hidet
from hidet.utils.benchmark import Bench
from hidet.graph.ops.quant.matmul import symmetric_quant_matmul as symmetric_quant_matmul_ref
from quant_matmul import symmetric_quant_matmul
from triton_matmul_experiment import triton_matmul

hidet.option.cache_dir('quant-ncu/outs/cache')
def get_args(M, K, N):
    a = hidet.from_torch(torch.randn(M, K, dtype=torch.float16, device='cuda'))
    b = hidet.from_torch((torch.randn(K, N, dtype=torch.float16, device='cuda') * 10).to(torch.int8))
    scale = hidet.from_torch(torch.randn(N, dtype=torch.float16, device='cuda'))
    return a, b, scale


def bench_packed_quant(i, **kwargs):
    m, k, n = i
    a, b, scale = get_args(m, k, n)

    sa = hidet.symbol(['s', k], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([k, n], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul(sa, sb, sscale, parallel_k_parts=kwargs['pk_parts'])
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)

def bench_ref(i, **kwargs):
    m, k, n = i
    a, b, scale = get_args(m, k, n)

    sa = hidet.symbol(['s', k], dtype=a.dtype, device='cuda')
    sb = hidet.symbol([k, n], dtype=b.dtype, device='cuda')
    sscale = hidet.symbol_like(scale)
    ys = symmetric_quant_matmul_ref(sa, sb, sscale, parallel_k_parts=kwargs['pk_parts'])
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda: func(a, b, scale)


MKN = (8, 4096, 11008)
bench_packed_quant(MKN, space=2, pk_parts=1)()
bench_ref(MKN, space=2, pk_parts=1)()

bench_packed_quant(MKN, space=2, pk_parts=4)()
bench_ref(MKN, space=2, pk_parts=4)()

MKN = (128, 4096 * 4, 4096 * 4)
bench_packed_quant(MKN, space=2, pk_parts=1)()
bench_ref(MKN, space=2, pk_parts=1)()

bench_packed_quant(MKN, space=2, pk_parts=4)()
bench_ref(MKN, space=2, pk_parts=4)()

# y3 = triton_matmul(a.torch(), b.torch(), scale.torch())

