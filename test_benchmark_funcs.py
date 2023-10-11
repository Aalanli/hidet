# %%
import torch
import hidet
from hidet.utils.benchmark import do_bench
from hidet.graph.ops.matmul.matmul_f16 import matmul_f16

def latency(f, warmup, repeat):
    f()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        f()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    warmup = max(1, int(warmup / estimate_ms))
    repeat = max(1, int(rep / estimate_ms))

    import time    
    for _ in range(warmup):
        f()
    results = []
    for _ in range(repeat):
        hidet.cuda.synchronize()
        t1 = time.time()
        f()
        hidet.cuda.synchronize()
        t2 = time.time()
        results.append((t2 - t1))
    percentiles = torch.quantile(torch.tensor(results), 0.5)
    return percentiles

def latency_2(f, g, warmup, repeat):
    f(*g())
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        f(*g())
    end_event.record()
    args = g()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    warmup = max(1, int(warmup / estimate_ms))
    repeat = max(1, int(rep / estimate_ms))

    start_event = [hidet.cuda.create_event() for i in range(repeat)]
    end_event = [hidet.cuda.create_event() for i in range(repeat)]


    for _ in range(warmup):
        f(*g())
    results = []
    for i in range(repeat):
        hidet.cuda.event_record(start_event[i])
        f(*args)
        hidet.cuda.event_record(end_event[i])
    hidet.cuda.synchronize()
    results = [hidet.cuda.event_elapsed_time(s, e) for s, e in zip(start_event, end_event)]
    percentiles = torch.quantile(torch.tensor(results), 0.5)
    return percentiles


from hidet.graph.ops.quant.matmul import symmetric_quant_matmul


def get_argsi8(m, k, n):
    a = hidet.from_torch(torch.randn(m, k, dtype=torch.float16, device='cuda'))
    b = hidet.from_torch((torch.randn(k, n, dtype=torch.float16, device='cuda') * 10).to(torch.int8))
    scale = hidet.from_torch(torch.randn(n, dtype=torch.float16, device='cuda'))
    return a, b, scale

def get_argsf16(m, k, n):
    a = hidet.from_torch(torch.randn(m, k, dtype=torch.float16, device='cuda'))
    b = hidet.from_torch(torch.randn(k, n, dtype=torch.float16, device='cuda'))
    return a, b


def bench_i8(i, **kwargs):
    m, k, n = i
    sa = hidet.symbol(['m', k], dtype='float16', device='cuda')
    sb = hidet.symbol([k, n], dtype="int8", device='cuda')
    sscale = hidet.symbol([n], dtype='float16', device='cuda')
    ys = symmetric_quant_matmul(sa, sb, sscale, parallel_k_parts=kwargs['pk_parts'])
    if kwargs['pk_parts'] > 1:
        ys = ys.sum(0)
    g = hidet.trace_from(ys, [sa, sb, sscale])
    func = g.build(space=kwargs['space'])
    return lambda a, b, scale: func(a, b, scale)


def bench_fp16(i, **kwargs):
    m, k, n = i
    sa = hidet.symbol(['m', k], dtype='float16', device='cuda')
    sb = hidet.symbol([k, n], dtype="float16", device='cuda')
    ys = matmul_f16(sa, sb, parallel_k_parts=kwargs['pk_parts'])
    if kwargs['pk_parts'] > 1:
        ys = ys.sum(0)
    g = hidet.trace_from(ys, [sa, sb])
    func = g.build(space=kwargs['space'])
    return lambda a, b: func(a, b)


warmup = 25
rep = 100
M, K, N = (128, 11008, 4096)
dataf16 = {'orig-latency': [], 'new-latency': [], 'orig-latency-with-event': []}
datai8 = {'orig-latency': [], 'new-latency': [], 'orig-latency-with-event': []}
for pk in [1, 2, 3, 4, 5]:
    f_f16 = bench_fp16((M, K, N), space=2, pk_parts=pk)
    g_f16 = lambda: get_argsf16(M, K, N)
    arg = g_f16()
    dataf16['orig-latency'].append(latency(lambda: f_f16(*arg), warmup, rep))
    dataf16['new-latency'].append(do_bench(lambda: f_f16(*arg), warmup, rep)[1])
    dataf16['orig-latency-with-event'].append(latency_2(f_f16, g_f16, warmup, rep))


    f_i8 = bench_i8((M, K, N), space=2, pk_parts=pk)
    g_i8 = lambda: get_argsi8(M, K, N)
    arg = g_i8()
    datai8['orig-latency'].append(latency(lambda: f_i8(*arg), warmup, rep))
    datai8['new-latency'].append(do_bench(lambda: f_i8(*arg), warmup, rep)[1])
    datai8['orig-latency-with-event'].append(latency_2(f_i8, g_i8, warmup, rep))


from matplotlib import pyplot as plt

for label, data in dataf16.items():
    plt.plot([1, 2, 3, 4, 5], data, label=label)
plt.legend()
plt.title(f'f16[{M}, {K}] x f16[{K}, {N}]')
plt.xlabel('pk-parts')
plt.ylabel('latency (ms)')
plt.show()

for label, data in datai8.items():
    plt.plot([1, 2, 3, 4, 5], data, label=label)
plt.legend()
plt.title(f'i8[{M}, {K}] x i8[{K}, {N}]')
plt.xlabel('pk-parts')
plt.ylabel('latency (ms)')
plt.show()
# %%
