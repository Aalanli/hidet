# %%
import triton
import triton.language as tl

import torch


def prune_seq_config(configs, args):
    seq_len = args['seq']
    return [config for config in configs if config.kwargs['block_s'] <= seq_len]

def get_block_s(seq):
    if seq <= 16:
        return 16
    elif seq <= 32:
        return 32
    elif seq <= 64:
        return 64
    raise ValueError(f"seq {seq} not supported")


def generate_configs(zero=False):
    for bm in [32, 64, 128]:
        for bh in [32, 64, 128, 256]:
            for split_h in [1, 2]:
                for nstage in [2, 3, 4, 5]:
                    for nwarp in [4, 8, 16]:
                        pre_hook = lambda x: x['y_ptr'].zero_() if zero else None
                        yield triton.Config({'block_m': bm, 'block_h': bh, 'split_h': split_h}, num_stages=nstage, num_warps=nwarp, pre_hook=pre_hook)

@triton.autotune(
    configs=list(generate_configs()),
    key=['seq', 'h_size', 'm_size'],
)
@triton.jit
def triton_fused_ffn(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    split_h: tl.constexpr
):
    pid = tl.program_id(0) // split_h
    pid_h = tl.program_id(0) % split_h

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m
    w2_ptrs = w1_ptrs + m_size
    y1_lhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    y1_rhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq

    m_mask = m_range[None, :] + pid * block_m < m_size
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_lhs = tl.load(w1_ptrs, mask=mask, other=0)
        y1_lhs += tl.dot(x, w1_lhs, out_dtype=tl.float16)
        w1_rhs = tl.load(w2_ptrs, mask=mask, other=0)
        y1_rhs += tl.dot(x, w1_rhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
        w2_ptrs += 2 * m_size * block_h

    y1 = tl.sigmoid(y1_lhs.to(tl.float32)).to(tl.float16) * y1_lhs * y1_rhs

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :] + pid_h * block_h
    y_ptrs = y_ptr + (s_range[:, None] + pid * seq) * h_size + h_range[None, :] + pid_h * block_h

    s_mask = s_range[:, None] < seq
    m_mask = m_range[:, None] + pid * block_m < m_size

    for k in range(tl.cdiv(h_size, block_h * split_h)):
        h_remaining = h_size - (k * block_h * split_h + pid_h * block_h)
        h_mask = h_range[None, :] < h_remaining
        w2 = tl.load(w2_ptrs, mask=h_mask & m_mask, other=0)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.store(y_ptrs, y, mask=s_mask & h_mask)
        w2_ptrs += block_h * split_h
        y_ptrs += block_h * split_h


# monkey patch to get dynamic buffer
import time, builtins
def cdiv(a, b):
    return (a + b - 1) // b

def get_y(**config):
    return torch.empty([cdiv(config['m_size'], config['block_m']), config['seq'], config['h_size']], dtype=torch.float16, device='cuda')

def run(self, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))
    if len(self.configs) > 1:
        all_args = {**self.nargs, **kwargs}
        _args = []
        for name in self.arg_names:
            if name in all_args:
                _args.append(all_args[name])
        key = tuple(_args[i] for i in self.key_idx)
        if key not in self.cache:
            # prune configs
            pruned_configs = self.prune_configs(kwargs)
            config_args = []
            for config in pruned_configs:
                new_args = list(args)
                new_args[3] = get_y(**config.kwargs, **all_args)
                config_args.append((tuple(new_args), config))
            
            bench_start = time.time()
            timings = {config: self._bench(*new_args, config=config, **kwargs)
                        for (new_args, config) in config_args}
            
            bench_end = time.time()
            self.bench_time = bench_end - bench_start
            self.cache[key] = builtins.min(timings, key=timings.get)
            self.hook(args)
            self.configs_timings = timings
        config = self.cache[key]
    else:
        config = self.configs[0]
    self.best_config = config
    full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
    Y = get_y(**full_nargs)
    new_args = list(args)
    new_args[3] = Y
    self.fn.run(*new_args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)
    return Y

triton_fused_ffn.run = lambda *args, **kwargs: run(triton_fused_ffn, *args, **kwargs)


@triton.autotune(
    configs=list(generate_configs(zero=True)),
    key=['m_size', 'h_size', 'h_size'],
)
@triton.jit
def triton_fused_ffn_atomic(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    split_h: tl.constexpr
):
    pid = tl.program_id(0) // split_h
    pid_h = tl.program_id(0) % split_h

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m
    w2_ptrs = w1_ptrs + m_size
    y1_lhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    y1_rhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq

    m_mask = m_range[None, :] + pid * block_m < m_size
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_lhs = tl.load(w1_ptrs, mask=mask, other=0)
        y1_lhs += tl.dot(x, w1_lhs, out_dtype=tl.float16)
        w1_rhs = tl.load(w2_ptrs, mask=mask, other=0)
        y1_rhs += tl.dot(x, w1_rhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
        w2_ptrs += 2 * m_size * block_h

    y1 = tl.sigmoid(y1_lhs.to(tl.float32)).to(tl.float16) * y1_lhs * y1_rhs

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :] + pid_h * block_h
    y_ptrs = y_ptr + (s_range[:, None]) * h_size + h_range[None, :] + pid_h * block_h

    s_mask = s_range[:, None] < seq
    m_mask = m_range[:, None] + pid * block_m < m_size

    for k in range(tl.cdiv(h_size, block_h * split_h)):
        h_remaining = h_size - (k * block_h * split_h + pid_h * block_h)
        h_mask = h_range[None, :] < h_remaining
        w2 = tl.load(w2_ptrs, mask=h_mask & m_mask, other=0)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.atomic_add(y_ptrs, y, mask=s_mask & h_mask)
        w2_ptrs += block_h * split_h
        y_ptrs += block_h * split_h

def triton_llama_ffn(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)
    
    grid = lambda META: (triton.cdiv(m_size, META['block_m']) * META['split_h'],)
    y = triton_fused_ffn[grid](
        x, w1, w2, None,
        seq,
        h_size,
        m_size,
        block_s=get_block_s(seq),
    )

    return y.sum(0)

def triton_llama_ffn_atomic(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)
    
    y = torch.empty(seq, h_size, dtype=torch.float16, device='cuda')
    grid = lambda META: (triton.cdiv(m_size, META['block_m']) * META['split_h'],)
    triton_fused_ffn_atomic[grid](
        x, w1, w2, y,
        seq,
        h_size,
        m_size,
        block_s=get_block_s(seq),
    )

    return y

def torch_ref(x, w1, w2):
    m_size = w2.size(0)
    x = x @ w1
    x = torch.nn.functional.silu(x[:, :m_size]) * x[:, m_size:]
    y2 = x @ w2
    return y2

def demo_triton():

    seq = 16
    h_size = 4096
    m_size = 12288

    x =  torch.randn(seq, h_size, dtype=torch.float16, device='cuda') / 10
    w1 = torch.randn(h_size, m_size * 2, dtype=torch.float16, device='cuda') / 10
    w2 = torch.randn(m_size, h_size, dtype=torch.float16, device='cuda') / 10

    y1 = torch_ref(x, w1, w2)
    y2 = triton_llama_ffn(x, w1, w2)
    y3 = triton_llama_ffn_atomic(x, w1, w2)
    torch.cuda.synchronize()

    # print(y1)
    # print(y2)

    print(torch.allclose(y1, y2, atol=5e-2, rtol=5e-2))
    print(torch.allclose(y1, y3, atol=5e-2, rtol=5e-2))
    return y1, y2, y3

y1, y2, y3 = demo_triton()
# %%
print((y1 - y3).abs().max())

# %%
for M in [1, 2, 4, 16]:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['D'],  # Argument names to use as an x-axis for the plot
            x_vals=[
                2048, 4096
            ],  # Different possible values for `x_name`
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=['torch_naive', 'triton_fused', 'triton_fused_atomic'],
            # Label name for the lines
            line_names=['torch_naive', 'triton_fused', 'triton_fused_atomic'],
            # Line styles
            styles=[('green', '-'), ('blue', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-')],
            ylabel="ms",  # Label name for the y-axis
            plot_name=f"mlp-performance-M={M}",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )
    def benchmark(D, provider):
        m_size = 3 * D
        a = torch.randn([M, D], dtype=torch.float16, device='cuda')
        w1 = torch.randn([D, m_size * 2], dtype=torch.float16, device='cuda')
        w2 = torch.randn([m_size, D], dtype=torch.float16, device='cuda')

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch_naive':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_ref(a, w1, w2), quantiles=quantiles)
        if provider == 'triton_fused':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_llama_ffn(a, w1, w2), quantiles=quantiles)
        if provider == 'triton_fused_atomic':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_llama_ffn_atomic(a, w1, w2), quantiles=quantiles)
        # if provider == 'triton_default':
        #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: two_triton(a, w1, w2), quantiles=quantiles)        
        # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # return perf(ms), perf(max_ms), perf(min_ms)
        return ms, max_ms, min_ms

    benchmark.run(show_plots=True, print_data=True)

