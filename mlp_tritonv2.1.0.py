# %%
import torch

import triton
from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl

def init_to_zero(nargs):
    nargs['Y'].zero_()

def gen_configs(init=True):
    for bn in [32, 64, 128, 256]:
        for bk in [32, 64, 128, 256]:
            for bh in [32, 64, 128, 256]:
                for sh in [1, 2, 3, 4]:
                    for ns in [1, 2, 3, 4, 5]:
                        for nw in [1, 2, 4, 8, 16]:
                            if init:
                                yield Config({'BLOCK_N': bn, 'BLOCK_K': bk, 'BLOCK_H': bh, 'SPLIT_H': sh}, num_stages=ns, num_warps=nw, pre_hook=init_to_zero)
                            else:
                                yield Config({'BLOCK_N': bn, 'BLOCK_K': bk, 'BLOCK_H': bh, 'SPLIT_H': sh}, num_stages=ns, num_warps=nw)

@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 1}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 1}, num_stages=5, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 1}, num_stages=3, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 1}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64, 'SPLIT_H': 1}, num_stages=2, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32, 'SPLIT_H': 1}, num_stages=5, num_warps=4, pre_hook=init_to_zero),

        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 2}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 2}, num_stages=5, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 2}, num_stages=3, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 2}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64, 'SPLIT_H': 2}, num_stages=2, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32, 'SPLIT_H': 2}, num_stages=5, num_warps=4, pre_hook=init_to_zero),

        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 3}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 3}, num_stages=5, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 3}, num_stages=3, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 3}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64, 'SPLIT_H': 3}, num_stages=2, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32, 'SPLIT_H': 3}, num_stages=5, num_warps=4, pre_hook=init_to_zero),
    ],
    # configs = list(gen_configs()),
    key=['D_UP', 'D', 'D_DOWN'],
)
@jit
def _mlp_fused_kernel_atomic(X, UP_PROJ, DOWN_PROJ, Y,  # pointers
                     M, D, D_UP, D_DOWN, # shapes
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
                     SPLIT_H: tl.constexpr):
    # mlp
    # X[M, D] x UP_PROJ[D, D_UP] -> X0[M, D_UP]
    # relu X0[M, D_UP] -> X1[M, D_UP]
    # X1[M, D_UP] x DOWN_PROJ[D_UP, D_DOWN] -> Y[M, D_DOWN]
    
    pid = tl.program_id(0) // SPLIT_H
    pid_h = tl.program_id(0) % SPLIT_H
    grid_n = tl.cdiv(D_UP, BLOCK_N)
    pid_n = pid % grid_n 

    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # rbn = tl.max_contiguous(tl.multiple_of(rn % D_UP, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    X = X + (rm[:, None] * D + rk[None, :])
    UP_PROJ = UP_PROJ + (rk[:, None] * D_UP + rn[None, :])
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(D, BLOCK_K)):
        # if EVEN_K:
        #     xs = tl.load(X)
        # else:
        k_remaining = D - k * BLOCK_K
        xs = tl.load(X, mask=(rm[:, None] < M) & (rk[None, :] < k_remaining), other=0)
        
        # if EVEN_M and EVEN_K:
        #     ds = tl.load(UP_PROJ)
        # else:
        ds = tl.load(UP_PROJ, mask=(rk[:, None] < k_remaining) & (rn[None, :] < D_UP), other=0)

        acc += tl.dot(xs, ds)
        X += BLOCK_K
        UP_PROJ += BLOCK_K * D_UP
    
    acc = acc.to(tl.float16)
    # relu
    # acc = tl.where(acc < 0, 0, acc)
    # rematerialize rm and rn to save registers

    rm = tl.arange(0, BLOCK_M)
    rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rh = tl.arange(0, BLOCK_H)

    DOWN_PROJ_ptr = DOWN_PROJ + (rk[:, None] * D_DOWN + rh[None, :] + pid_h * BLOCK_H)
    Y_ptr = Y + (rm[:, None] * D_DOWN + rh[None, :] + pid_h * BLOCK_H)

    for h in range(0, tl.cdiv(D_DOWN, BLOCK_H * SPLIT_H)):
        h_remain = D_DOWN - (pid_h * BLOCK_H + h * BLOCK_H * SPLIT_H)
        hs = tl.load(DOWN_PROJ_ptr, mask=(rk[:, None] < D_UP) & (rh[None, :] < h_remain), other=0)

        res = tl.dot(acc, hs).to(tl.float16)
        tl.atomic_add(Y_ptr, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
        DOWN_PROJ_ptr += BLOCK_H * SPLIT_H
        Y_ptr += BLOCK_H * SPLIT_H
        

@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 1}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 1}, num_stages=5, num_warps=8),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 1}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64, 'SPLIT_H': 1}, num_stages=2, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32, 'SPLIT_H': 1}, num_stages=5, num_warps=4),

        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 2}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 2}, num_stages=5, num_warps=8),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 2}, num_stages=3, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 2}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64, 'SPLIT_H': 2}, num_stages=2, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32, 'SPLIT_H': 2}, num_stages=5, num_warps=4),

        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 3}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 3}, num_stages=5, num_warps=8),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128, 'SPLIT_H': 3}, num_stages=3, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64, 'SPLIT_H': 3}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64, 'SPLIT_H': 3}, num_stages=2, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32, 'SPLIT_H': 3}, num_stages=5, num_warps=4),
    ],
    # configs = list(gen_configs(init=False)),
    key=['D_UP', 'D', 'D_DOWN'],
)
@jit
def _mlp_fused_kernel(X, UP_PROJ, DOWN_PROJ, Y,  # pointers
                     M, D, D_UP, D_DOWN, # shapes
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
                     SPLIT_H: tl.constexpr):
    # mlp
    # X[M, D] x UP_PROJ[D, D_UP] -> X0[M, D_UP]
    # relu X0[M, D_UP] -> X1[M, D_UP]
    # X1[M, D_UP] x DOWN_PROJ[D_UP, D_DOWN] -> Y[M, D_DOWN]

    pid = tl.program_id(0) // SPLIT_H
    pid_h = tl.program_id(0) % SPLIT_H
    grid_n = tl.cdiv(D_UP, BLOCK_N)
    pid_n = pid % grid_n 

    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # rbn = tl.max_contiguous(tl.multiple_of(rn % D_UP, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    X = X + (rm[:, None] * D + rk[None, :])
    UP_PROJ = UP_PROJ + (rk[:, None] * D_UP + rn[None, :])
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(D, BLOCK_K)):
        # if EVEN_K:
        #     xs = tl.load(X)
        # else:
        k_remaining = D - k * BLOCK_K
        xs = tl.load(X, mask=(rm[:, None] < M) & (rk[None, :] < k_remaining), other=0)
        
        # if EVEN_M and EVEN_K:
        #     ds = tl.load(UP_PROJ)
        # else:
        ds = tl.load(UP_PROJ, mask=(rk[:, None] < k_remaining) & (rn[None, :] < D_UP), other=0)

        acc += tl.dot(xs, ds)
        X += BLOCK_K
        UP_PROJ += BLOCK_K * D_UP
    
    acc = acc.to(tl.float16)
    # relu
    # acc = tl.where(acc < 0, 0, acc)
    # rematerialize rm and rn to save registers
    offset_block = pid_n * D_DOWN * M
    rm = tl.arange(0, BLOCK_M)
    rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rh = tl.arange(0, BLOCK_H)

    DOWN_PROJ_ptr = DOWN_PROJ + (rk[:, None] * D_DOWN + rh[None, :] + pid_h * BLOCK_H)
    Y_ptr = Y + (rm[:, None] * D_DOWN + rh[None, :] + offset_block + pid_h * BLOCK_H)

    for h in range(0, tl.cdiv(D_DOWN, BLOCK_H * SPLIT_H)):
        h_remain = D_DOWN - (pid_h * BLOCK_H + h * BLOCK_H * SPLIT_H)
        hs = tl.load(DOWN_PROJ_ptr, mask=(rk[:, None] < D_UP) & (rh[None, :] < h_remain), other=0)

        res = tl.dot(acc, hs).to(tl.float16)
        tl.store(Y_ptr, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
        DOWN_PROJ_ptr += BLOCK_H * SPLIT_H
        Y_ptr += BLOCK_H * SPLIT_H


import time, builtins

def get_y(**config):
    return torch.empty([cdiv(config['D_UP'], config['BLOCK_N']), config['M'], config['D_DOWN']], dtype=torch.float16, device='cuda')

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

_mlp_fused_kernel.run = lambda *args, **kwargs: run(_mlp_fused_kernel, *args, **kwargs)



def fused_mlp_ref(X, UP_PROJ, DOWN_PROJ):
    # x1 = torch.relu(X @ UP_PROJ)
    x1 = X @ UP_PROJ
    return x1 @ DOWN_PROJ

def fused_mlp_atomic(X, UP_PROJ, DOWN_PROJ):
    Y = torch.empty((X.shape[0], DOWN_PROJ.shape[1]), device=X.device, dtype=X.dtype)
    msize = X.shape[0]
    if msize <= 16:
        MSIZE = 16
    elif msize <= 32:
        MSIZE = 32
    elif msize <= 64:
        MSIZE = 64
    else:
        raise ValueError(f"MSIZE {msize} not supported")
    
    D = X.shape[1]
    D_UP = UP_PROJ.shape[1]
    D_DOWN = DOWN_PROJ.shape[1]
    grid = lambda META: (cdiv(D_UP, META['BLOCK_N']) * META['SPLIT_H'],)
    _mlp_fused_kernel_atomic[grid](X, UP_PROJ, DOWN_PROJ, Y, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE)
    return Y

def fused_mlp(X, UP_PROJ, DOWN_PROJ):
    msize = X.shape[0]
    if msize <= 16:
        MSIZE = 16
    elif msize <= 32:
        MSIZE = 32
    elif msize <= 64:
        MSIZE = 64
    else:
        raise ValueError(f"MSIZE {msize} not supported")
    
    D = X.shape[1]
    D_UP = UP_PROJ.shape[1]
    D_DOWN = DOWN_PROJ.shape[1]
    grid = lambda META: (cdiv(D_UP, META['BLOCK_N']) * META['SPLIT_H'],)
    xs = _mlp_fused_kernel[grid](X, UP_PROJ, DOWN_PROJ, None, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE)
    return xs.sum(0)

class FusedMLP(torch.nn.Module):
    def forward(self, X, UP_PROJ, DOWN_PROJ):
        return torch.relu(X @ UP_PROJ) @ DOWN_PROJ

triton_max_autotune = torch.compile(FusedMLP())

from triton.ops.matmul import matmul
def two_triton(X, UP_PROJ, DOWN_PROJ):
    x1 = matmul(X, UP_PROJ)
    return matmul(x1, DOWN_PROJ)

def test_kernels(M, D):
    a = torch.ones((M, D), device='cuda', dtype=torch.float16) / 100
    w1 = torch.ones((D, D * 4), device='cuda', dtype=torch.float16) / 100
    w2 = torch.ones((D * 4, D), device='cuda', dtype=torch.float16) / 100

    y3 = fused_mlp(a, w1, w2)
    y1 = fused_mlp_atomic(a, w1, w2)
    y2 = fused_mlp_ref(a, w1, w2)
    print(y3)
    print(y2)
    print(y1)
    print((y1 - y2).abs().max())
    print((y2 - y3).abs().max())
    print(torch.allclose(y1, y2, atol=1e-1, rtol=1e-1))
    return y3

# test_kernels(1, 4096)

# # %%
# test_kernels(1, 32)
# test_kernels(2, 64)
# test_kernels(4, 128)
print(triton.__version__)
# %%
test_kernels(1, 4096)
test_kernels(4, 4096)
test_kernels(8, 4096)


# %%
for M in [1, 2, 4, 8, 32]:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['D'],  # Argument names to use as an x-axis for the plot
            x_vals=[
                32, 64, 128, 512, 1024, 2048, 4096
            ],  # Different possible values for `x_name`
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=['torch_naive', 'triton_fused', 'triton_fused_atomic', 'triton_default'],
            # Label name for the lines
            line_names=['torch_naive', 'triton_fused', 'triton_fused_atomic', 'triton_default'],
            # Line styles
            styles=[('green', '-'), ('blue', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-')],
            ylabel="ms",  # Label name for the y-axis
            plot_name=f"mlp-performance-M={M}",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )
    def benchmark(D, provider):
        D, D_UP, D_DOWN = D, D * 4, D
        a = torch.randn([M, D], dtype=torch.float16, device='cuda')
        w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
        w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

        quantiles = [0.5, 0.2, 0.8]
        # if provider == 'torch_naive':
        #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp_ref(a, w1, w2), quantiles=quantiles)
        # if provider == 'triton_fused':
        #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp(a, w1, w2), quantiles=quantiles)
        # if provider == 'triton_fused_atomic':
        #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp_atomic(a, w1, w2), quantiles=quantiles)
        # if provider == 'triton_default':
        #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: two_triton(a, w1, w2), quantiles=quantiles)
        if provider == 'torch_naive':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp_ref(a, w1, w2))
        if provider == 'triton_fused':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp(a, w1, w2))
        if provider == 'triton_fused_atomic':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp_atomic(a, w1, w2))
        if provider == 'triton_default':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: two_triton(a, w1, w2))
        
        # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # return perf(ms), perf(max_ms), perf(min_ms)
        return ms, max_ms, min_ms

    benchmark.run(show_plots=True, print_data=True)



# %%
for k, v in _mlp_fused_kernel.cache.items():
    print(k, v)

for k, v in _mlp_fused_kernel_atomic.cache.items():
    print(k, v)

