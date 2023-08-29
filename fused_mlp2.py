# # %%
# import torch

# from triton import Config, autotune, cdiv, heuristics, jit
# from triton import language as tl


# def init_to_zero(name):
#     return lambda nargs: nargs[name].zero_()

# def gen_configs():
#     configs = []
#     for block_n in [32, 64, 128, 256]:
#         for block_k in [32, 64, 128, 256]:
#             for block_h in [32, 64, 128, 256]:
#                 for stages in [2, 3, 4, 5]:
#                     for warps in [2, 4, 8]:
#                         configs.append(Config({
#                             'BLOCK_N': block_n,
#                             'BLOCK_K': block_k,
#                             'BLOCK_H': block_h,
#                         }, num_stages=stages, num_warps=warps, pre_hook=init_to_zero('Y')))
#     return configs

# @autotune(
#     configs=[
#         Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_stages=4, num_warps=8, pre_hook=init_to_zero('Y')),
#         Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128}, num_stages=5, num_warps=8, pre_hook=init_to_zero('Y')),
#         Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128}, num_stages=3, num_warps=8, pre_hook=init_to_zero('Y')),
#         Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_stages=4, num_warps=8, pre_hook=init_to_zero('Y')),
#         Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64}, num_stages=2, num_warps=8, pre_hook=init_to_zero('Y')),
#     ],
#     key=['D_UP', 'D', 'D_DOWN'],
# )
# @heuristics({
#     'EVEN_KH': lambda args: args['D'] % (args['BLOCK_K']) == 0 and args['D_DOWN'] % (args['BLOCK_H']) == 0,
# })
# @jit
# def _mlp_fused_kernel(X, UP_PROJ, DOWN_PROJ, Y,  # pointers
#                      M, D, D_UP, D_DOWN, # shapes
#                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
#                      EVEN_M: tl.constexpr, EVEN_KH: tl.constexpr):
#     # mlp
#     # X[M, D] x UP_PROJ[D, D_UP] -> X0[M, D_UP]
#     # relu X0[M, D_UP] -> X1[M, D_UP]
#     # X1[M, D_UP] x DOWN_PROJ[D_UP, D_DOWN] -> Y[M, D_DOWN]

#     pid = tl.program_id(0)
#     grid_n = tl.cdiv(D_UP, BLOCK_N)
#     pid_n = pid % grid_n 

#     rm = tl.arange(0, BLOCK_M)
#     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     # rbn = tl.max_contiguous(tl.multiple_of(rn % D_UP, BLOCK_N), BLOCK_N)
#     rk = tl.arange(0, BLOCK_K)
#     # pointers
#     X = X + (rm[:, None] * D + rk[None, :])
#     UP_PROJ = UP_PROJ + (rk[:, None] * D_UP + rn[None, :])
    
#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

#     for k in range(0, tl.cdiv(D, BLOCK_K)):
#         if EVEN_M and EVEN_KH:
#             xs = tl.load(X)
#             ds = tl.load(UP_PROJ)
#         else:
#             k_remaining = D - k * BLOCK_K
#             xs = tl.load(X, mask=(rm[:, None] < M) & (rk[None, :] < k_remaining), other=0)
#             ds = tl.load(UP_PROJ, mask=(rk[:, None] < k_remaining) & (rn[None, :] < D_UP), other=0)

#         acc += tl.dot(xs, ds, out_dtype=tl.float16)
#         X += BLOCK_K
#         UP_PROJ += BLOCK_K * D_UP
#     # relu
#     acc = tl.where(acc < 0, 0, acc)
#     # rematerialize rm and rn to save registers
#     rm = tl.arange(0, BLOCK_M)
#     rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     rh = tl.arange(0, BLOCK_H)


#     block_offset = tl.program_id(0) % tl.cdiv(D_DOWN, BLOCK_H)
#     for h in range(0, tl.cdiv(D_DOWN, BLOCK_H)):
#         block_swizzle = (block_offset + h) % tl.cdiv(D_DOWN, BLOCK_H)

#         rh_swizzle = rh + block_swizzle * BLOCK_H
#         Y_ptr = Y + (rm[:, None] * D_DOWN + rh_swizzle[None, :])
#         DOWN_PROJ_ptr = DOWN_PROJ + (rk[:, None] * D_DOWN + rh_swizzle[None, :])
#         if EVEN_M and EVEN_KH:
#             hs = tl.load(DOWN_PROJ_ptr)
#         else:
#             hs = tl.load(DOWN_PROJ_ptr, mask=(rk[:, None] < D_UP) & (rh_swizzle[None, :] < D_DOWN), other=0)

#         res = tl.dot(acc, hs, out_dtype=tl.float16)
#         if EVEN_M and EVEN_KH:
#             tl.atomic_add(Y_ptr, res)
#         else:
#             tl.atomic_add(Y_ptr, res, mask=(rm[:, None] < M) & (rh_swizzle[None, :] < D_DOWN))


# def fused_mlp_ref(X, UP_PROJ, DOWN_PROJ):
#     x1 = torch.relu(X @ UP_PROJ)
#     return x1 @ DOWN_PROJ

# def fused_mlp(X, UP_PROJ, DOWN_PROJ):
#     Y = torch.empty((X.shape[0], DOWN_PROJ.shape[1]), device=X.device, dtype=X.dtype)
#     msize = X.shape[0]
#     if msize <= 32:
#         MSIZE = 32
#     elif msize <= 64:
#         MSIZE = 64
#     else:
#         raise ValueError(f"MSIZE {msize} not supported")
    
#     D = X.shape[1]
#     D_UP = UP_PROJ.shape[1]
#     D_DOWN = DOWN_PROJ.shape[1]
#     grid = lambda META: (cdiv(D_UP, META['BLOCK_N']),)
#     _mlp_fused_kernel[grid](X, UP_PROJ, DOWN_PROJ, Y, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE, EVEN_M = msize == MSIZE)
#     return Y

# class FusedMLP(torch.nn.Module):
#     def forward(self, X, UP_PROJ, DOWN_PROJ):
#         return fused_mlp_ref(X, UP_PROJ, DOWN_PROJ)

# triton_max_autotune = torch.compile(FusedMLP())

# # a = torch.randn((32, 32), device='cuda', dtype=torch.float16)
# # w1 = torch.randn((32, 128), device='cuda', dtype=torch.float16)
# # w2 = torch.randn((128, 32), device='cuda', dtype=torch.float16)

# # y1 = fused_mlp(a, w1, w2)
# # y2 = fused_mlp_ref(a, w1, w2)
# # print((y1 - y2).abs().max())

# from hidet.utils.benchmark import Bench


# def torch_naive(C, **kwargs):
#     M = kwargs['M']
#     D, D_UP, D_DOWN = C, C * 4, C
#     a = torch.randn([M, D], dtype=torch.float16, device='cuda')
#     w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
#     w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')
#     return lambda: fused_mlp_ref(a, w1, w2)

# def triton_fused(C, **kwargs):
#     M = kwargs['M']
#     D, D_UP, D_DOWN = C, C * 4, C
#     a = torch.randn([M, D], dtype=torch.float16, device='cuda')
#     w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
#     w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

#     return lambda: fused_mlp(a, w1, w2)

# def triton_default(C, **kwargs):
#     M = kwargs['M']
#     D, D_UP, D_DOWN = C, C * 4, C
#     a = torch.randn([M, D], dtype=torch.float16, device='cuda')
#     w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
#     w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

#     return lambda: triton_max_autotune.forward(a, w1, w2)

# bn = Bench(x_vals=[64, 128, 512, 1024, 4096], x_name='C', M=32)
# # bn.measure_flops(lambda C: C**2 * 2)
# bn.bench(torch_naive)
# bn.bench(triton_fused)
# # bn.bench(triton_default)

# data = bn.run()
# data.show_plot(title='M=32')


# %%
import torch

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl

@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128}, num_stages=5, num_warps=8),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128}, num_stages=3, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_stages=4, num_warps=8),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64}, num_stages=2, num_warps=8),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=5, num_warps=4), #32 4096 16384 / 32 16384 4096

    ],
    key=['D_UP', 'D', 'D_DOWN'],
)
@jit
def _mlp_fused_kernel(X, UP_PROJ, DOWN_PROJ, Y,  # pointers
                     M, D, D_UP, D_DOWN, # shapes
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
                     SWIZZLE_BLOCK: tl.constexpr):
    # mlp
    # X[M, D] x UP_PROJ[D, D_UP] -> X0[M, D_UP]
    # relu X0[M, D_UP] -> X1[M, D_UP]
    # X1[M, D_UP] x DOWN_PROJ[D_UP, D_DOWN] -> Y[M, D_DOWN]

    pid = tl.program_id(0)
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
    if SWIZZLE_BLOCK:
        block_swizzle_offset = tl.program_id(0) % tl.cdiv(D_DOWN, BLOCK_H)

        rm = tl.arange(0, BLOCK_M)
        rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rh = tl.arange(0, BLOCK_H)

        DOWN_PROJ_ptr = DOWN_PROJ + (rk[:, None] * D_DOWN + rh[None, :])
        Y_ptr = Y + (rm[:, None] * D_DOWN + rh[None, :])

        for h in range(0, tl.cdiv(D_DOWN, BLOCK_H)):
            hi = (h + block_swizzle_offset) % tl.cdiv(D_DOWN, BLOCK_H)
            h_remain = D_DOWN - hi * BLOCK_H
            hs = tl.load(DOWN_PROJ_ptr + hi * BLOCK_H, mask=(rk[:, None] < D_UP) & (rh[None, :] < h_remain), other=0)

            res = tl.dot(acc, hs).to(tl.float16)
            tl.atomic_add(Y_ptr + hi * BLOCK_H, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
    else:
        offset_block = pid_n * D_DOWN * M
        rm = tl.arange(0, BLOCK_M)
        rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rh = tl.arange(0, BLOCK_H)

        DOWN_PROJ_ptr = DOWN_PROJ + (rk[:, None] * D_DOWN + rh[None, :])
        Y_ptr = Y + (rm[:, None] * D_DOWN + rh[None, :] + offset_block)

        for h in range(0, tl.cdiv(D_DOWN, BLOCK_H)):
            h_remain = D_DOWN - h * BLOCK_H
            hs = tl.load(DOWN_PROJ_ptr, mask=(rk[:, None] < D_UP) & (rh[None, :] < h_remain), other=0)

            res = tl.dot(acc, hs).to(tl.float16)
            tl.store(Y_ptr, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
            DOWN_PROJ_ptr += BLOCK_H
            Y_ptr += BLOCK_H

import time
import builtins

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
    self.fn.run(*new_args, num_warps=config.num_warps, num_stages=config.num_stages,
                        num_ctas=config.num_ctas,
                        enable_warp_specialization=config.enable_warp_specialization, **kwargs, **config.kwargs)
    return Y

_mlp_fused_kernel.run = lambda *args, **kwargs: run(_mlp_fused_kernel, *args, **kwargs)

def fused_mlp_ref(X, UP_PROJ, DOWN_PROJ):
    x1 = torch.relu(X @ UP_PROJ)
    return x1 @ DOWN_PROJ

def fused_mlp(X, UP_PROJ, DOWN_PROJ, swizzle=True):
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
    grid = lambda META: (cdiv(D_UP, META['BLOCK_N']),)
    xs = _mlp_fused_kernel[grid](X, UP_PROJ, DOWN_PROJ, None, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE, SWIZZLE_BLOCK = swizzle)
    return xs


class FusedMLP(torch.nn.Module):
    def forward(self, X, UP_PROJ, DOWN_PROJ):
        return fused_mlp_ref(X, UP_PROJ, DOWN_PROJ)

triton_max_autotune = torch.compile(FusedMLP())

a = torch.randn((32, 32), device='cuda', dtype=torch.float16)
w1 = torch.randn((32, 128), device='cuda', dtype=torch.float16)
w2 = torch.randn((128, 32), device='cuda', dtype=torch.float16)

y1 = fused_mlp(a, w1, w2, swizzle=False)
y2 = fused_mlp_ref(a, w1, w2)
print((y1 - y2).abs().max())

# %%
from hidet.utils.benchmark import Bench


def torch_naive(C, **kwargs):
    M = kwargs['M']
    D, D_UP, D_DOWN = C, C * 4, C
    a = torch.randn([M, D], dtype=torch.float16, device='cuda')
    w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
    w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')
    return lambda: fused_mlp_ref(a, w1, w2)

def triton_fused(C, **kwargs):
    M = kwargs['M']
    D, D_UP, D_DOWN = C, C * 4, C
    a = torch.randn([M, D], dtype=torch.float16, device='cuda')
    w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
    w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

    return lambda: fused_mlp(a, w1, w2, swizzle=False)

def triton_fused_swizzle(C, **kwargs):
    M = kwargs['M']
    D, D_UP, D_DOWN = C, C * 4, C
    a = torch.randn([M, D], dtype=torch.float16, device='cuda')
    w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
    w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

    return lambda: fused_mlp(a, w1, w2, swizzle=True)

def triton_default(C, **kwargs):
    M = kwargs['M']
    D, D_UP, D_DOWN = C, C * 4, C
    a = torch.randn([M, D], dtype=torch.float16, device='cuda')
    w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
    w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

    return lambda: triton_max_autotune.forward(a, w1, w2)

for M in [1, 2, 4, 8]:
    bn = Bench(x_vals=[64, 128, 512, 1024, 4096], x_name='C', M=M)
    # bn.measure_flops(lambda C: C**2 * 2)
    bn.bench(torch_naive)
    bn.bench(triton_fused)
    # bn.bench(triton_fused_swizzle)
    # bn.bench(triton_default)

    data = bn.run()
    data.show_plot(title=f'M={M}')




# %%
import torch
import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=5, num_warps=8), #1 4096 16384 / 1 16384 4096
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4), #32 4096 16384 / 32 16384 4096
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=5, num_warps=4), #32 4096 16384 / 32 16384 4096
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    #c = accumulator.to(tl.float16)
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N2': 32, 'GROUP_SIZE_M': 1}, num_stages=5, num_warps=4), #1 4096 16384 / 1 16384 4096
        #triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N2': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4), #32 4096 16384 / 32 16384 4096
    ],
    key=['M1', 'K1', 'N1', 'K2', 'N2'],
)
@triton.jit
def two_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, output_ptr,
    # Matrix dimensions
    M1, K1, N1, K2, N2,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ck, stride_cn,
    stride_outputd, stride_outputm, stride_outputn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N2: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M1, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N1, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M1
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N1
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K1, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K1 - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K1 - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    #c = accumulator.to(tl.float16)
    y = accumulator.to(tl.bfloat16)

    # By far, the first matmul computes the [BLOCK_SIZE_M, BLOCK_SIZE_N] block of the output matrix.
    # We then directly use the output matrix as the input matrix of the second matmul.
    # y[offset_ym: offsetym + BLOCK_SIZE_M, offset_yn: offsetyn + BLOCK_SIZE_N]
    # Let's assume BLOCK_SIZE_M >= M1, then we have
    # y[0: M1, offset_yn: offsetyn + BLOCK_SIZE_N] *
    # c[offset_yn: offsetyn + BLOCK_SIZE_N: 0: N2] ->
    # output[0: M1, 0: M2]
    # Note that this is only a partial results, we need to do a global reduction
    # on N1 // BLOCK_SIZE_N blocks of output matrix to get the final results.
    offs_ck = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.arange(0, BLOCK_SIZE_N2)
    #accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N2), dtype=tl.float32)
    for n in range(0, tl.cdiv(N2, BLOCK_SIZE_N2)):
        # Load the next blokc of C, generate a mask by checking the K2 dimension.
        # If it is out of bounds, set it to 0.
        c_ptrs = c_ptr + (offs_ck[:, None] * stride_ck + offs_cn[None, :] * stride_cn)
        c = tl.load(c_ptrs, mask=offs_ck[:, None] < K2, other=0.0)
        # We accumulate along the K dimension.
        dot_tmp = tl.dot(y, c, out_dtype=tl.float32)
        # Write back the block of the output matrix output with masks.
        #y2 = accumulator.to(tl.bfloat16)
        offs_outputd = pid_n
        offs_outputm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_outputn = n * BLOCK_SIZE_N2 + tl.arange(0, BLOCK_SIZE_N2)
        output_ptrs = output_ptr + stride_outputd * offs_outputd + \
                    stride_outputm * offs_outputm[:, None] + stride_outputn * offs_outputn[None, :]
        output_mask = (offs_outputm[:, None] < M1) & (offs_outputn[None, :] < N2)

        tl.store(output_ptrs, dot_tmp, mask=output_mask)

        c_ptrs += BLOCK_SIZE_N2 * stride_cn

    # # -----------------------------------------------------------
    # # Write back the block of the output matrix C with masks.
    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # tl.store(c_ptrs, c, mask=c_mask)

    # TODO: We are still missing the final reduction step.

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    #import pdb; pdb.set_trace()
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c


def two_matmul(a, b, c, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert b.shape[1] == c.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix C must be contiguous"
    M1, K1 = a.shape
    K1, N1 = b.shape
    K2, N2 = c.shape
    # Allocates output.
    BLOCK_SIZE_N = 128
    n_dup = triton.cdiv(N1, BLOCK_SIZE_N)
    # output = torch.empty((M1, N2), device=a.device, dtype=a.dtype)
    #output = torch.empty((n_dup, M1, N2), device=a.device, dtype=a.dtype)
    output = torch.empty((n_dup, M1, N2), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M1, META['BLOCK_SIZE_M']) * triton.cdiv(N1, META['BLOCK_SIZE_N']),
    )
    #import pdb; pdb.set_trace()
    two_matmul_kernel[grid](
        a, b, c, output,
        M1, K1, N1, K2, N2,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        ACTIVATION=activation
    )

    # output_reduction = torch.zeros((M1, N2), device=a.)

    # return output[0, :, :]
    output_reduction = torch.sum(output, dim=0).to(torch.bfloat16)
    #import pdb; pdb.set_trace()
    return output_reduction


def benchmark_single_matmul(M, N, K, dtype):
    print(f"M={M}, N={N}, K={K}, dtype={dtype}")
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)

    # correctness check
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-1):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    print(f"torch: {ms:.3f} ms")
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    print(f"triton: {ms:.3f} ms")


def benchmark_back2back_matmul(bsz, hidden_dim, ffn_dim, dtype):
    print(f"bsz={bsz}, hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, dtype={dtype}")
    torch.manual_seed(0)
    low: float = -1e-3
    high: float = 1e-3
    x = torch.empty((bsz, hidden_dim), device='cuda', dtype=dtype).uniform_(low, high)
    w1 = torch.empty((hidden_dim, ffn_dim), device='cuda', dtype=dtype).uniform_(low, high)
    w2 = torch.empty((ffn_dim, hidden_dim), device='cuda', dtype=dtype).uniform_(low, high)

    # correctness check
    def torch_matmul(x, w1, w2):
        y = torch.matmul(x, w1)
        #y = torch.nn.GELU(approximate="tanh")(y)
        y = torch.matmul(y, w2)
        return y

    def triton_matmul(x, w1, w2):
        y = matmul(x, w1)
        #y = torch.nn.GELU(approximate="tanh")(y)
        y = matmul(y, w2)
        return y

    triton_output = triton_matmul(x, w1, w2)
    torch_output = torch_matmul(x, w1, w2)
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    # import pdb; pdb.set_trace()
    triton_output2 = two_matmul(x, w1, w2)
    if torch.allclose(triton_output2, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    #import pdb; pdb.set_trace()

    quantiles = [0.5, 0.2, 0.8]
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"torch_matmul")
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_matmul(x, w1, w2), quantiles=quantiles)
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"torch: {ms:.3f} ms")
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(x, w1, w2), quantiles=quantiles)
    print(f"triton(unfused): {ms:.3f} ms")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"triton_matmul")
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: two_matmul(x, w1, w2), quantiles=quantiles)
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"triton(fused 2matmuls): {ms:.3f} ms")


bsz = 1
# 6.7b
hidden_dim = 4096
# 26b
#hidden_dim = 6720
# 52b
#hidden_dim = 8192

tp = 1

ffn_dim = hidden_dim * 4 // tp
#dtype = torch.float16
#dtype = torch.float16
dtype = torch.bfloat16

# # fc1
# # wrong for bfloat16
# # correct for float16
# m = bsz
# n = hidden_dim * 4
# k = hidden_dim
# benchmark_single_matmul(m, n, k, dtype)

# # fc2
# # correct
# m = bsz
# n = hidden_dim
# k = hidden_dim * 4
# benchmark_single_matmul(m, n, k, torch.bfloat16)

# m = 512
# n = 512
# k = 512
# benchmark_single_matmul(m, n, k, torch.bfloat16)

benchmark_back2back_matmul(bsz, hidden_dim, ffn_dim, dtype)

# 64layers
# (0.569-0.434)*64 = 8.64ms reduction

# 64layers
# (0.569-0.375)*64 = 12.416ms reduction

'''
52B
'''
# bsz 1 tp=4
# torch: 0.569 ms
# triton(unfused): 0.543 ms
# triton(fused 2matmuls): 0.434 ms


# bsz 1 tp=4
# torch: 0.569 ms
# triton(unfused): 0.543 ms
# triton(fused 2matmuls): 0.375 ms

'''
26B
'''
# bsz 1
# torch: 1.465 ms
# triton(unfused): 1.458 ms
# triton(fused 2matmuls): 1.122 ms

# bsz 1 tp=2
# torch: 0.737 ms
# triton(unfused): 0.736 ms
# triton(fused 2matmuls): 0.512 ms
# 48layers
# (0.737-0.512)*48 = 10.8ms reduction

# bsz 1 tp=4
# triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N2': 32, 'GROUP_SIZE_M': 1}, num_stages=5, num_warps=4), #1 4096 16384 / 1 16384 4096
# torch: 0.365 ms
# triton(unfused): 0.378 ms
# triton(fused 2matmuls): 0.274 ms
# 48layers
# (0.365-0.274)*48 = 4.368ms reduction

'''
6.7B
'''
# bsz 1
# torch: 0.585 ms
# triton(unfused): 0.636 ms
# triton(fused 2matmuls): 0.368 ms

# bsz 4
# torch: 0.564 ms
# triton(unfused): 0.612 ms
# triton(fused 2matmuls): 0.368 ms

# bsz 8
# torch: 0.572 ms
# triton: 0.617 ms
# triton(fused 2matmuls): 0.381 ms

# bsz 16
# torch: 0.565 ms
# triton(unfused): 0.611 ms
# triton(fused 2matmuls): 0.418 ms

# bsz 32
# torch: 0.564 ms
# triton(unfused): 0.616 ms
# triton(fused 2matmuls): 0.500 ms