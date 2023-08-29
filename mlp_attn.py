# %%

import torch

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                               num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                              num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@jit
def _kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            dot_out_dtype: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)



class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = {}

    @staticmethod
    def _call(a, b, dot_out_dtype):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        # if a.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5] or\
        #    b.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
        #     c_dtype = torch.float16
        # else:
        #     c_dtype = get_higher_dtype(a.dtype, b.dtype)
        c_dtype = get_higher_dtype(a.dtype, b.dtype)
        c = torch.empty((M, N), device=device, dtype=c_dtype)
        if dot_out_dtype is None:
            if c_dtype in [torch.float16, torch.float32, torch.bfloat16]:
                dot_out_dtype = tl.float32
            else:
                dot_out_dtype = tl.int32
        else:
            assert isinstance(dot_out_dtype, torch.dtype), "dot_out_dtype must be a torch.dtype"
            if dot_out_dtype == torch.float16:
                dot_out_dtype = tl.float16
            elif dot_out_dtype in [torch.float32, torch.bfloat16]:
                dot_out_dtype = tl.float32
            else:
                dot_out_dtype = tl.int32
        ab_dtype = True
        # if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
        #     ab_dtype = False
        # launch kernel
        grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      dot_out_dtype=dot_out_dtype,
                      GROUP_M=8, AB_DTYPE=ab_dtype)
        return c

    @staticmethod
    def forward(ctx, a, b, dot_out_dtype=None):
        return _matmul._call(a, b, dot_out_dtype=dot_out_dtype)

a = torch.randn([512, 512], dtype=torch.float16, device='cuda')
b = torch.randn([512, 512], dtype=torch.float16, device='cuda')
_matmul.apply(a, b)

# %%
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
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr):
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
    acc = tl.where(acc < 0, 0, acc)
    # rematerialize rm and rn to save registers
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


def init_to_zero(nargs):
    nargs['Y'].zero_()

@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 32, 'BLOCK_K': 128, 'BLOCK_H': 128}, num_stages=5, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_H': 128}, num_stages=3, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_stages=4, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 64}, num_stages=2, num_warps=8, pre_hook=init_to_zero),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero), #32 4096 16384 / 32 16384 4096
    ],
    key=['D_UP', 'D', 'D_DOWN'],
)
@jit
def _mlp_fused_kernel_atomic(X, UP_PROJ, DOWN_PROJ, Y,  # pointers
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
    acc = tl.where(acc < 0, 0, acc)
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
        rm = tl.arange(0, BLOCK_M)
        rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rh = tl.arange(0, BLOCK_H)

        DOWN_PROJ_ptr = DOWN_PROJ + (rk[:, None] * D_DOWN + rh[None, :])
        Y_ptr = Y + (rm[:, None] * D_DOWN + rh[None, :])

        for h in range(0, tl.cdiv(D_DOWN, BLOCK_H)):
            h_remain = D_DOWN - h * BLOCK_H
            hs = tl.load(DOWN_PROJ_ptr, mask=(rk[:, None] < D_UP) & (rh[None, :] < h_remain), other=0)

            res = tl.dot(acc, hs).to(tl.float16)
            tl.atomic_add(Y_ptr, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
            DOWN_PROJ_ptr += BLOCK_H
            Y_ptr += BLOCK_H
        
        

def fused_mlp_ref(X, UP_PROJ, DOWN_PROJ):
    x1 = torch.relu(X @ UP_PROJ)
    return x1 @ DOWN_PROJ

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
    grid = lambda META: (cdiv(D_UP, META['BLOCK_N']),)
    xs = _mlp_fused_kernel[grid](X, UP_PROJ, DOWN_PROJ, None, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE)
    return xs

def fused_mlp_atomic(X, UP_PROJ, DOWN_PROJ, swizzle=False):
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
    grid = lambda META: (cdiv(D_UP, META['BLOCK_N']),)
    _mlp_fused_kernel_atomic[grid](X, UP_PROJ, DOWN_PROJ, Y, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE, SWIZZLE_BLOCK = swizzle)
    return Y

class FusedMLP(torch.nn.Module):
    def forward(self, X, UP_PROJ, DOWN_PROJ):
        return fused_mlp_ref(X, UP_PROJ, DOWN_PROJ)

triton_max_autotune = torch.compile(FusedMLP())

def check_kernels(M=32, D=32):
    a = torch.randn((M, D), device='cuda', dtype=torch.float16)
    w1 = torch.randn((D, D * 4), device='cuda', dtype=torch.float16)
    w2 = torch.randn((D * 4, D), device='cuda', dtype=torch.float16)

    y1 = fused_mlp_ref(a, w1, w2)
    y2 = fused_mlp_atomic(a, w1, w2)
    y3 = fused_mlp(a, w1, w2)
    print((y1 - y2).abs().max())
    print((y1 - y3).abs().max())

check_kernels(32, 128)

# %%
#  #%%
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

    return lambda: fused_mlp(a, w1, w2)

def triton_fused_atomic(C, **kwargs):
    M = kwargs['M']
    D, D_UP, D_DOWN = C, C * 4, C
    a = torch.randn([M, D], dtype=torch.float16, device='cuda')
    w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
    w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

    return lambda: fused_mlp_atomic(a, w1, w2, swizzle=False)

def triton_fused_swizzle(C, **kwargs):
    M = kwargs['M']
    D, D_UP, D_DOWN = C, C * 4, C
    a = torch.randn([M, D], dtype=torch.float16, device='cuda')
    w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
    w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

    return lambda: fused_mlp_atomic(a, w1, w2, swizzle=True)

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
    bn.bench(triton_fused_atomic)
    # bn.bench(triton_fused_swizzle)
    # bn.bench(triton_default)

    data = bn.run()
    data.show_plot(title=f'M={M}')


