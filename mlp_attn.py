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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
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
        acc += tl.dot(a, b, out_dtype=dot_out_dtype)
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


MSIZE = 32

def get_configs_io_bound_mlp():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_k in [32, 64]:
            for block_n in [32, 64, 128, 256]:
                num_warps = 2 if block_n <= 64 else 4
                configs.append(
                    Config({'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                            num_stages=num_stages, num_warps=num_warps))
                # split_k
                for split_k in [2, 4, 8, 16]:
                    configs.append(Config({'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                            num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs

@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=3, num_warps=8, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=3, num_warps=8, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 64,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 256, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 64,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 32,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 32,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=5, num_warps=2, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 32,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=4, num_warps=2, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 32,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=3, num_warps=2, pre_hook=init_to_zero('Y')),
        Config({'BLOCK_N': 32,  'BLOCK_K': 32, 'BLOCK_H': 32}, num_stages=2, num_warps=2, pre_hook=init_to_zero('Y')),
    ],
    key=['D_UP', 'D', 'D_DOWN'],
)
@heuristics({
    'EVEN_K': lambda args: args['D'] % (args['BLOCK_K']) == 0,
    'EVEN_M': lambda args: args['M'] % MSIZE == 0,
    'EVEN_H': lambda args: args['D_DOWN'] % (args['BLOCK_H']) == 0,
})
@jit
def mlp_fused(
    X, UP_PROJ, DOWN_PROJ, Y,  # pointers
    M, D, D_UP, D_DOWN, # shapes
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_M: tl.constexpr, EVEN_H: tl.constexpr
):
    # mlp
    # X[M, D] x UP_PROJ[D, D_UP] -> X0[M, D_UP]
    # relu X0[M, D_UP] -> X1[M, D_UP]
    # X1[M, D_UP] x DOWN_PROJ[D_UP, D_DOWN] -> Y[M, D_DOWN]

    pid = tl.program_id(0)
    grid_n = tl.cdiv(D_UP, BLOCK_N)
    pid_n = pid % grid_n 

    rm = tl.arange(0, MSIZE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # rbn = tl.max_contiguous(tl.multiple_of(rn % D_UP, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    X = X + (rm[:, None] * D + rk[None, :])
    UP_PROJ = UP_PROJ + (rk[:, None] * D_UP + rn[None, :])
    
    acc = tl.zeros((MSIZE, BLOCK_N))

    for k in range(0, tl.cdiv(D, BLOCK_K)):
        if EVEN_K:
            xs = tl.load(X)
        else:
            k_remaining = D - k * BLOCK_K
            xs = tl.load(X, mask=rk[None, :] < k_remaining, other=0)
        
        if EVEN_M and EVEN_K:
            ds = tl.load(UP_PROJ)
        else:
            m_remain = M - MSIZE
            ds = tl.load(UP_PROJ, mask=(rk[:, None] < k_remaining) & (rm[None, :] < m_remain), other=0)

        acc += tl.dot(xs, ds)
        X += BLOCK_K
        UP_PROJ += BLOCK_K * D_UP

    # relu
    acc = tl.where(acc < 0, 0, acc)
    # rematerialize rm and rn to save registers
    rm = tl.arange(0, MSIZE)
    rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rh = tl.arange(0, BLOCK_H)

    DOWN_PROJ = DOWN_PROJ + (rk[:, None] * D_DOWN + rh[None, :])
    Y = Y + (rm[:, None] * D_DOWN + rh[None, :])

    for h in range(0, tl.cdiv(D_DOWN, BLOCK_H)):
        h_remain = D_DOWN - h * BLOCK_H
        if EVEN_H:
            hs = tl.load(DOWN_PROJ)
        else:
            hs = tl.load(DOWN_PROJ, mask=(rh[None, :] < h_remain), other=0)

        res = tl.dot(acc, hs)
        tl.atomic_add(Y, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
        DOWN_PROJ += BLOCK_H
        Y += BLOCK_H


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
        if a.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5] or\
           b.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
            c_dtype = torch.float16
        else:
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
        if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
            ab_dtype = False
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


def fused_mlp_ref(X, UP_PROJ, DOWN_PROJ):
    x1 = torch.relu(X @ UP_PROJ)
    return x1 @ DOWN_PROJ

def fused_mlp(X, UP_PROJ, DOWN_PROJ):
    Y = torch.empty((X.shape[0], DOWN_PROJ.shape[1]), device=X.device, dtype=X.dtype)
    assert X.shape[0] <= MSIZE
    D = X.shape[1]
    D_UP = UP_PROJ.shape[1]
    D_DOWN = DOWN_PROJ.shape[1]
    grid = lambda META: (cdiv(D_UP, META['BLOCK_N']),)
    mlp_fused[grid](X, UP_PROJ, DOWN_PROJ, Y, X.shape[0], D, D_UP, D_DOWN)
    return Y


D = 512
D_UP = D * 4
D_DOWN = D

a = torch.randn([MSIZE, D], dtype=torch.float16, device='cuda')
w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

y1 = fused_mlp_ref(a, w1, w2)
y2 = fused_mlp(a, w1, w2)
