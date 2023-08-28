# %%
import torch

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

MSIZE = 64

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
    'EVEN_M': lambda args: args['M'] % (MSIZE) == 0, 
    'EVEN_H': lambda args: args['D_DOWN'] % (args['BLOCK_H']) == 0,
})
@jit
def _mlp_fused_kernel(X, UP_PROJ, DOWN_PROJ, Y,  # pointers
                     M, D, D_UP, D_DOWN, # shapes
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
                     EVEN_K: tl.constexpr, EVEN_M: tl.constexpr, EVEN_H: tl.constexpr):
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

    tl.store(Y, acc)
    # for h in range(0, tl.cdiv(D_DOWN, BLOCK_H)):
    #     h_remain = D_DOWN - h * BLOCK_H
    #     if EVEN_H:
    #         hs = tl.load(DOWN_PROJ)
    #     else:
    #         hs = tl.load(DOWN_PROJ, mask=(rh[None, :] < h_remain), other=0)

    #     res = tl.dot(acc, hs)
    #     tl.atomic_add(Y, res, mask=(rm[:, None] < M) & (rh[None, :] < h_remain))
    #     DOWN_PROJ += BLOCK_H
    #     Y += BLOCK_H


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
    _mlp_fused_kernel[grid](X, UP_PROJ, DOWN_PROJ, Y, X.shape[0], D, D_UP, D_DOWN)
    return Y


D = 512
D_UP = D * 4
D_DOWN = D

a = torch.randn([MSIZE, D], dtype=torch.float16, device='cuda')
w1 = torch.randn([D, D_UP], dtype=torch.float16, device='cuda')
w2 = torch.randn([D_UP, D_DOWN], dtype=torch.float16, device='cuda')

y1 = fused_mlp_ref(a, w1, w2)
y2 = fused_mlp(a, w1, w2)
