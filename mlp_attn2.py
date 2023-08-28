# %%
import torch

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl


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
        xs = tl.load(X, mask=(rm < M) & (rk[None, :] < k_remaining), other=0)
        
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

def fused_mlp(X, UP_PROJ, DOWN_PROJ, swizzle=True):
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
    _mlp_fused_kernel[grid](X, UP_PROJ, DOWN_PROJ, Y, X.shape[0], D, D_UP, D_DOWN, BLOCK_M = MSIZE, SWIZZLE_BLOCK = swizzle)
    return Y

class FusedMLP(torch.nn.Module):
    def forward(self, X, UP_PROJ, DOWN_PROJ):
        return fused_mlp_ref(X, UP_PROJ, DOWN_PROJ)

triton_max_autotune = torch.compile(FusedMLP())

a = torch.randn((32, 32), device='cuda', dtype=torch.float16)
w1 = torch.randn((32, 128), device='cuda', dtype=torch.float16)
w2 = torch.randn((128, 32), device='cuda', dtype=torch.float16)

y1 = fused_mlp(a, w1, w2)
y2 = fused_mlp_ref(a, w1, w2)
print((y1 - y2).abs().max())

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
    bn.bench(triton_default)

    data = bn.run()
    data.show_plot(title=f'M={M}')



# %%
