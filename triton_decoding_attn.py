# %%
import triton
import triton.language as tl
from triton import autotune, Config

import torch

def set_y(nargs):
    nargs['Y'].zero_()


@autotune(
    configs=[
        Config({'BLOCK_K': 64, 'BLOCK_N': 64, 'BLOCK_O': 64}, num_stages=3, num_warps=4, pre_hook=set_y)
    ],
    key=['D', 'L', 'D']
)
@triton.jit
def decoding_attn_kernel(
    X, # [B, S, D] // S <= 32 
    Y, # [B, S, D]
    WQ, WK, WV, # [D, H * DH] 
    WO, # [H * DH, D]
    qk_scale, # float
    k_cache, # [B, H, LS, DH]
    v_cache, # [B, H, LS, DH]
    S, B, H, L, LS, D,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DH: tl.constexpr, # DH \in {32, 64, 128}
    BLOCK_N: tl.constexpr,
    BLOCK_O: tl.constexpr,
    # IS_CASUAL: tl.constexpr,
):
    # assumes that BLOCK_DH == DH
    # computes x[:, L:L+S. :] of the output
    batch_id = tl.program_id(1)
    head_id = tl.program_id(0)

    stride_hd = H * BLOCK_DH

    xm = tl.arange(0, BLOCK_M)
    xk1 = tl.arange(0, BLOCK_K)
    xn = tl.arange(0, BLOCK_DH)

    x_ptr = X + (xm[:, None] * D + xk1[None, :] + batch_id * S * D)
    xqkv = xk1[:, None] * stride_hd + xn[None, :] + head_id * BLOCK_DH
    wq_ptr = WQ + xqkv
    wk_ptr = WK + xqkv
    wv_ptr = WV + xqkv

    # compute x[:, L:L+S, :] * WQ[:, head_id * DH:(head_id + 1) * DH]     -> xq[:, S, DH]
    # compute (x[:, L:L+S, :] * WK[:, head_id * DH:(head_id + 1) * DH])^T -> xk[:, DH, S]
    # compute x[:, L:L+S, :] * WV[:, head_id * DH:(head_id + 1) * DH]     -> xv[:, S, DH]

    q = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    k = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    v = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    for _ in range(0, tl.cdiv(D, BLOCK_K)):
        # assumes that D % BLOCK_K1 == 0
        x = tl.load(x_ptr, mask=xm[:, None] < S)
        wq = tl.load(wq_ptr)
        q += tl.dot(x, wq)
        wk = tl.load(wk_ptr)
        k += tl.dot(x, wk)
        wv = tl.load(wv_ptr)
        v += tl.dot(x, wv)
        x_ptr += BLOCK_K
        wq_ptr += BLOCK_K * stride_hd
        wk_ptr += BLOCK_K * stride_hd
        wv_ptr += BLOCK_K * stride_hd

    q = q.to(tl.float16)
    k = k.to(tl.float16)
    v = v.to(tl.float16)

    # store k, v back into kv_cache[batch_id, head_id, L:L+S, :]
    kvl = tl.arange(0, BLOCK_M)
    kvd = tl.arange(0, BLOCK_DH)
    kv_cache_idx = kvl[:, None] * BLOCK_DH + kvd[None, :] + BLOCK_DH * L + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = k_cache + kv_cache_idx
    tl.store(k_cache_ptr, k, mask=kvl[:, None] < S)
    v_cache_ptr = v_cache + kv_cache_idx
    tl.store(v_cache_ptr, v, mask=kvl[:, None] < S)

    qk_ = tl.dot(q, tl.trans(k))
    qk_max = tl.max(qk_, 1)
    qk_ = tl.exp(qk_ - qk_max[:, None])
    qk_sum = tl.sum(qk_, 1)
    qk_ = qk_ / qk_sum[:, None]
    qk_ = qk_.to(tl.float16)
    y_acc = tl.dot(qk_, v).to(tl.float16) # [BLOCK_M, BLOCK_DH]

    # [B, H, S, DH]
    ym = tl.arange(0, BLOCK_M)
    yd = tl.arange(0, BLOCK_DH)
    y_ptr = Y + (ym[:, None] * BLOCK_DH + yd[None, :] + head_id * S * BLOCK_DH + batch_id * H * S * BLOCK_DH)
    tl.store(y_ptr, y_acc, mask=ym[:, None] < S)


def triton_decoding_attn(x, WQ, WK, WV, WO, k_cache, v_cache, L, heads, head_dim):
    for d in (x, WQ, WK, WV, WO, k_cache, v_cache):
        assert d.dtype == torch.float16
    
    B, S, D = x.shape
    H = heads
    DH = head_dim
    assert DH in (32, 64, 128)
    assert D == WQ.shape[0] == WK.shape[0] == WV.shape[0] == WO.shape[1]
    assert WQ.shape[1] == WK.shape[1] == WV.shape[1] == WO.shape[0] == H * DH

    LS = k_cache.shape[2]
    assert L + S <= LS
    assert k_cache.shape == v_cache.shape == (B, H, LS, DH)
    if S <= 16:
        BLOCK_M = 16
    elif S <= 32:
        BLOCK_M = 32
    else:
        raise RuntimeError(f"Unsupported sequence length {S}")
    y = torch.zeros((B, H, S, DH), dtype=x.dtype, device=x.device)
    decoding_attn_kernel[(H, B)](
        x, y, WQ, WK, WV, WO, 1.0 / (DH ** 0.5), k_cache, v_cache, S, B, H, L, LS, D,
        BLOCK_M=BLOCK_M, BLOCK_DH=DH
    )
    return y

def ref_decoding_attn(x, WQ, WK, WV, WO, k_cache, v_cache, L, heads, head_dim):
    B = x.shape[0]
    S = x.shape[1]
    D = x.shape[2]
    assert WQ.shape == WK.shape == WV.shape == (D, heads * head_dim)
    assert WO.shape == (heads * head_dim, D)
    q = (x @ WQ).reshape([B, S, heads, head_dim]).transpose(1, 2)
    k = (x @ WK).reshape([B, S, heads, head_dim]).transpose(1, 2)
    v = (x @ WV).reshape([B, S, heads, head_dim]).transpose(1, 2)
    k_cache[:, :, L:L+S, :] = k
    v_cache[:, :, L:L+S, :] = v

    # qk = q @ (k_cache[:, :, :L+S, :]).transpose(-1, -2)
    # qk = torch.softmax(qk, dim=-1)
    # y = (qk @ v_cache[:, :, :L+S, :]).transpose(1, 2).reshape([B, -1, heads * head_dim]) @ WO
    print(q.shape, k.shape)
    qk = q @ k.transpose(-1, -2)
    qk = torch.softmax(qk, dim=-1)
    y = (qk @ v)
    return y

B = 1
S = 32
D = 1024
H = 8
DH = 64
L = 128
LS = 2048

x = torch.randn(B, S, D, dtype=torch.float16, device='cuda')
wq = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
wk = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
wv = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
wo = torch.randn(H * DH, D, dtype=torch.float16, device='cuda')
k_cache1 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')
v_cache1 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')

k_cache2 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')
v_cache2 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')

y1 = triton_decoding_attn(x, wq, wk, wv, wo, k_cache1, v_cache1, L, H, DH)
y2 = ref_decoding_attn(x, wq, wk, wv, wo, k_cache2, v_cache2, L, H, DH)

print(y1)
print(y2)

# %%

import torch
import triton
from triton import autotune, Config
import triton.language as tl
print(triton.__version__)
from triton.ops import attention


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    B, H, L, S,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(1)
    h_idx = tl.program_id(0)
    # initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    off_q = offs_m[:, None] * BLOCK_D + offs_d[None, :]
    off_k = b_idx * H * L * BLOCK_D + h_idx * L * BLOCK_D + offs_n[None, :] * BLOCK_D + offs_d[:, None]
    off_v = b_idx * H * L * BLOCK_D + h_idx * L * BLOCK_D + offs_n[:, None] * BLOCK_D + offs_d[None, :]
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S)
    # loop over k, v and update accumulator
    for start_n in range(0, tl.cdiv(L, BLOCK_N)):
        L_remaining = L - start_n * BLOCK_N
        # -- compute qk ----
        k = tl.load(k_ptrs, mask=offs_n[None, :] < L_remaining)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        # qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l

        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(V.dtype.element_ty)
        v = tl.load(v_ptrs, offs_n[:, None] < L_remaining)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * BLOCK_D
        v_ptrs += BLOCK_N * BLOCK_D
    # rematerialize offsets to save registers
    b_idx = tl.program_id(1)
    h_idx = tl.program_id(0)
    
    offs_m = tl.arange(0, BLOCK_M)
    # write back l and m
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_D)
    off_o = b_idx * H * S * BLOCK_D + h_idx * S * BLOCK_D + offs_m[:, None] * BLOCK_D + offs_n[None, :]
    out_ptrs = Out + off_o
    acc = acc.to(Out.dtype.element_ty)
    tl.store(out_ptrs, acc)

def forward(q, k, v, sm_scale):
    # only support for Ampere now
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise RuntimeError("Flash attention currently only supported for compute capability < 80")
    BLOCK_M = 32
    BLOCK_N = 32
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    # assert Lk in {16, 32, 64, 128}
    assert Lk in {64}  # TODO: fix other cases
    o = torch.empty_like(q)
    B, H, L, D = k.shape
    S = q.shape[2]
    grid = (H, B)

    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        o,
        B, H, L, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_D=D, num_warps=num_warps,
        num_stages=2,
    )
    return o

@autotune(
    configs=[
        Config({'BLOCK_N': 64}, num_stages=3, num_warps=4)
    ],
    key=['L']
)
@triton.jit
def decoding_attn_kernel_part2(
    Q, # [B, H, S, DH]
    Y, # [B, S, D]
    qk_scale, # float
    k_cache, # [B, H, LS, DH]
    v_cache, # [B, H, LS, DH]
    S, B, H, L,
    BLOCK_M: tl.constexpr,
    BLOCK_DH: tl.constexpr, # DH \in {32, 64, 128}
    BLOCK_N: tl.constexpr,
    # IS_CASUAL: tl.constexpr,
):
    batch_id = tl.program_id(1)
    head_id = tl.program_id(0)

    xm = tl.arange(0, BLOCK_M)
    xn = tl.arange(0, BLOCK_DH)

    q_ptr = Q + (xm[:, None] * BLOCK_DH + xn[None, :] + head_id * S * BLOCK_DH + batch_id * H * S * BLOCK_DH)
    q = tl.load(q_ptr, mask=xm[:, None] < S)
    # qk_scale = 1.44269504
    # q = (q * qk_scale).to(tl.float16)

    acc = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    qk_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    qk_sum = tl.zeros([BLOCK_M], dtype=tl.float32)


    kvl = tl.arange(0, BLOCK_N)
    kvd = tl.arange(0, BLOCK_DH)
    # kv_cache_idx = kvl[:, None] * LS + kvd[None, :] + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = k_cache + (kvl[None, :] * BLOCK_DH + kvd[:, None] + head_id * L * BLOCK_DH + batch_id * H * L * BLOCK_DH)
    v_cache_ptr = v_cache + (kvl[:, None] * BLOCK_DH + kvd[None, :] + head_id * L * BLOCK_DH + batch_id * H * L * BLOCK_DH)

    for l in range(0, tl.cdiv(L, BLOCK_N)):
        l_remaining = L - l * BLOCK_N
        k_ = tl.load(k_cache_ptr, mask=kvl[None, :] < l_remaining)
        qk = tl.dot(q, k_)
        qk_max_curr = tl.maximum(tl.max(qk, 1), qk_max)
        qk_sum *= tl.exp(qk_max - qk_max_curr)
        p = tl.exp(qk - qk_max_curr[:, None])
        qk_sum_curr = tl.sum(p, 1) + qk_sum
        sum_rcp = 1.0 / qk_sum_curr
        p *= sum_rcp[:, None]
        p = p.to(Q.dtype.element_ty)

        acc *= (qk_sum * sum_rcp)[:, None]
        vc = tl.load(v_cache_ptr, mask=kvl[:, None] < l_remaining)
        acc += tl.dot(p, vc)
        
        qk_sum = qk_sum_curr
        qk_max = qk_max_curr
        
        k_cache_ptr += BLOCK_N * BLOCK_DH
        v_cache_ptr += BLOCK_N * BLOCK_DH
    
    acc = acc.to(Q.dtype.element_ty)
    ym = tl.arange(0, BLOCK_M)
    yd = tl.arange(0, BLOCK_DH)
    y_ptr = Y + (ym[:, None] * BLOCK_DH + yd[None, :] + head_id * S * BLOCK_DH + batch_id * H * S * BLOCK_DH) # [BLOCK_M, BLOCK_O]
    tl.store(y_ptr, acc, mask=ym[:, None] < S)

def triton_decoding_attn_p2(q, k_cache, v_cache):
    B, H, S, DH = q.shape
    assert DH in (32, 64, 128)
    L = k_cache.shape[-2]

    assert k_cache.shape == v_cache.shape == (B, H, L, DH)
    if S <= 16:
        BLOCK_M = 16
    elif S <= 32:
        BLOCK_M = 32
    elif S <= 64:
        BLOCK_M = 64
    elif S <= 128:
        BLOCK_M = 128
    else:
        raise RuntimeError(f"Unsupported sequence length {S}")
    y = torch.ones((B, H, S, DH), dtype=q.dtype, device=q.device) * -1
    decoding_attn_kernel_part2[(H, B)](
        q, y, 1.0 / (DH ** 0.5), k_cache, v_cache, S, B, H, L,
        BLOCK_M=BLOCK_M, BLOCK_DH=DH
    )
    return y

q = torch.randn(1, 1, 32, 64, dtype=torch.float16, device='cuda') * 100
k = torch.randn(1, 1, 64, 64, dtype=torch.float16, device='cuda') * 100
v = torch.randn(1, 1, 64, 64, dtype=torch.float16, device='cuda') * 100

print(q.abs().max(), k.abs().max(), v.abs().max())

def attn(q, k, v):
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    qk = q @ k.transpose(-1, -2)
    qk = torch.softmax(qk, dim=-1)
    y = qk @ v
    return y

y = attn(q, k, v)
y1 = forward(q, k, v, 1.0)
y2 = triton_decoding_attn_p2(q, k, v)
# y3 = attention(q, k, v, False, 1.0)

print((y - y1).abs().max())
print((y - y2).abs().max())
# print((y - y3).abs().max())
# err = (y - y1).abs().squeeze()
# print(err[:, 0])


# %%
