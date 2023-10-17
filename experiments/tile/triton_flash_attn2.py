# %%
import torch

import triton
from triton import cdiv, jit, autotune, Config
from triton import language as tl

@autotune(
    configs=[
        Config({'BLOCK_K': 64, 'BLOCK_N': 64, 'BLOCK_O': 64}, num_stages=3, num_warps=4)
    ],
    key=['D', 'L', 'D']
)
@triton.jit
def decoding_attn_kernel(
    X, # [B, S, D] // S <= 32 
    Y, # [B, H, S, D]
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

    x_ptr = tl.make_block_ptr(
        base=X + batch_id * S * D,
        shape=(S, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    wq_ptr = tl.make_block_ptr(
        base=WQ + head_id * BLOCK_DH,
        shape=(D, BLOCK_DH),
        strides=(H * BLOCK_DH, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_DH),
        order=(1, 0)
    )
    wk_ptr = tl.make_block_ptr(
        base=WK + head_id * BLOCK_DH,
        shape=(D, BLOCK_DH),
        strides=(H * BLOCK_DH, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_DH),
        order=(1, 0)
    )
    wv_ptr = tl.make_block_ptr(
        base=WV + head_id * BLOCK_DH,
        shape=(D, BLOCK_DH),
        strides=(H * BLOCK_DH, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_DH),
        order=(1, 0)
    )
    q = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    k_ = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    v_ = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    for _ in range(0, tl.cdiv(D, BLOCK_K)):
        x = tl.load(x_ptr)
        wq = tl.load(wq_ptr)
        q += tl.dot(x, wq)
        wk = tl.load(wk_ptr)
        k_ += tl.dot(x, wk)
        wv = tl.load(wv_ptr)
        v_ += tl.dot(x, wv)
        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        wq_ptr = tl.advance(wq_ptr, (BLOCK_K, 0))
        wk_ptr = tl.advance(wk_ptr, (BLOCK_K, 0))
        wv_ptr = tl.advance(wv_ptr, (BLOCK_K, 0))
    q = q.to(tl.float16)
    k_ = k_.to(tl.float16)
    v_ = v_.to(tl.float16)

    offset_cache = BLOCK_DH * L + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = tl.make_block_ptr(
        base=k_cache + offset_cache,
        shape=(S, BLOCK_DH),
        strides=(BLOCK_DH, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DH),
        order=(1, 0)
    )
    v_cache_ptr = tl.make_block_ptr(
        base=v_cache + offset_cache,
        shape=(S, BLOCK_DH),
        strides=(BLOCK_DH, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DH),
        order=(1, 0)
    )
    tl.store(k_cache_ptr, k_)
    tl.store(v_cache_ptr, v_)

    acc = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    qk_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    qk_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    qk_ = tl.dot(q, tl.trans(k_))
    qk_max = tl.max(qk_, 1)
    qk_ = tl.exp(qk_ - qk_max[:, None])
    qk_sum = tl.sum(qk_, 1)
    qk_ = qk_ / qk_sum[:, None]
    qk_ = qk_.to(tl.float16)
    acc += tl.dot(qk_, v_) # [BLOCK_M, BLOCK_DH]


    offset_cache = head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = tl.make_block_ptr(
        base=k_cache + offset_cache,
        shape=(BLOCK_DH, L + S),
        strides=(1, BLOCK_DH),
        offsets=(0, 0),
        block_shape=(BLOCK_DH, BLOCK_N),
        order=(0, 1)
    )
    v_cache_ptr = tl.make_block_ptr(
        base=v_cache + offset_cache,
        shape=(L + S, BLOCK_DH),
        strides=(BLOCK_DH, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DH),
        order=(1, 0)
    )
    qk_scale = qk_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = (q * qk_scale).to(tl.float16)
    


    for start_n in range(0, tl.cdiv(L + S, BLOCK_N)):
        # -- load k, v --
        k = tl.load(k_cache_ptr)
        v = tl.load(v_cache_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        qk_max_new = tl.maximum(qk_max, tl.max(qk, 1))
        alpha = tl.math.exp2(qk_max - qk_max_new)
        p = tl.math.exp2(qk - qk_max_new[:, None])
        # -- scale and update acc --
        acc_scale = qk_sum * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v, allow_tf32=True)
        # -- update m_i and l_i --
        qk_sum = qk_sum * alpha + tl.sum(p, 1)
        qk_max = qk_max_new
        # update pointers
        k_cache_ptr = tl.advance(k_cache_ptr, (0, BLOCK_N))
        v_cache_ptr = tl.advance(v_cache_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / qk_sum[:, None]
    acc = acc.to(tl.float16)
    y_ptr = tl.make_block_ptr(
        base=Y + batch_id * H * S * D + head_id * S * D,
        shape=(S, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DH),
        order=(1, 0)
    )
    tl.store(y_ptr, acc)

    

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
    y = torch.ones((B, H, S, D), dtype=x.dtype, device=x.device) * -1
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

    qk = q @ (k_cache[:, :, :L+S, :]).transpose(-1, -2)
    qk = torch.softmax(qk, dim=-1)
    y = (qk @ v_cache[:, :, :L+S, :]).transpose(1, 2).reshape([B, S, heads * head_dim]) @ WO
    return y

def test():
    B = 1
    S = 32
    D = 64
    H = 1
    DH = 64
    L = 0
    LS = 32

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

    # print(y1)
    # print(y2)
    print((k_cache1 - k_cache2).abs().max())
    print((y1 - y2).abs().max())
    return y1, y2

y1, y2 = test()
print(y1)
