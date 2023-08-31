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

print((k_cache1 - k_cache2).abs().max())
print((y1 - y2).abs().max())
