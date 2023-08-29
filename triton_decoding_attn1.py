# %%
import os
os.chdir('/home/allan/Programs/triton/python')

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

    # qk_ = tl.dot(q, tl.trans(k))
    # qk_max = tl.max(qk_, 1)
    # qk_ = tl.exp(qk_ - qk_max[:, None])
    # qk_sum = tl.sum(qk_, 1)
    # qk_ = qk_ / qk_sum[:, None]
    # qk_ = qk_.to(tl.float16)
    # y_acc = tl.dot(qk_, v) # [BLOCK_M, BLOCK_DH]
    qk_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    qk_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # now do flash attention on the kv_cache for [0, L]
    kvl = tl.arange(0, BLOCK_N)
    kvd = tl.arange(0, BLOCK_DH)
    # kv_cache_idx = kvl[:, None] * LS + kvd[None, :] + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = k_cache + (kvl[None, :] * BLOCK_DH + kvd[:, None] + head_id * LS * BLOCK_DH + batch_id * H * LS * BLOCK_DH)
    v_cache_ptr = v_cache + (kvl[:, None] * BLOCK_DH + kvd[None, :] + head_id * LS * BLOCK_DH + batch_id * H * LS * BLOCK_DH)

    acc = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32) # + y_acc
    for l in range(0, tl.cdiv(L + S, BLOCK_N)):
        l_remaining = L + S - l * BLOCK_N
        k_ = tl.load(k_cache_ptr, mask=kvl[:, None] < l_remaining)
        qk = tl.dot(q, k_)
        qk_max_curr = tl.maximum(tl.max(qk, 1), qk_max)
        qk_sum *= tl.exp(qk_max - qk_max_curr)
        p = tl.exp(qk - qk_max_curr[:, None])
        qk_sum_curr = tl.sum(p, 1) + qk_sum
        sum_rcp = 1.0 / qk_sum_curr
        p *= sum_rcp[:, None]
        p = p.to(tl.float16)

        acc *= (qk_sum * sum_rcp)[:, None]
        vc = tl.load(v_cache_ptr, mask=kvl[:, None] < l_remaining)
        acc += tl.dot(p, vc)
        
        qk_sum = qk_sum_curr
        qk_max = qk_max_curr

        # vc = tl.load(v_cache_ptr, mask=kvl[:, None] < l_remaining)
        # acc += tl.dot((qkc - qk_max[:, None]).to(tl.float16), vc)
        
        k_cache_ptr += BLOCK_N * BLOCK_DH
        v_cache_ptr += BLOCK_N * BLOCK_DH

    # acc[BLOCK_M, BLOCK_DH] @ WO[head_id * BLOCK_DH:(head_id + 1) * BLOCK_DH, :] -> y_part[BLOCK_M, D]    
    acc = acc.to(tl.float16)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(0)
    ym = tl.arange(0, BLOCK_M)
    yd = tl.arange(0, BLOCK_DH)
    oh = tl.arange(0, BLOCK_O)


    y_ptr = Y + (ym[:, None] * D + oh[None, :] + batch_id * S * D) # [BLOCK_M, BLOCK_O]
    wo_ptr = WO + (yd[:, None] * D + oh[None, :] + head_id * BLOCK_DH * D) # [BLOCK_DH, BLOCK_O]
    for i in range(0, tl.cdiv(D, BLOCK_O)):
        d_remaining = D - i * BLOCK_O
        wo = tl.load(wo_ptr, mask=oh[None, :] < d_remaining)
        y = tl.dot(acc, wo)
        y = y.to(tl.float16)
        tl.atomic_add(y_ptr, y, mask=(ym[:, None] < S) & (oh[None, :] < d_remaining))
        y_ptr += BLOCK_O
        wo_ptr += BLOCK_O

    # y = acc.to(tl.float16)#  + y_acc # a compiler bug
    # batch_id = tl.program_id(1)
    # head_id = tl.program_id(0)
    # stride_hd = H * BLOCK_DH
    # ym = tl.arange(0, BLOCK_M)
    # yd = tl.arange(0, BLOCK_DH)

    # res_ptr = Y + (ym[:, None] * D + yd[None, :] + head_id * stride_hd + batch_id * S * D)
    # tl.atomic_add(res_ptr, y)


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
    y = torch.empty((B, S, D), dtype=x.dtype, device=x.device)
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
    q = (x @ WQ).reshape([B, heads, -1, head_dim])
    k = (x @ WK).reshape([B, heads, -1, head_dim])
    v = (x @ WV).reshape([B, heads, -1, head_dim])
    k_cache[:, :, L:L+S, :] = k
    v_cache[:, :, L:L+S, :] = v

    qk = q @ (k_cache[:, :, :L+S, :]).transpose(-1, -2)
    qk = torch.softmax(qk, dim=-1)
    y = (qk @ v_cache[:, :, :L+S, :]).reshape([B, -1, heads * head_dim]) @ WO
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
print((k_cache1 - k_cache2).abs().max())


y_prime1 = triton_decoding_attn(x, wq, wk, wv, wo, k_cache1, v_cache1, L, H, DH)
y_prime2 = ref_decoding_attn(x, wq, wk, wv, wo, k_cache2, v_cache2, L, H, DH)

print((y1 - y_prime1).abs().max())
print((y2 - y_prime2).abs().max())
