# %%
import triton
import torch
import triton.language as tl
from triton import autotune, Config
@autotune(
    configs=[
        Config({'BLOCK_N': 64}, num_stages=3, num_warps=4),
        Config({'BLOCK_N': 64}, num_stages=2, num_warps=4)
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

q = torch.randn(1, 1, 32, 128, dtype=torch.float16, device='cuda') * 100
k = torch.randn(1, 1, 64, 128, dtype=torch.float16, device='cuda') * 100
v = torch.randn(1, 1, 64, 128, dtype=torch.float16, device='cuda') * 100

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
# y1 = forward(q, k, v, 1.0)
y2 = triton_decoding_attn_p2(q, k, v)
# y3 = attention(q, k, v, False, 1.0)

# print((y - y1).abs().max())
print((y - y2).abs().max())
# print((y - y3).abs().max())
# err = (y - y1).abs().squeeze()
# print(err[:, 0])
