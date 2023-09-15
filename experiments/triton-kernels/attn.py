# %%
import os
# os.chdir('/home/allan/Programs/triton/python')

import triton
import triton.language as tl
from triton.ops import attention
from triton import autotune, Config

import torch


def get_configs():
    configs = []
    for BK in [32, 64, 128]:
        for BN in [32, 64, 128]:
            for num_warps in [2, 4, 8]:
                for num_stages in [2, 3, 4]:
                    configs.append(
                        Config(
                            {'BLOCK_K': BK, 'BLOCK_N': BN, 'BLOCK_O': 64}, num_stages=num_stages, num_warps=num_warps
                        )
                    )
    return configs


@autotune(
    configs=get_configs(),
    key=['D', 'L', 'D']
)
@triton.jit
def decoding_attn_kernel_part2(
    X,  # [B, S, D] // S <= 32
    # Q, # [B, H, S, DH]
    Y,  # [B, S, D]
    WQ, WK, WV,  # [D, H * DH]
    WO,  # [H * DH, D]
    qk_scale,  # float
    k_cache,  # [B, H, LS, DH]
    v_cache,  # [B, H, LS, DH]
    S, B, H, L, LS, D,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DH: tl.constexpr,  # DH \in {32, 64, 128}
    BLOCK_N: tl.constexpr,
    BLOCK_O: tl.constexpr,
    # IS_CASUAL: tl.constexpr,
):

    batch_id = tl.program_id(1)
    head_id = tl.program_id(0)

    stride_hd = H * BLOCK_DH

    # compute x[:, L:L+S, :] * WQ[:, head_id * DH:(head_id + 1) * DH]     -> xq[:, S, DH]
    # compute (x[:, L:L+S, :] * WK[:, head_id * DH:(head_id + 1) * DH])^T -> xk[:, DH, S]
    # compute x[:, L:L+S, :] * WV[:, head_id * DH:(head_id + 1) * DH]     -> xv[:, S, DH]

    xm = tl.arange(0, BLOCK_M)
    xk1 = tl.arange(0, BLOCK_K)
    xn = tl.arange(0, BLOCK_DH)

    x_ptr = X + (xm[:, None] * D + xk1[None, :] + batch_id * S * D)
    xqkv = xk1[:, None] * stride_hd + xn[None, :] + head_id * BLOCK_DH
    wk_ptr = WK + xqkv

    k = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    for _ in range(0, tl.cdiv(D, BLOCK_K)):
        # assumes that D % BLOCK_K1 == 0
        x = tl.load(x_ptr, mask=xm[:, None] < S)
        wk = tl.load(wk_ptr)
        k += tl.dot(x, wk)

        x_ptr += BLOCK_K
        wk_ptr += BLOCK_K * stride_hd

    k = k.to(tl.float16)

    kvl = tl.arange(0, BLOCK_M)
    kvd = tl.arange(0, BLOCK_DH)
    kv_cache_idx = kvl[:, None] * BLOCK_DH + kvd[None,
                                             :] + BLOCK_DH * L + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = k_cache + kv_cache_idx
    tl.store(k_cache_ptr, k, mask=kvl[:, None] < S)

    xm = tl.arange(0, BLOCK_M)
    xk1 = tl.arange(0, BLOCK_K)
    xn = tl.arange(0, BLOCK_DH)

    x_ptr = X + (xm[:, None] * D + xk1[None, :] + batch_id * S * D)
    xqkv = xk1[:, None] * stride_hd + xn[None, :] + head_id * BLOCK_DH
    wv_ptr = WV + xqkv

    v = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    for _ in range(0, tl.cdiv(D, BLOCK_K)):
        # assumes that D % BLOCK_K1 == 0
        x = tl.load(x_ptr, mask=xm[:, None] < S)
        wv = tl.load(wv_ptr)
        v += tl.dot(x, wv)

        x_ptr += BLOCK_K
        wv_ptr += BLOCK_K * stride_hd

    v = v.to(tl.float16)

    kvl = tl.arange(0, BLOCK_M)
    kvd = tl.arange(0, BLOCK_DH)
    kv_cache_idx = kvl[:, None] * BLOCK_DH + kvd[None,
                                             :] + BLOCK_DH * L + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    v_cache_ptr = v_cache + kv_cache_idx
    tl.store(v_cache_ptr, v, mask=kvl[:, None] < S)

    xm = tl.arange(0, BLOCK_M)
    xk1 = tl.arange(0, BLOCK_K)
    xn = tl.arange(0, BLOCK_DH)

    x_ptr = X + (xm[:, None] * D + xk1[None, :] + batch_id * S * D)
    xqkv = xk1[:, None] * stride_hd + xn[None, :] + head_id * BLOCK_DH
    wq_ptr = WQ + xqkv
    wv_ptr = WV + xqkv

    q = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    for _ in range(0, tl.cdiv(D, BLOCK_K)):
        # assumes that D % BLOCK_K1 == 0
        x = tl.load(x_ptr, mask=xm[:, None] < S)
        wq = tl.load(wq_ptr)
        q += tl.dot(x, wq)
        x_ptr += BLOCK_K
        wq_ptr += BLOCK_K * stride_hd

    qk_scale = 1.44269504
    q = (q * qk_scale).to(tl.float16)

    # q_cache_ptr = Q + (kvl[:, None] * BLOCK_DH + kvd[None, :] + head_id * S * BLOCK_DH + batch_id * H * S * BLOCK_DH)
    # tl.store(q_cache_ptr, q, mask=kvl[:, None] < S)

    batch_id = tl.program_id(1)
    head_id = tl.program_id(0)

    xm = tl.arange(0, BLOCK_M)
    xn = tl.arange(0, BLOCK_DH)

    # q_ptr = Q + (xm[:, None] * BLOCK_DH + xn[None, :] + head_id * S * BLOCK_DH + batch_id * H * S * BLOCK_DH)
    # q = tl.load(q_ptr, mask=xm[:, None] < S)
    # qk_scale = 1.44269504
    # q = (q * qk_scale).to(tl.float16)

    acc = tl.zeros([BLOCK_M, BLOCK_DH], dtype=tl.float32)
    qk_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    qk_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    kvl = tl.arange(0, BLOCK_N)
    kvd = tl.arange(0, BLOCK_DH)
    # kv_cache_idx = kvl[:, None] * LS + kvd[None, :] + head_id * LS * BLOCK_DH + batch_id * BLOCK_DH * LS * H
    k_cache_ptr = k_cache + (
            kvl[None, :] * BLOCK_DH + kvd[:, None] + head_id * LS * BLOCK_DH + batch_id * H * LS * BLOCK_DH)
    v_cache_ptr = v_cache + (
            kvl[:, None] * BLOCK_DH + kvd[None, :] + head_id * LS * BLOCK_DH + batch_id * H * LS * BLOCK_DH)

    for l in range(0, tl.cdiv(L + S, BLOCK_N)):
        l_remaining = L + S - l * BLOCK_N
        k_ = tl.load(k_cache_ptr, mask=kvl[None, :] < l_remaining)
        vc = tl.load(v_cache_ptr, mask=kvl[:, None] < l_remaining)
        qk = tl.dot(q, k_)

        qk_max_curr = tl.maximum(tl.max(qk, 1), qk_max)
        alpha = tl.math.exp2(qk_max - qk_max_curr)
        p = tl.math.exp2(qk - qk_max_curr[:, None])

        acc_scale = qk_sum * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(Y.dtype.element_ty), vc)

        qk_sum = qk_sum * alpha + tl.sum(p, 1)
        qk_max = qk_max_curr

        k_cache_ptr += BLOCK_N * BLOCK_DH
        v_cache_ptr += BLOCK_N * BLOCK_DH
    acc = acc / qk_sum[:, None]
    acc = acc.to(Y.dtype.element_ty)
    ym = tl.arange(0, BLOCK_M)
    yd = tl.arange(0, BLOCK_DH)
    y_ptr = Y + (ym[:, None] * BLOCK_DH + yd[None,
                                          :] + head_id * S * BLOCK_DH + batch_id * H * S * BLOCK_DH)  # [BLOCK_M, BLOCK_O]
    tl.store(y_ptr, acc, mask=ym[:, None] < S)


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
    y = torch.empty((B, H, S, DH), dtype=x.dtype, device=x.device)
    decoding_attn_kernel_part2[(H, B)](
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
    k_cache[:, :, L:L + S, :] = k
    v_cache[:, :, L:L + S, :] = v

    qk = q @ (k_cache[:, :, :L + S, :]).transpose(-1, -2)
    qk = torch.softmax(qk, dim=-1)
    y = (qk @ v_cache[:, :, :L + S, :])  # .transpose(1, 2).reshape([B, S, heads * head_dim]) @ WO
    return y


def ref_decoding_attn2(x, WQKV, WO, k_cache, v_cache, L, heads, head_dim):
    # x: [B, S, D]
    # WQKV: [D, 3 * H * DH]
    B, S, D = x.shape
    qkv = x @ WQKV
    qkv = qkv.reshape([B, S, 3, heads, head_dim]).transpose(1, 3)
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]
    k_cache[:, :, L:L + S, :] = k
    v_cache[:, :, L:L + S, :] = v
    qk = q @ (k_cache[:, :, :L + S, :]).transpose(-1, -2)
    qk = torch.softmax(qk, dim=-1)
    y = (qk @ v_cache[:, :, :L + S, :])  # .transpose(1, 2).reshape([B, S, heads * head_dim]) @ WO
    return y


def triton_flash_attn(x, WQ, WK, WV, WO, k_cache, v_cache, L, heads, head_dim):
    for d in (x, WQ, WK, WV, WO, k_cache, v_cache):
        assert d.dtype == torch.float16

    B, S, D = x.shape
    H = heads
    DH = head_dim
    assert DH in (32, 64, 128)
    assert D == WQ.shape[0] == WK.shape[0] == WV.shape[0] == WO.shape[1]
    assert WQ.shape[1] == WK.shape[1] == WV.shape[1] == WO.shape[0] == H * DH
    q = (x @ WQ).reshape([B, S, heads, head_dim]).transpose(1, 2)
    k = (x @ WK).reshape([B, S, heads, head_dim]).transpose(1, 2)
    v = (x @ WV).reshape([B, S, heads, head_dim]).transpose(1, 2)
    k_cache[:, :, L:L + S, :] = k
    v_cache[:, :, L:L + S, :] = v
    with torch.no_grad():
        y = attention(q, k_cache[:, :, :L + S], v_cache[:, :, :L + S], False, 1.0)
    return y


def demo():

    B = 1
    D = 128
    H = 1
    DH = 32

    S = 16
    L = 0
    LS = L + S

    x = torch.randn(B, S, D, dtype=torch.float16, device='cuda')
    wq = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
    wk = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
    wv = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
    wo = torch.randn(H * DH, D, dtype=torch.float16, device='cuda')
    k_cache1 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')
    v_cache1 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')

    k_cache2 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')
    v_cache2 = torch.zeros(B, H, LS, DH, dtype=torch.float16, device='cuda')

    y1 = ref_decoding_attn(x, wq, wk, wv, wo, k_cache1, v_cache1, L, H, DH)
    y2 = triton_decoding_attn(x, wq, wk, wv, wo, k_cache2, v_cache2, L, H, DH)
    print("k-cache closeness:", (k_cache1 - k_cache2).abs().max())
    print("v-cache closeness:", (v_cache1 - v_cache2).abs().max())
    # print(k_cache1)
    # print(k_cache2)
    y3 = triton_flash_attn(x, wq, wk, wv, wo, k_cache2, v_cache2, L, H, DH)
    print("ref-fused closeness:", (y1 - y2).abs().max())
    print("ref-flash-attn closeness:", (y1 - y3).abs().max())


demo()

# %%
# x[1, S, D] x WQ[D, H * DH] -> q[1, H, S, DH]
# x[1, S, D] x WK[D, H * DH] -> k[1, H, S, DH]
# x[1, S, D] x WV[D, H * DH] -> v[1, H, S, DH]
# k_cache[1, H, L:L+S, DH] <- q[1, H, S, DH]
# v_cache[1, H, L:L+S, DH] <- v[1, H, S, DH]
# flash-attn on q[1, H, S, DH], k_cache[1, H, L+S, DH], v_cache[1, H, L+S, DH] -> y[1, H, S, DH]


L_MAX = 2048

DH = 64
S = 4
D = 1024
H = 8
B = 1

for S in [1, 2, 4, 8]:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['L'],  # Argument names to use as an x-axis for the plot
            x_vals=[
                64, 128, 512, 1024
            ],  # Different possible values for `x_name`
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=['torch', 'triton'],
            # Label name for the lines
            line_names=['torch', 'triton', ],
            # Line styles
            styles=[('green', '-'), ('blue', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-')],
            ylabel="ms",  # Label name for the y-axis
            plot_name=f"attn_performance-H={H}-D={D}-DH={DH}-S={S}",
            # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )
    def benchmark(L, provider):
        print(f"Running benchmark with L={L}, provider={provider}")
        x = torch.randn(B, S, D, dtype=torch.float16, device='cuda')
        wq = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
        wk = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
        wv = torch.randn(D, H * DH, dtype=torch.float16, device='cuda')
        wo = torch.randn(H * DH, D, dtype=torch.float16, device='cuda')
        k_cache = torch.zeros(B, H, L_MAX, DH, dtype=torch.float16, device='cuda')
        v_cache = torch.zeros(B, H, L_MAX, DH, dtype=torch.float16, device='cuda')

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            wqkv = torch.cat([wq, wk, wv], dim=1)
            if triton.__version__.split('.')[0] == '1':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: ref_decoding_attn2(x, wqkv, wo, k_cache, v_cache, L, H, DH)
                )
            elif triton.__version__.split('.')[0] == '2':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: ref_decoding_attn2(x, wqkv, wo, k_cache, v_cache, L, H, DH), quantiles=quantiles
                )
        if provider == 'triton':
            if triton.__version__.split('.')[0] == '1':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: triton_decoding_attn(x, wq, wk, wv, wo, k_cache, v_cache, L, H, DH)
                )
            elif triton.__version__.split('.')[0] == '2':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: triton_decoding_attn(x, wq, wk, wv, wo, k_cache, v_cache, L, H, DH), quantiles=quantiles
                )
        if provider == 'triton-flash':
            if triton.__version__.split('.')[0] == '1':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: triton_flash_attn(x, wq, wk, wv, wo, k_cache, v_cache, L, H, DH)
                )
            elif triton.__version__.split('.')[0] == '2':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: triton_flash_attn(x, wq, wk, wv, wo, k_cache, v_cache, L, H, DH), quantiles=quantiles
                )

        # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        # return perf(ms), perf(max_ms), perf(min_ms)
        return ms, max_ms, min_ms


    benchmark.run(show_plots=True, print_data=True)

