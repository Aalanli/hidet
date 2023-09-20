# %%
import triton
import triton.language as tl

import torch

@triton.autotune(
    configs=[
        triton.Config({'block_s': 16, 'block_m': 32, 'block_h': 32}, num_stages=3, num_warps=8),
    ],
    key=['seq', 'h_size', 'm_size']
)
@triton.jit
def triton_fused_ffn(
    x_ptr,  # [seq, h_size]
    w1_ptr, # [h_size, 2 * m_size]
    w2_ptr, # [m_size, h_size]
    y_ptr,  # [seq, h_size]
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr
):
    pid = tl.program_id(0)

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m
    w2_ptrs = w1_ptrs + m_size
    y1_lhs = tl.zeros((block_s, block_m), dtype=tl.float32)
    y1_rhs = tl.zeros((block_s, block_m), dtype=tl.float32)

    for k in range(tl.cdiv(h_size, block_h)):
        x = tl.load(x_ptrs)
        w1_lhs = tl.load(w1_ptrs)
        y1_lhs += tl.dot(x, w1_lhs)
        w1_rhs = tl.load(w2_ptrs)
        y1_rhs += tl.dot(x, w1_rhs)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
        w2_ptrs += 2 * m_size * block_h

    y1 = tl.sigmoid(y1_lhs) * y1_lhs * y1_rhs
    y1 = y1.to(tl.float16)

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :]
    y_ptrs = y_ptr + (s_range[:, None] + pid * seq) * h_size + h_range[None, :]

    mask = s_range[:, None] < seq

    for k in range(tl.cdiv(h_size, block_h)):
        w2 = tl.load(w2_ptrs)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.store(y_ptrs, value=y, mask=mask)
        w2_ptrs += block_h
        y_ptrs += block_h

def triton_llama_ffn(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)

    block_m = 32

    y = torch.zeros(m_size // block_m, seq, h_size, dtype=torch.float16, device='cuda')

    grid = lambda META: (triton.cdiv(m_size, META['block_m']),)
    triton_fused_ffn[grid](
        x, w1, w2, y,
        seq,
        h_size,
        m_size
    )

    return y.sum(0)

def torch_ref(x, w1, w2):
    m_size = w2.size(0)
    y1 = x @ w1
    y1 = torch.nn.functional.silu(y1[:, :m_size]) * y1[:, m_size:]
    print(y1.max(), y1.min())
    y2 = y1.to(torch.float32) @ w2.to(torch.float32)
    return y2.to(torch.float16)

seq = 16
h_size = 4096
m_size = 12288

x = torch.randn(seq, h_size, dtype=torch.float16, device='cuda') / 10
w1 = torch.randn(h_size, m_size * 2, dtype=torch.float16, device='cuda') / 10
w2 = torch.randn(m_size, h_size, dtype=torch.float16, device='cuda') / 10

y1 = torch_ref(x, w1, w2)
y2 = triton_llama_ffn(x, w1, w2)

print((y1 - y2).abs().max())
# print(y2)
