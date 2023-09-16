# %%
import triton_utils
import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({}, num_warps=1, num_stages=1)
], key=[])
@triton.jit
def test(a, b, c):
    a_idx = tl.arange(0, 32)
    b_idx = tl.arange(0, 16)

    a_ptr = a + a_idx
    b_ptr = b + b_idx
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    cres = a[:, None] + b[None, :]
    tl.store(c + (a_idx[:, None] * 16 + b_idx[None, :]), cres)


import torch
a = torch.randn(32, device='cuda')
b = torch.randn(16, device='cuda')
c = torch.empty([32, 16], device='cuda')

test[(1,)](a, b, c)
print(torch.allclose(c, a[:, None] + b[None, :]))

