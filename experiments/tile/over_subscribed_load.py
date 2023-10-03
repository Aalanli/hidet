# %%
import triton
import triton.language as tl

# @triton.autotune(configs=[
#     triton.Config({}, num_warps=1, num_stages=1)
# ], key=[])
@triton.jit
def test(a, b, c):
    a_idx = tl.arange(0, 32)
    b_idx = tl.arange(0, 64)

    a_ptr = a + a_idx
    b_ptr = b + b_idx
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    cres = b[:, None] + a[None, :]
    tl.store(c + (b_idx[:, None] * 32 + a_idx[None, :]), cres)


import torch
a = torch.randn(32, device='cuda')
b = torch.randn(64, device='cuda')
c = torch.empty([64, 32], device='cuda')

test[(1,)](a, b, c)
print(torch.allclose(c, b[:, None] + a[None, :]))
from triton.compiler import compile

n_warps = 2
compiled = compile(test, num_warps=n_warps, signature='*fp32, *fp32, *fp32')

path = f'triton-ir/over_sub_w{n_warps}'
with open(path + '.ttir', 'w') as f:
    f.write(str(compiled.asm['ttir']))
with open(path + '.ttgir', 'w') as f:
    f.write(str(compiled.asm['ttgir']))
with open(path + '.ptx', 'w') as f:
    f.write(str(compiled.asm['ptx']))

