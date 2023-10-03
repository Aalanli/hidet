# %%
import triton
import triton.language as tl


@triton.jit
def test1(a, b, c):
    a_idx1 = tl.arange(0, 32)
    a_idx2 = tl.arange(0, 64)
    b_idx1 = tl.arange(0, 8)
    b_idx2 = tl.arange(0, 64)
    d_idx = tl.arange(0, 2)
    
    a_idx = d_idx[:, None, None] * 2048 + (a_idx2[None, :, None] * 32 + a_idx1[None, None, :])
    b_idx = d_idx[:, None, None] * 512 + (b_idx2[None, :, None] * 8 + b_idx1[None, None, :])

    a_ptr = a + a_idx
    b_ptr = b + b_idx
    a1 = tl.load(a_ptr)
    b1 = tl.load(b_ptr)
    c1 = b1 * tl.sum(tl.sum(a1, 2), 1)[:, None, None]

    tl.store(c + b_idx, c1)

import torch
a = torch.randn([2, 64, 32], device='cuda', dtype=torch.float32)
b = torch.randn([2, 64, 8], device='cuda', dtype=torch.float32)
c = torch.zeros([2, 64, 8], device='cuda', dtype=torch.float32)
test1[(1,)](a, b, c)

c1 = b * a.sum(2, keepdim=True).sum(1, keepdim=True)
print(torch.allclose(c1, c))

# %%

from triton.compiler import compile

n_warps = 4
from collections import namedtuple
Spec = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])
specialization = Spec(divisible_by_16={0, 1, 2}, equal_to_1=set())

compiled = compile(test1, num_warps=n_warps, configs=[specialization], signature='*fp32, *fp32, *fp32')

path = f'triton-ir/reduce/double_expansion_reduction{n_warps}'
with open(path + '.ttir', 'w') as f:
    f.write(str(compiled.asm['ttir']))
with open(path + '.ttgir', 'w') as f:
    f.write(str(compiled.asm['ttgir']))
with open(path + '.ptx', 'w') as f:
    f.write(str(compiled.asm['ptx']))