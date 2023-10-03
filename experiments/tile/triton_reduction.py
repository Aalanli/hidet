# %%
import triton
import triton.language as tl

# %%
@triton.jit
def test1(a, b, c):
    a_idx1 = tl.arange(0, 32)
    a_idx2 = tl.arange(0, 64)
    b_idx1 = tl.arange(0, 16)
    b_idx2 = tl.arange(0, 64)
    
    a_idx = a_idx2[:, None] * 32 + a_idx1[None, :]
    b_idx = b_idx2[:, None] * 16 + b_idx1[None, :]

    a_ptr = a + a_idx
    b_ptr = b + b_idx
    a1 = tl.load(a_ptr)
    b1 = tl.load(b_ptr)
    c1 = b1 * tl.sum(a1, 1)[:, None]

    tl.store(c + b_idx, c1)


from triton.compiler import compile

n_warps = 4
compiled = compile(test1, num_warps=n_warps, signature='*fp32, *fp32, *fp32')

path = f'triton-ir/reduce/reduction_split_w{n_warps}'
with open(path + '.ttir', 'w') as f:
    f.write(str(compiled.asm['ttir']))
with open(path + '.ttgir', 'w') as f:
    f.write(str(compiled.asm['ttgir']))
with open(path + '.ptx', 'w') as f:
    f.write(str(compiled.asm['ptx']))

# %%
@triton.jit
def test1(a, c):
    a_idx1 = tl.arange(0, 32)
    a_idx2 = tl.arange(0, 64)
    
    a_idx = a_idx2[:, None] * 32 + a_idx1[None, :]

    a_ptr = a + a_idx
    a1 = tl.load(a_ptr)
    c1 = a1 * tl.sum(a1, 1)[:, None]

    tl.store(c + a_idx, c1)


from triton.compiler import compile

n_warps = 1
compiled = compile(test1, num_warps=n_warps, signature='*fp32, *fp32')

path = f'triton-ir/reduce/reduction_split2_w{n_warps}'
with open(path + '.ttir', 'w') as f:
    f.write(str(compiled.asm['ttir']))
with open(path + '.ttgir', 'w') as f:
    f.write(str(compiled.asm['ttgir']))
with open(path + '.ptx', 'w') as f:
    f.write(str(compiled.asm['ptx']))

