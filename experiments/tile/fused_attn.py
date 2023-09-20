# %%
import hidet
from hidet.ir.dtypes import float16, float32
import torch

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()

hidet.utils.clear_cache_dir()

def demo_flash_attn(B, H, M, N, D, dtype, BLOCK_M, BLOCK_N, BLOCK_D, num_warps=4):
    from hidet.ir.type import data_type
    from hidet.lang.types import f32, f16
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    dtype = data_type(dtype)
    q_part = ti.cdiv(M, BLOCK_M)
    num_blocks = q_part * B * H
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and D == BLOCK_D

    with hidet.script_module() as script_module:
        @hidet.script
        def flash_attn(q_ptr: ~dtype, k_ptr: ~dtype, v_ptr: ~dtype, y_ptr: ~dtype):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = num_warps * 32
            attrs.cuda.grid_dim = num_blocks

            pid = ti.program_id()
            bh_id = pid // q_part
            pid_m = pid % q_part
            offset_q = bh_id * M * D + pid_m * BLOCK_M * D
            offset_kv = bh_id * N * D

            midx = ti.arange(0, BLOCK_M)
            nidx = ti.arange(0, BLOCK_N)
            didx = ti.arange(0, BLOCK_D)

            q_ptrs = q_ptr + ti.expand_dims(midx * D, 1) + ti.expand_dims(didx, 0) + offset_q
            k_ptrs = k_ptr + ti.expand_dims(nidx * D, 0) + ti.expand_dims(didx, 1) + offset_kv
            v_ptrs = v_ptr + ti.expand_dims(nidx * D, 1) + ti.expand_dims(didx, 0) + offset_kv
            
            q = ti.load(q_ptrs)
            maxes = ti.zeros([BLOCK_M], dtype=f32) - float('inf')
            sums  = ti.zeros([BLOCK_M], dtype=f32)
            acc = ti.zeros([BLOCK_M, BLOCK_D], dtype=f32)

            for ki in range((N + BLOCK_N - 1) // BLOCK_N):
                k = ti.load(k_ptrs)
                qk = ti.dot(q, k)
                qk1 = ti.cast(qk, f32)
                new_max = ti.maximum(ti.max(qk1, 1), maxes)
                alpha = ti.exp(maxes - new_max)
                acc *= ti.expand_dims(alpha, 1)
                p = ti.exp(qk1 - ti.expand_dims(new_max, 1))
                sums = sums * alpha + ti.sum(p, 1)
                maxes = new_max
                p1 = ti.cast(p, dtype)
                v = ti.load(v_ptrs)
                acc += ti.dot(p1, v)
                k_ptrs += BLOCK_N * D
                v_ptrs += BLOCK_N * D
            
            acc1 = acc / ti.expand_dims(sums, 1)
            acc2 = ti.cast(acc1, dtype)
            midx = ti.arange(0, BLOCK_M)
            didx = ti.arange(0, BLOCK_D)
            y_ptrs = y_ptr + ti.expand_dims(midx * D, 1) + ti.expand_dims(didx, 0) + offset_q
            ti.store(y_ptrs, acc2)
    
    func = script_module.build()
    def run(q, k, v):
        assert k.shape == v.shape == (B, H, N, D)
        assert q.dtype == dtype
        Y = hidet.empty([B, H, M, D], dtype=q.dtype, device=q.device)
        func(q, k, v, Y)
        return Y
    return run

def attn_ref(q, k, v):
    return torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v

B = 2
H = 4
M = 32
N = 32
D = 64
q = torch.randn(B, H, M, D, dtype=torch.float16, device='cuda')
k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
run = demo_flash_attn(B, H, M, N, D, float16, BLOCK_M=32, BLOCK_N=32, BLOCK_D=64)

y1 = attn_ref(q, k, v)
qh = hidet.from_torch(q)
kh = hidet.from_torch(k)
vh = hidet.from_torch(v)
y2 = run(qh, kh, vh)


