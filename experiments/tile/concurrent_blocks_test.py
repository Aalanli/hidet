# %%
import hidet
from hidet.lang import attrs, f32, boolean
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads


BLOCK_DIM = 1024
GRID_DIM = 1024

with hidet.script_module() as module:
    @hidet.script
    def kernel(
        xs: ~f32, # [64] 
        ready: ~boolean, # [2] init to false
        buf: ~f32, # [64]
        ys: ~f32 # [64]
    ):
        attrs.cuda.block_dim = BLOCK_DIM
        attrs.cuda.grid_dim = GRID_DIM

        tid = threadIdx.x
        pid = blockIdx.x

        x1 = xs[tid + pid * BLOCK_DIM]
        buf[tid + pid * BLOCK_DIM] = x1
        syncthreads()
        other = (pid // 2) * 2 + (pid + 1) % 2
        if tid == 0:
            ready[pid] = True
        
            while buf[other] == False:
                pass
        
        syncthreads()
        x2 = buf[tid + BLOCK_DIM * other]

        if pid % 2 == 0:
            ys[tid + BLOCK_DIM * pid] = x1 + x2
        else:
            ys[tid + BLOCK_DIM * pid] = x1 - x2

func = module.build()

xs = hidet.ones([BLOCK_DIM * GRID_DIM], device='cuda')
ready = hidet.zeros([GRID_DIM], device='cuda', dtype=boolean)
buf = hidet.zeros([BLOCK_DIM * GRID_DIM], device='cuda') + 1000
ys = hidet.zeros([BLOCK_DIM * GRID_DIM], device='cuda') - 1
func(xs, ready, buf, ys)
print(ys)

import torch
ts = torch.zeros([GRID_DIM, BLOCK_DIM], device='cuda')
ts[::2, :] = 2
ts = ts.reshape(-1).contiguous()
ys = ys.torch()
print(torch.allclose(ts, ys))
print((ts - ys).abs().max())