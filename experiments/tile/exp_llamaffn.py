# %%
import triton
import triton.language as tl

import torch
print(triton.__version__)

def prune_seq_config(configs, args):
    seq_len = args['seq']
    return [config for config in configs if config.kwargs['block_s'] <= seq_len]

def get_block_s(seq):
    if seq <= 16:
        return 16
    elif seq <= 32:
        return 32
    elif seq <= 64:
        return 64
    raise ValueError(f"seq {seq} not supported")


def generate_configs(zero=False):
    for bm in [32, 64, 128]:
        for bh in [32, 64, 128, 256]:
            for split_h in [1, 2]:
                for nstage in [2, 3, 4, 5]:
                    for nwarp in [4, 8, 16]:
                        pre_hook = lambda x: x['y_ptr'].zero_() if zero else None
                        yield triton.Config({'block_m': bm, 'block_h': bh, 'split_h': split_h}, num_stages=nstage, num_warps=nwarp, pre_hook=pre_hook)

def generate_configs1():
    configs = [
        triton.Config({'block_m': 32, 'block_h': 32,  'block_h1': 32, 'split_h': 1}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 32,  'block_h1': 64, 'split_h': 1}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64, 'split_h': 1}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 128, 'split_h': 1}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64, 'split_h': 1}, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 128, 'split_h': 1}, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 128, 'split_h': 2}, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64, 'split_h': 2}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128, 'block_h1': 128, 'split_h': 1}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128, 'block_h1': 128, 'split_h': 2}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128, 'block_h1': 128, 'split_h': 1}, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128, 'block_h1': 128, 'split_h': 2}, num_stages=4, num_warps=4),
    ]



def generate_configs2(zero=False):
    for bm in [32, 64, 128]:
        for bh in [32, 64, 128, 256]:
            for bh1 in [32, 64, 128, 256]:
                if bh1 >= bh:
                    for split_h in [1, 2]:
                        for nstage in [2, 3, 4]:
                            for nwarp in [4, 8]:
                                pre_hook = lambda x: x['y_ptr'].zero_() if zero else None
                                yield triton.Config({'block_m': bm, 'block_h': bh, 'block_h1': bh1, 'split_h': split_h}, num_stages=nstage, num_warps=nwarp, pre_hook=pre_hook)

def generate_configs3(zero=False):
    for bm in [32, 64, 128]:
        for bh in [32, 64, 128, 256]:
            for bh1 in [32, 64, 128, 256]:
                if bh1 >= bh:
                    for nstage in [2, 3, 4]:
                        for nwarp in [4, 8]:
                            pre_hook = lambda x: x['y_ptr'].zero_() if zero else None
                            yield triton.Config({'block_m': bm, 'block_h': bh, 'block_h1': bh1}, num_stages=nstage, num_warps=nwarp, pre_hook=pre_hook)


@triton.autotune(
    configs=[
        triton.Config({'block_m': 32, 'block_h': 32,  'block_h1': 64, 'split_h': 2},   num_stages=2, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64, 'split_h': 1},   num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128,  'block_h1': 128, 'split_h': 2}, num_stages=4, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128,  'block_h1': 128, 'split_h': 2}, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 128, 'split_h': 2},  num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64, 'split_h': 1},   num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64, 'split_h': 1},   num_stages=3, num_warps=4),
        triton.Config({'block_m': 64, 'block_h': 32,  'block_h1': 64, 'split_h': 1},   num_stages=3, num_warps=4),

    ],
    key=['m_size', 'h_size', 'h_size', 'h_size'],
)
@triton.jit
def triton_fused_ffn(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    block_h1: tl.constexpr,
    split_h: tl.constexpr
):
    pid = tl.program_id(0) // split_h
    pid_h = tl.program_id(0) % split_h

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m
    w2_ptrs = w1_ptrs + m_size
    y1_lhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    y1_rhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq

    m_mask = m_range[None, :] + pid * block_m < m_size
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_lhs = tl.load(w1_ptrs, mask=mask, other=0)
        y1_lhs += tl.dot(x, w1_lhs, out_dtype=tl.float16)
        w1_rhs = tl.load(w2_ptrs, mask=mask, other=0)
        y1_rhs += tl.dot(x, w1_rhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
        w2_ptrs += 2 * m_size * block_h

    y1 = tl.sigmoid(y1_lhs.to(tl.float32)).to(tl.float16) * y1_lhs * y1_rhs

    h_range = tl.arange(0, block_h1)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :] + pid_h * block_h1
    y_ptrs = y_ptr + (s_range[:, None] + pid * seq) * h_size + h_range[None, :] + pid_h * block_h1

    s_mask = s_range[:, None] < seq
    m_mask = m_range[:, None] + pid * block_m < m_size

    for k in range(tl.cdiv(h_size, block_h1 * split_h)):
        h_remaining = h_size - (k * block_h1 * split_h + pid_h * block_h1)
        h_mask = h_range[None, :] < h_remaining
        w2 = tl.load(w2_ptrs, mask=h_mask & m_mask, other=0)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.store(y_ptrs, y, mask=s_mask & h_mask)
        w2_ptrs += block_h1 * split_h
        y_ptrs += block_h1 * split_h

@triton.autotune(
    configs=[
        triton.Config({'block_m': 128, 'block_h': 64,  'block_h1': 64, }, num_stages=4, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 32,  'block_h1': 32,  }, num_stages=4, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 32,  'block_h1': 32,  }, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64,  }, num_stages=4, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64,  }, num_stages=4, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 64,  'block_h1': 64,  }, num_stages=3, num_warps=4),
        triton.Config({'block_m': 32, 'block_h': 128,  'block_h1': 128,}, num_stages=2, num_warps=4),
        triton.Config({'block_m': 128, 'block_h': 64,  'block_h1': 64, }, num_stages=2, num_warps=4),

    ],
    key=['m_size', 'h_size', 'h_size'],
)
@triton.jit
def triton_fused_ffn_concurrent_block(
    x_ptr, # [seq, h_size]
    w1_ptr, # [h_size, 2 * m_size]
    w2_ptr,
    y_ptr,
    y_buf, # [seq, 2 * m_size]
    ready, # [2 * m_size // block_m] also == n_blocks
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    block_h1: tl.constexpr,
):
    pid = tl.program_id(0) // 2
    pid_m = tl.program_id(0) % 2

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m + pid_m * m_size
    y1 = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq

    m_mask = m_range[None, :] + pid * block_m < m_size
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_lhs = tl.load(w1_ptrs, mask=mask, other=0)
        y1 += tl.dot(x, w1_lhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
    
    s_range = tl.arange(0, block_s)
    m_range = tl.arange(0, block_m)
    buf_ptrs = y_buf + s_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m + pid_m * m_size
    tl.store(buf_ptrs, y1, mask=(s_range[:, None] < seq) & (m_range[None, :] + pid * block_m < m_size))
    tl.atomic_xchg(ready + tl.program_id(0), 1)
    other_id = pid * 2 + (pid_m + 1) % 2
    # fingers crossed there is no deadlock, eg blocks are launched in groups of 2
    while tl.atomic_cas(ready + other_id, 1, 0) == 0:
        pass
    # while tl.load(ready + other_id) == 0:
    #     pass
    tl.debug_barrier()
    other_buf_ptrs = y_buf + s_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m + (pid_m + 1) % 2 * m_size
    y1_other = tl.load(other_buf_ptrs, mask=(s_range[:, None] < seq) & (m_range[None, :] + pid * block_m < m_size))
    if pid_m == 0:
        y1 = tl.sigmoid(y1.to(tl.float32)).to(tl.float16) * y1 * y1_other
    else:
        y1 = tl.sigmoid(y1_other.to(tl.float32)).to(tl.float16) * y1 * y1_other
    

    # y1 how contains the result of the first matmul
    # there are automatically two concurrent blocks per h1 row block
    h_range = tl.arange(0, block_h1)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :] + pid_m * block_h
    y_ptrs = y_ptr + (s_range[:, None] + pid * seq) * h_size + h_range[None, :] + pid_m * block_h

    s_mask = s_range[:, None] < seq
    m_mask = m_range[:, None] + pid * block_m < m_size

    split_h = 2
    for k in range(tl.cdiv(h_size, block_h * split_h)):
        h_remaining = h_size - (k * block_h * split_h + pid_m * block_h)
        h_mask = h_range[None, :] < h_remaining
        w2 = tl.load(w2_ptrs, mask=h_mask & m_mask, other=0)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.store(y_ptrs, y, mask=s_mask & h_mask)
        w2_ptrs += block_h * split_h
        y_ptrs += block_h * split_h


@triton.autotune(
    configs=[
        triton.Config({'block_m': 128, 'block_h': 64, 'split_h': 1}, num_warps=4, num_stages=2),
        triton.Config({'block_m': 32, 'block_h': 32, 'split_h': 2}, num_warps=4, num_stages=4),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 1}, num_warps=4, num_stages=4),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 2}, num_warps=4, num_stages=4),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 2}, num_warps=4, num_stages=4),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 1}, num_warps=4, num_stages=5),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 2}, num_warps=4, num_stages=4),
        triton.Config({'block_m': 64, 'block_h': 64, 'split_h': 1}, num_warps=4, num_stages=4),
    ],
    key=['m_size', 'h_size', 'h_size'],
)
@triton.jit
def triton_fused_ffn_loop_split(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    split_h: tl.constexpr
):
    pid = tl.program_id(0) // split_h
    pid_h = tl.program_id(0) % split_h

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m
    y1_lhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq

    m_mask = m_range[None, :] + pid * block_m < m_size
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_lhs = tl.load(w1_ptrs, mask=mask, other=0)
        y1_lhs += tl.dot(x, w1_lhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
    
    y1_lhs = tl.sigmoid(y1_lhs.to(tl.float32)).to(tl.float16) * y1_lhs

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)
    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w2_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m + m_size
    y1_rhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_rhs = tl.load(w2_ptrs, mask=mask, other=0)
        y1_rhs += tl.dot(x, w1_rhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w2_ptrs += 2 * m_size * block_h

    y1 = y1_lhs * y1_rhs

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :] + pid_h * block_h
    y_ptrs = y_ptr + (s_range[:, None] + pid * seq) * h_size + h_range[None, :] + pid_h * block_h

    s_mask = s_range[:, None] < seq
    m_mask = m_range[:, None] + pid * block_m < m_size

    for k in range(tl.cdiv(h_size, block_h * split_h)):
        h_remaining = h_size - (k * block_h * split_h + pid_h * block_h)
        h_mask = h_range[None, :] < h_remaining
        w2 = tl.load(w2_ptrs, mask=h_mask & m_mask, other=0)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.store(y_ptrs, y, mask=s_mask & h_mask)
        w2_ptrs += block_h * split_h
        y_ptrs += block_h * split_h

# monkey patch to get dynamic buffer
import time, builtins
def cdiv(a, b):
    return (a + b - 1) // b

def get_y(**config):
    return torch.empty([cdiv(config['m_size'], config['block_m']), config['seq'], config['h_size']], dtype=torch.float16, device='cuda')

def run(self, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))
    if len(self.configs) > 1:
        all_args = {**self.nargs, **kwargs}
        _args = []
        for name in self.arg_names:
            if name in all_args:
                _args.append(all_args[name])
        key = tuple(_args[i] for i in self.key_idx)
        if key not in self.cache:
            # prune configs
            pruned_configs = self.prune_configs(kwargs)
            config_args = []
            for config in pruned_configs:
                new_args = list(args)
                new_args[3] = get_y(**config.kwargs, **all_args)
                config_args.append((tuple(new_args), config))
            
            bench_start = time.time()
            timings = {config: self._bench(*new_args, config=config, **kwargs)
                        for (new_args, config) in config_args}
            
            bench_end = time.time()
            self.bench_time = bench_end - bench_start
            self.cache[key] = builtins.min(timings, key=timings.get)
            self.hook(args)
            self.configs_timings = timings
        config = self.cache[key]
    else:
        config = self.configs[0]
    self.best_config = config
    full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
    Y = get_y(**full_nargs)
    new_args = list(args)
    new_args[3] = Y
    self.fn.run(*new_args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)
    return Y

triton_fused_ffn.run = lambda *args, **kwargs: run(triton_fused_ffn, *args, **kwargs)
triton_fused_ffn_loop_split.run = lambda *args, **kwargs: run(triton_fused_ffn_loop_split, *args, **kwargs)
triton_fused_ffn_concurrent_block.run = lambda *args, **kwargs: run(triton_fused_ffn_concurrent_block, *args, **kwargs)

def prehook(x):
    x['y_ptr'].zero_()

@triton.autotune(
    configs=[
        triton.Config({'block_m': 64, 'block_h': 32, 'split_h': 2}, num_warps=4, num_stages=2, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 32, 'split_h': 1}, num_warps=4, num_stages=3, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 32, 'split_h': 2}, num_warps=4, num_stages=3, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 2}, num_warps=4, num_stages=3, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 128, 'split_h': 2}, num_warps=4, num_stages=4, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 256, 'split_h': 1}, num_warps=4, num_stages=2, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 2}, num_warps=4, num_stages=3, pre_hook=prehook),
        triton.Config({'block_m': 32, 'block_h': 64, 'split_h': 1}, num_warps=8, num_stages=3, pre_hook=prehook),
    ],
    key=['m_size', 'h_size', 'h_size'],
)
@triton.jit
def triton_fused_ffn_atomic(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    seq,
    h_size,
    m_size,
    block_s: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    split_h: tl.constexpr
):
    pid = tl.program_id(0) // split_h
    pid_h = tl.program_id(0) % split_h

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    x_ptrs = x_ptr + s_range[:, None] * h_size + h_range[None, :]
    w1_ptrs = w1_ptr + h_range[:, None] * 2 * m_size + m_range[None, :] + pid * block_m
    w2_ptrs = w1_ptrs + m_size
    y1_lhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    y1_rhs = tl.zeros((block_s, block_m), dtype=tl.float16)
    s_mask = s_range[:, None] < seq

    m_mask = m_range[None, :] + pid * block_m < m_size
    for k in range(tl.cdiv(h_size, block_h)):
        h_remaining = h_size - k * block_h
        h_mask = h_range < h_remaining
        mask = h_mask[:, None] & m_mask
        x = tl.load(x_ptrs, mask=s_mask & h_mask[None, :], other=0)
        w1_lhs = tl.load(w1_ptrs, mask=mask, other=0)
        y1_lhs += tl.dot(x, w1_lhs, out_dtype=tl.float16)
        w1_rhs = tl.load(w2_ptrs, mask=mask, other=0)
        y1_rhs += tl.dot(x, w1_rhs, out_dtype=tl.float16)
        x_ptrs += block_h
        w1_ptrs += 2 * m_size * block_h
        w2_ptrs += 2 * m_size * block_h

    y1 = tl.sigmoid(y1_lhs.to(tl.float32)).to(tl.float16) * y1_lhs * y1_rhs

    h_range = tl.arange(0, block_h)
    m_range = tl.arange(0, block_m)
    s_range = tl.arange(0, block_s)

    w2_ptrs = w2_ptr + (m_range[:, None] + pid * block_m) * h_size + h_range[None, :] + pid_h * block_h
    y_ptrs = y_ptr + (s_range[:, None]) * h_size + h_range[None, :] + pid_h * block_h

    s_mask = s_range[:, None] < seq
    m_mask = m_range[:, None] + pid * block_m < m_size

    for k in range(tl.cdiv(h_size, block_h * split_h)):
        h_remaining = h_size - (k * block_h * split_h + pid_h * block_h)
        h_mask = h_range[None, :] < h_remaining
        w2 = tl.load(w2_ptrs, mask=h_mask & m_mask, other=0)
        y = tl.dot(y1, w2, out_dtype=tl.float16)
        tl.atomic_add(y_ptrs, y, mask=s_mask & h_mask)
        w2_ptrs += block_h * split_h
        y_ptrs += block_h * split_h

def triton_llama_ffn(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)
    
    grid = lambda META: (triton.cdiv(m_size, META['block_m']) * META['split_h'],)
    y = triton_fused_ffn[grid](
        x, w1, w2, None,
        seq,
        h_size,
        m_size,
        block_s=get_block_s(seq),
    )

    return y.sum(0)


def triton_llama_concurrent_block_ffn(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)
    
    grid = lambda META: (triton.cdiv(m_size, META['block_m']) * 2,)
    y_buf = torch.empty(seq, 2 * m_size, dtype=torch.float16, device='cuda')
    ready = torch.zeros(2 * m_size // 16, dtype=torch.int32, device='cuda')
    y = triton_fused_ffn_concurrent_block[grid](
        x, w1, w2, None, y_buf, ready,
        seq,
        h_size,
        m_size,
        block_s=get_block_s(seq),
    )

    return y.sum(0)


def triton_llama_ffn_loop_split(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)
    
    grid = lambda META: (triton.cdiv(m_size, META['block_m']) * META['split_h'],)
    y = triton_fused_ffn_loop_split[grid](
        x, w1, w2, None,
        seq,
        h_size,
        m_size,
        block_s=get_block_s(seq),
    )

    return y.sum(0)

def triton_llama_ffn_atomic(x, w1, w2):
    seq = x.size(0)
    h_size = w1.size(0)
    m_size = w2.size(0)
    
    y = torch.empty(seq, h_size, dtype=torch.float16, device='cuda')
    grid = lambda META: (triton.cdiv(m_size, META['block_m']) * META['split_h'],)
    triton_fused_ffn_atomic[grid](
        x, w1, w2, y,
        seq,
        h_size,
        m_size,
        block_s=get_block_s(seq),
    )

    return y

def torch_ref(x, w1, w2):
    m_size = w2.size(0)
    x = x @ w1
    x = torch.nn.functional.silu(x[:, :m_size]) * x[:, m_size:]
    y2 = x @ w2
    return y2

def demo_triton():

    seq = 16
    h_size = 4096
    m_size = h_size * 3

    x =  torch.randn(seq, h_size, dtype=torch.float16, device='cuda') / 10
    w1 = torch.randn(h_size, m_size * 2, dtype=torch.float16, device='cuda') / 10
    w2 = torch.randn(m_size, h_size, dtype=torch.float16, device='cuda') / 10

    y1 = torch_ref(x, w1, w2)
    y2 = triton_llama_ffn(x, w1, w2)
    y21 = triton_llama_ffn_loop_split(x, w1, w2)
    y3 = triton_llama_ffn_atomic(x, w1, w2)
    y4 = triton_llama_concurrent_block_ffn(x, w1, w2)
    torch.cuda.synchronize()

    # print(y1)
    # print(y2)

    print(torch.allclose(y1, y2, atol=5e-2, rtol=5e-2))
    print(torch.allclose(y1, y3, atol=5e-2, rtol=5e-2))
    print(torch.allclose(y1, y21, atol=5e-2, rtol=5e-2))
    print(torch.allclose(y1, y4, atol=5e-2, rtol=5e-2))
    return y1, y21


def bench_triton():
    seq = 16
    h_size = 4096
    m_size = 12288

    x =  torch.randn(seq, h_size, dtype=torch.float16, device='cuda') / 10
    w1 = torch.randn(h_size, m_size * 2, dtype=torch.float16, device='cuda') / 10
    w2 = torch.randn(m_size, h_size, dtype=torch.float16, device='cuda') / 10

    y1 = torch_ref(x, w1, w2)
    y2 = triton_llama_ffn(x, w1, w2)
    y21 = triton_llama_ffn_loop_split(x, w1, w2)
    y3 = triton_llama_ffn_atomic(x, w1, w2)
    y4 = triton_llama_concurrent_block_ffn(x, w1, w2)

bench_triton()
# demo_triton()
# %%
# y1, y4 = demo_triton()
# # %%
# print((y1 - y4).abs().max())

# # %% 
# if __name__ == "__main__":
#     from hidet.utils.ncu_utils import ncu_run
#     ncu_run(bench_triton, no_bench=True).visualize()

# y1, y2, y3 = demo_triton()
# # %%
# print((y1 - y3).abs().max())

# # %%
# for M in [1, 2, 4, 16]:
#     @triton.testing.perf_report(
#         triton.testing.Benchmark(
#             x_names=['D'],  # Argument names to use as an x-axis for the plot
#             x_vals=[
#                 32, 64, 128, 256, 512, 1024, 2048, 4096
#             ],  # Different possible values for `x_name`
#             line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
#             # Possible values for `line_arg`
#             line_vals=['torch_naive', 'triton_fused', 'triton_fused_loop_split', 'triton_fused_atomic', 'triton_concurrent_block'],
#             # Label name for the lines
#             line_names=['torch_naive', 'triton_fused', 'triton_fused_loop_split', 'triton_fused_atomic', 'triton_concurrent_block'],
#             # Line styles
#             styles=[('green', '-'), ('blue', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-')],
#             ylabel="ms",  # Label name for the y-axis
#             plot_name=f"mlp-performance-M={M}",  # Name for the plot, used also as a file name for saving the plot.
#             args={},
#         )
#     )
#     def benchmark(D, provider):
#         m_size = 3 * D
#         a = torch.randn([M, D], dtype=torch.float16, device='cuda')
#         w1 = torch.randn([D, m_size * 2], dtype=torch.float16, device='cuda')
#         w2 = torch.randn([m_size, D], dtype=torch.float16, device='cuda')

#         quantiles = [0.5, 0.2, 0.8]
#         if provider == 'torch_naive':
#             ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_ref(a, w1, w2), quantiles=quantiles)
#         if provider == 'triton_fused':
#             ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_llama_ffn(a, w1, w2), quantiles=quantiles)
#         if provider == 'triton_fused_loop_split':
#             ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_llama_ffn_loop_split(a, w1, w2), quantiles=quantiles)
#         if provider == 'triton_fused_atomic':
#             ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_llama_ffn_atomic(a, w1, w2), quantiles=quantiles)
#         if provider == 'triton_concurrent_block':
#             ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_llama_concurrent_block_ffn(a, w1, w2), quantiles=quantiles)
#         # if provider == 'triton_default':
#         #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: two_triton(a, w1, w2), quantiles=quantiles)        
#         # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#         # return perf(ms), perf(max_ms), perf(min_ms)
#         return ms, max_ms, min_ms


#     benchmark.run(show_plots=True, print_data=True)

# # %%
# for k, v in triton_fused_ffn.cache.items():
#     print(k, v)

# # %%
# for k, v in triton_fused_ffn_loop_split.cache.items():
#     print(k, v)
