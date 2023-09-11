import hidet
import hllm
import vllm.model_executor.parallel_utils
from hidet.ir.dtypes import float16, float32
from hidet.transforms import lower, PassContext, instruments
from hidet.utils.ncu_utils import ncu_run
import numpy as np

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()
# hidet.option.debug_show_var_id()

hidet.utils.clear_cache_dir()

import numpy

numpy.set_printoptions(precision=2, edgeitems=64, linewidth=256)


def demo_cp_async():
    from hidet.lang import attrs, printf, cast, deref
    from hidet.lang.types import f16, f32, int32
    from hidet.lang.cuda import cp_async, shared_tensor, threadIdx, cp_async_wait_all
    from hidet.ir.expr import left_shift

    with hidet.script_module() as script_module:
        @hidet.script
        def func(ptr: ~f16):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32

            smem = shared_tensor(f16, shape=[1])

            if threadIdx.x == 0:
                cp_async(dst=smem, src=ptr, cp_size=2, src_size=2)
                cp_async_wait_all()
                a = deref(cast(~smem[0], ~int32))
                for i in range(32):
                    if a & left_shift(1, 31 - i):
                        printf('1')
                    else:
                        printf('0')
                printf('\n')

    func = script_module.build()

    a = hidet.full(shape=[1], fill_value=0b00001111000011110000111100001111, dtype='int32', device='cuda')
    func(a)


def demo_matmul(m_size=1024, n_size=1024, k_size=1024, dtype='float32', bench=False):
    from hidet.ir.type import data_type
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs, cast
    from hidet.lang import tile as ti

    num_warps = 8
    block_m = 32
    block_n = 128
    block_k = 32
    dtype = data_type(dtype)

    with hidet.script_module() as script_module:
        @hidet.script
        def matmul(a_ptr: ~dtype, b_ptr: ~dtype, c_ptr: ~dtype):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = num_warps * 32
            attrs.cuda.grid_dim = ((m_size + block_m - 1) // block_m) * ((n_size + block_n - 1) // block_n)

            pid = ti.program_id()
            num_n_blocks = (n_size + block_n - 1) // block_n
            pid_m = pid // num_n_blocks
            pid_n = pid % num_n_blocks

            m_offsets = pid_m * block_m + ti.arange(0, block_m)
            n_offsets = pid_n * block_n + ti.arange(0, block_n)
            k_offsets = ti.arange(0, block_k)
            a_ptrs = a_ptr + ti.expand_dims(m_offsets * k_size, 1) + ti.arange(0, block_k)
            b_ptrs = b_ptr + ti.expand_dims(ti.arange(0, block_k) * n_size, 1) + n_offsets
            c = ti.zeros([block_m, block_n], dtype=dtype)

            for k in range((k_size + block_k - 1) // block_k):
                a = ti.load(a_ptrs, mask=ti.expand_dims(k_offsets, axis=0) < k_size - k * block_k)
                b = ti.load(b_ptrs, mask=ti.expand_dims(k_offsets, axis=1) < k_size - k * block_k)
                c += ti.dot(a, b)

                a_ptrs += block_k
                b_ptrs += block_k * n_size

            c_ptrs = c_ptr + ti.expand_dims(m_offsets * n_size, 1) + n_offsets
            c_mask = ti.expand_dims(m_offsets, axis=1) < m_size and ti.expand_dims(n_offsets, axis=0) < n_size
            ti.store(c_ptrs, ti.cast(c, dtype), mask=c_mask)

    func = script_module.build()

    a = hidet.randn([m_size, k_size], dtype=dtype, stddev=0.1, device='cuda')
    b = hidet.randn([k_size, n_size], dtype=dtype, stddev=0.1, device='cuda')
    c = hidet.empty([m_size, n_size], dtype=dtype, device='cuda')

    if bench:
        print('{} x {} x {}'.format(m_size, n_size, k_size))

    func(a, b, c)
    if bench:
        print('  tile: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: func(a, b, c), repeat=20)))

    import torch
    ta, tb, tc = a.torch(), b.torch(), c.torch().clone()
    torch.matmul(ta, tb, out=tc)
    if bench:
        print(' torch: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: torch.matmul(ta, tb, out=tc), repeat=20)))

    if dtype == float16:
        atol, rtol = 5e-2, 5e-2
    elif dtype == float32:
        atol, rtol = 1e-4, 1e-4
    else:
        assert False
    hidet.utils.assert_close(c, tc, atol=atol, rtol=rtol)


def bench_matmul():
    for m_size, n_size, k_size in [
        # [32, 4096, 4096],
        # [32, 12288, 4096],
        [16, 12288, 4096],
        # [1023, 1023, 1024],
        # [1024, 1024, 1023],
        # [1111, 1111, 1111]
    ]:
        demo_matmul(m_size, n_size, k_size, dtype='float16', bench=True)
        # report = ncu_run(demo_matmul, m_size, n_size, k_size, dtype='float32')
        # report.visualize()


def demo_llama_ffn(seq=16, hidden_size=4096, intermediate_size=11008):
    import torch
    import torch.distributed
    import vllm.worker.worker
    import vllm.config
    from hidet.lang import attrs
    from hidet.lang.types import f16
    from hidet.lang import tile as ti

    vllm.worker.worker._init_distributed_environment(
        parallel_config=vllm.config.ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1, worker_use_ray=False
        ),
        rank=0,
        distributed_init_method='tcp://localhost:4444',
    )

    # x: [seq, h]
    # w1: [h, 2m]
    # w2: [m, h]
    # y1 = x @ w1           # [seq, 2m]
    # y2 = silu_and_mul(y1) # [seq, m]
    # y3 = y2 @ w2          # [seq, h]

    h_size = hidden_size
    m_size = intermediate_size

    block_s = (seq + 15) // 16 * 16
    block_m = 32
    block_h = 64

    assert m_size % block_m == 0
    assert h_size % block_h == 0

    with hidet.script_module() as script_module:
        @hidet.script
        def llama_ffn(x_ptr: ~f16, w1_ptr: ~f16, w2_ptr: ~f16, y_ptr: ~f16):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = intermediate_size // block_m

            pid = ti.program_id()

            x_ptrs = x_ptr + ti.expand_dims(ti.arange(0, block_s), axis=1) * h_size + ti.arange(0, block_h)
            w1_ptrs = (
                w1_ptr + ti.expand_dims(ti.arange(0, block_h), axis=1) * (2 * m_size) + ti.arange(0, block_m)
                + pid * block_m
            )

            y1_lhs = ti.zeros([block_s, block_m], dtype=f16)
            y1_rhs = ti.zeros([block_s, block_m], dtype=f16)

            for k in range(h_size // block_h):
                x = ti.load(x_ptrs)  # [block_s, block_h]
                w1_lhs = ti.load(w1_ptrs)  # [block_h, block_m]
                w1_rhs = ti.load(w1_ptrs + m_size)
                y1_lhs += ti.dot(x, w1_lhs)
                y1_rhs += ti.dot(x, w1_rhs)
                x_ptrs += block_m
                w1_ptrs += 2 * m_size * block_h

            y1 = ti.silu(y1_lhs) * y1_rhs  # [block_s, block_m]

            w2_ptrs = (
                w2_ptr + ti.expand_dims(ti.arange(0, block_m) + pid * block_m, axis=1) * h_size + ti.arange(0, block_h)
            )
            y_ptrs = y_ptr + ti.expand_dims(ti.arange(0, block_s) + pid * seq, axis=1) * h_size + ti.arange(0, block_h)

            mask = ti.expand_dims(ti.arange(0, block_s), axis=1) < seq

            for k in range(h_size // block_h):
                w2 = ti.load(w2_ptrs)  # [block_m, block_h]
                y = ti.dot(y1, w2)  # [block_s, block_h]
                ti.store(ptr=y_ptrs, value=y, mask=mask)
                w2_ptrs += block_h
                y_ptrs += block_h

    func1 = script_module.build()

    reduce_block_h = 64
    reduce_block_s = block_s
    assert h_size % reduce_block_h == 0

    with hidet.script_module() as script_module:
        @hidet.script
        def reduce(x_ptr: ~f16, y_ptr: ~f16):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = h_size // reduce_block_h

            pid = ti.program_id()
            h_offsets = pid * reduce_block_h + ti.arange(0, reduce_block_h)

            x_ptrs = (
                x_ptr
                + ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) * h_size
                + h_offsets
            )
            acc = ti.zeros([reduce_block_s, reduce_block_h], dtype=f16)
            for k in range((m_size // block_m) * seq // reduce_block_s):
                mask = (
                    ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) < m_size // block_m * seq - k * reduce_block_s
                )
                acc += ti.load(x_ptrs, mask=mask)
                x_ptrs += reduce_block_s * h_size

            ti.store(
                ptr=y_ptr + ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) * h_size + h_offsets,
                value=acc,
                mask=ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) < seq
            )

    func2 = script_module.build()

    torch_ffn = hllm.models.llama.LlamaMLP(hidden_size=h_size, intermediate_size=m_size, hidden_act='silu')

    def hidet_func(x):
        y1 = torch.empty([seq * (m_size // block_m), h_size], dtype=torch.float16, device='cuda')
        y2 = torch.empty([seq, h_size], dtype=torch.float16, device='cuda')
        w1 = torch_ffn.gate_up_proj.weight
        w2 = torch_ffn.down_proj.weight
        func1(x, w1, w2, y1)
        func2(y1, y2)
        return y2

    def torch_func(x):
        return torch_ffn(x)

    x = torch.randn([seq, h_size], dtype=torch.float16, device='cuda')
    y1 = hidet_func(x)
    y2 = torch_func(x)

    hidet.utils.assert_close(y1, y2, atol=1e-2, rtol=1e-2)


def main():
    demo_llama_ffn()


if __name__ == '__main__':
    main()
