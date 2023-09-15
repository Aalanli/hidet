import numpy
import hidet
import hllm
import vllm.model_executor.parallel_utils
from hidet.ir.dtypes import float16, float32
from hidet.transforms import lower, PassContext, instruments
from hidet.utils.ncu_utils import ncu_run
from hidet import ops
import numpy as np

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()
# hidet.option.debug_show_var_id()

# hidet.utils.clear_cache_dir()
# hidet.option.parallel_build(False)

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


def demo_type_infer():
    from hidet.ir.dtypes import f16, i32
    from hidet.ir.expr import Var
    from hidet.ir.tile.ops.creation import zeros
    from hidet.lang import tile as ti
    from hidet.ir.tools import infer_type

    pa = Var('pa', ~f16) + zeros([16, 16], dtype=i32)
    pb = Var('pb', ~f16)
    diff = pa - pb
    print(infer_type(diff))


def demo_llama_ffn(seq=16, hidden_size=4096, intermediate_size=12288):
    import torch
    import torch.distributed
    import vllm.worker.worker
    import vllm.config

    hidet.option.search_space(0)

    vllm.worker.worker._init_distributed_environment(
        parallel_config=vllm.config.ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1, worker_use_ray=False
        ),
        rank=0,
        distributed_init_method='tcp://localhost:4444',
    )
    torch.set_grad_enabled(False)

    m_size = intermediate_size
    h_size = hidden_size

    torch_ffn = hllm.models.llama.LlamaMLP(h_size, m_size, hidden_act='silu').eval().half()
    torch.nn.init.normal_(torch_ffn.gate_up_proj.weight, mean=0, std=0.01)
    torch.nn.init.normal_(torch_ffn.down_proj.weight, mean=0, std=0.01)
    w1 = hidet.from_torch(torch_ffn.gate_up_proj.weight.T.contiguous())
    w2 = hidet.from_torch(torch_ffn.down_proj.weight.T.contiguous())

    op = hllm.ops.llama.mlp.LlamaMLPOperator(seq, h_size, m_size)

    def hidet_func(x):
        y2 = torch.empty([seq, h_size], dtype=torch.float16, device='cuda')
        op(x, w1, w2, y2)
        return y2

    def torch_func(x):
        return torch_ffn(x)

    def hidet_origin_func(x):
        x = hidet.from_torch(x)
        y1 = x @ w1
        y1 = ops.silu(y1[:, :m_size]) * y1[:, m_size:]
        y2 = y1 @ w2
        return y2

    x = torch.randn([seq, h_size], dtype=torch.float16, device='cuda')
    y1 = hidet_func(x)
    y2 = torch_func(x)
    # y3 = hidet_origin_func(x)

    # hidet.utils.assert_close(y2, y3, atol=5e-2, rtol=5e-2)
    hidet.utils.assert_close(y1, y2, atol=5e-2, rtol=5e-2)

    print('        torch: {:.3f}'.format(hidet.utils.benchmark_func(lambda: torch_func(x), repeat=100)))
    print('   hidet-tile: {:.3f}'.format(hidet.utils.benchmark_func(lambda: hidet_func(x), repeat=100)))


def main():
    # demo_matmul()
    # ncu_run(demo_llama_ffn, hidden_size=4096, intermediate_size=12288)
    demo_llama_ffn(hidden_size=4096, intermediate_size=12288)


if __name__ == '__main__':
    main()
