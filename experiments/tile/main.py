import hidet
from hidet.ir.dtypes import float16, float32
from hidet.transforms import lower, PassContext, instruments
from hidet.utils.ncu_utils import ncu_run

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()
# hidet.option.debug_show_var_id()

hidet.utils.clear_cache_dir()


import numpy
numpy.set_printoptions(precision=2, edgeitems=64, linewidth=256)


def demo_arange():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange(a_ptr: ~f32, b_ptr: ~f32, n: int32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            mask = a < n
            a_ptrs = a_ptr + a
            b_ptrs = b_ptr + a
            ti.store(b_ptrs, ti.load(a_ptrs, mask) + 1, mask)

    # lower_script_module(script_module)
    func = script_module.build()
    n = 16
    a = hidet.randn([16], dtype=hidet.float32, device='cuda')
    b = hidet.empty([16], dtype=hidet.float32, device='cuda')

    func(a, b, n)
    hidet.utils.assert_close(b, a + 1)


def demo_debug_print():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange():
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            ti.debug_print(a)

            b = ti.full(1, [4, 4])
            ti.debug_print(b)

            c = ti.full(1, [4, 4, 4])
            ti.debug_print(c)

            d = ti.full(1, [16, 16])
            ti.debug_print(d)

    # lower_script_module(script_module)
    func = script_module.build()
    func()


def demo_vector_add(n: int = 1024):
    from hidet.lang.types import f32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    block_size = 128

    with hidet.script_module() as script_module:
        @hidet.script
        def vec_add(a: ~f32, b: ~f32, c: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = (n + block_size - 1) // block_size

            pid = ti.program_id()
            offsets = pid * block_size + ti.arange(0, block_size)
            mask = offsets < n

            result = ti.load(a + offsets, mask=mask) + ti.load(b + offsets, mask=mask)
            ti.store(c + offsets, result, mask=mask)

    a = hidet.randn([n], dtype=hidet.float32, device='cuda')
    b = hidet.randn([n], dtype=hidet.float32, device='cuda')
    c = hidet.zeros([n], dtype=hidet.float32, device='cuda')
    func = script_module.build()
    func(a, b, c)
    hidet.utils.assert_close(c, a + b)


def demo_expand_dims():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange():
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            b = ti.arange(0, 16)
            c = ti.expand_dims(a, 1) * 16 + ti.expand_dims(b, 0)
            ti.debug_print(c)

    func = script_module.build()
    func()


def demo_for_and_increment():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange():
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            for k in range(10):
                a += 1
            ti.debug_print(a)

    func = script_module.build()
    func()


def demo_reduce():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange():
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            b = ti.arange(0, 16)
            c = ti.expand_dims(a, 1) * 16 + ti.expand_dims(b, 0)
            ti.debug_print(c)
            ti.debug_print(ti.sum(c, axis=0, keepdims=False))
            ti.debug_print(ti.sum(c, axis=1, keepdims=False))
            ti.debug_print(ti.min(c, axis=0, keepdims=False))
            ti.debug_print(ti.min(c, axis=1, keepdims=False))
            ti.debug_print(ti.max(c, axis=0, keepdims=False))
            ti.debug_print(ti.max(c, axis=1, keepdims=False))

    func = script_module.build()
    func()


def demo_dot_simt():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange():
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.ones([16, 16])
            b = ti.ones([16, 16])
            c = ti.dot(a, b)
            ti.debug_print(c)

    func = script_module.build()
    func()


def demo_ldst():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    block_m = 128
    block_k = 16

    with hidet.script_module() as script_module:
        @hidet.script
        def matmul(a_ptr: ~f32, b_ptr: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            offsets = ti.expand_dims(ti.arange(0, block_m), axis=1) * block_k + ti.arange(0, block_k)
            a_ptrs = a_ptr + offsets
            b_ptrs = b_ptr + offsets
            ti.store(b_ptrs, ti.load(a_ptrs))

    a = hidet.randn([block_m, block_k], device='cuda')
    b = hidet.empty([block_m, block_k], device='cuda')
    func = script_module.build()
    func(a, b)

    print(a)
    print(b)
    hidet.utils.assert_close(a, b)


def demo_ldgsts_lds128():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs, cast, shared_tensor, register_tensor
    from hidet.lang.cuda import threadIdx, cp_async, cp_async_wait_all
    from hidet.ir.mapping import spatial_map

    with hidet.script_module() as script_module:
        @hidet.script
        def func(out: f32[256]):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 32
            attrs.cuda.grid_dim = 1

            s = 0.0
            tid = threadIdx.x
            a = shared_tensor('float32', shape=[256])

            # ldgsts
            # L1 exc, L1, L1 ideal, L2 exc, L2, L2 ideal
            # 0	4	4	0	16	16
            cp_async(
                dst=~a[tid * 4],
                src=~out[tid * 4],
                cp_size=16
            )
            cp_async_wait_all()

            # 4	8	4	16	32	16
            cp_async(
                dst=~a[((3 - (tid // 8)) * 8 + tid % 8) * 4],
                src=~out[tid * 8],
                cp_size=16
            )
            cp_async_wait_all()

            # 4	8	4	16	32	16
            cp_async(
                dst=~a[((3 - (tid // 8)) * 8 + tid % 8) * 8],
                src=~out[tid * 8],
                cp_size=16
            )
            cp_async_wait_all()

            # lds
            b = register_tensor('float32', shape=[4])
            c = register_tensor('float32', shape=[4])
            for i in range(4):
                b[i] = a[tid * 4 + i]
            for i in range(4):
                group_id = tid // 8
                id_in_group = tid % 8
                c[i] = a[((3 - group_id) * 8 + id_in_group) * 4 + i]
            for i in range(4):
                s += b[i]
            for i in range(4):
                s += c[i]

            for i in range(4):
                out[tid] = s

    func = script_module.build()
    out = hidet.empty([256], device='cuda')
    func(out)


def demo_matmul_x2(m_size=1024, n_size=1024, k_size=1024, dtype='float32', bench=False):
    from hidet.ir.type import data_type
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs, cast
    from hidet.lang import tile as ti

    num_warps = 4
    block_m = 128
    block_n = 64
    block_k = 32
    dtype = data_type(dtype)

    with hidet.script_module() as script_module:
        @hidet.script
        def matmul(a_ptr: ~dtype, b_ptr: ~dtype, c_ptr: ~dtype, d_ptr: ~dtype):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = num_warps * 32
            attrs.cuda.grid_dim = (m_size // block_m) * (n_size // block_n)

            pid = ti.program_id()
            num_n_blocks = n_size // block_n
            pid_m = pid // num_n_blocks
            pid_n = pid % num_n_blocks

            m_offsets = pid_m * block_m + ti.arange(0, block_m)
            n_offsets = pid_n * block_n + ti.arange(0, block_n)
            a_ptrs = a_ptr + ti.expand_dims(m_offsets * k_size, 1) + ti.arange(0, block_k)
            b_ptrs = b_ptr + ti.expand_dims(ti.arange(0, block_k) * n_size, 1) + n_offsets
            c = ti.zeros([block_m, block_n])

            for k in range(k_size // block_k):
                a = ti.load(a_ptrs)
                b = ti.load(b_ptrs)
                c += ti.dot(a, b)

                a_ptrs += block_k
                b_ptrs += block_k * n_size

            c_ptrs = c_ptr + ti.expand_dims(m_offsets * n_size, 1) + n_offsets
            ti.store(c_ptrs, c)

    func = script_module.build()

    a = hidet.randn([m_size, k_size], device='cuda')
    b = hidet.randn([k_size, n_size], device='cuda')
    c = hidet.empty([m_size, n_size], dtype='float32', device='cuda')

    func(a, b, c)
    if bench:
        print('  tile: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: func(a, b, c), repeat=20)))

    import torch
    ta, tb, tc = a.torch(), b.torch(), c.torch().clone()
    torch.matmul(ta, tb, out=tc)
    if bench:
        print(' torch: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: torch.matmul(ta, tb, out=tc), repeat=20)))

    import numpy
    numpy.set_printoptions(precision=2, edgeitems=64, linewidth=256)
    # print(c.cpu().numpy())
    # print(tc.cpu().numpy())

    hidet.utils.assert_close(c, tc, atol=1e-4, rtol=1e-4)


def demo_matmul(m_size=1024, n_size=1024, k_size=1024, dtype='float32', bench=False):
    from hidet.ir.type import data_type
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs, cast
    from hidet.lang import tile as ti

    num_warps = 4
    block_m = 128
    block_n = 64
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


def main():
    for m_size, n_size, k_size in [
        [1024, 1024, 1024],
        # [1023, 1023, 1024],
        # [1024, 1024, 1023],
        # [1111, 1111, 1111]
    ]:
        demo_matmul(m_size, n_size, k_size, dtype='float32', bench=True)
        # report = ncu_run(demo_matmul, m_size, n_size, k_size, dtype='float32')
        # report.visualize()


if __name__ == '__main__':
    main()
