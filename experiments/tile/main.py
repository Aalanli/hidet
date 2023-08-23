import hidet
from hidet.transforms import lower, PassContext, instruments
from hidet.utils.ncu_utils import ncu_run

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()
# hidet.option.debug_show_var_id()

hidet.utils.clear_cache_dir()


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


def demo_vector_add(n: int = 128):
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

    report = ncu_run(func, a, b)
    report.visualize()

    print(a)
    print(b)
    hidet.utils.assert_close(a, b)


def demo_matmul():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    m_size = 1024
    n_size = 1024
    k_size = 1024

    block_m = 64
    block_n = 32
    block_k = 8

    with hidet.script_module() as script_module:
        @hidet.script
        def matmul(a_ptr: ~f32, b_ptr: ~f32, c_ptr: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = (m_size // block_m) * (n_size // block_n)

            pid = ti.program_id()
            num_n_blocks = n_size // block_n
            pid_m = pid // num_n_blocks
            pid_n = pid % num_n_blocks

            m_offsets = pid_m * block_m + ti.arange(0, block_m)
            n_offsets = pid_n * block_n + ti.arange(0, block_n)

            a_ptrs = a_ptr + ti.expand_dims(m_offsets * n_size, 1) + ti.arange(0, block_k)
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
    c = hidet.empty([m_size, n_size], device='cuda')

    func(a, b, c)
    print('  tile: {:.2f} ms'.format(hidet.utils.benchmark_func(lambda: func(a, b, c), repeat=20)))

    import torch
    ta, tb, tc = a.torch(), b.torch(), c.torch()
    print(' torch: {:.2f} ms'.format(hidet.utils.benchmark_func(lambda: torch.matmul(ta, tb, out=tc), repeat=20)))

    hidet.utils.assert_close(c, tc)

    report = ncu_run(func, a, b, c)
    report.visualize()


def main():
    # demo_arange()
    #
    # demo_debug_print()
    #
    # demo_vector_add()
    #
    # demo_expand_dims()
    #
    # demo_for_and_increment()
    #
    # demo_reduce()
    #
    # demo_dot_simt()
    #
    # demo_ldst()

    demo_matmul()


if __name__ == '__main__':
    main()
