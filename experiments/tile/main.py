import hidet
from hidet.transforms import lower, PassContext, instruments

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()

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

    demo_reduce()


if __name__ == '__main__':
    main()
