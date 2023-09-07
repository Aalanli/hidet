import hidet


def test_dot(m=16, n=16, k=16):
    from hidet.lang.types import f32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def dot(a_ptr: ~f32, b_ptr: ~f32, c_ptr: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a_ptrs = a_ptr + ti.expand_dims(ti.arange(0, m) * k, axis=1) + ti.arange(0, k)
            b_ptrs = b_ptr + ti.expand_dims(ti.arange(0, k) * n, axis=1) + ti.arange(0, n)
            c_ptrs = c_ptr + ti.expand_dims(ti.arange(0, m) * n, axis=1) + ti.arange(0, n)

            a = ti.load(a_ptrs)
            b = ti.load(b_ptrs)
            c = ti.dot(a, b)
            ti.store(c_ptrs, c)

    a = hidet.randn([m, k], dtype='float32', device='cuda')
    b = hidet.randn([k, n], dtype='float32', device='cuda')
    c1 = hidet.empty([m, n], dtype='float32', device='cuda')
    func = script_module.build()
    func(a, b, c1)

    c2 = a @ b

    hidet.utils.assert_close(c1, c2)

