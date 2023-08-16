import hidet
from hidet.transforms import lower, PassContext, instruments

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()

hidet.utils.clear_cache_dir()


def lower_script_module(script_module):
    ir_module = script_module.ir_module()
    with PassContext(
        instruments=[
            instruments.SaveIRInstrument('./outs/ir')
        ]
    ) as ctx:
        ir_module = lower(ir_module)
    print(ir_module)


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
    b = hidet.randn([16], dtype=hidet.float32, device='cuda')

    print(a)
    func(a, b, n)
    print(b)


def vector_add(n: int):
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
    print(a)
    print(b)
    print(c)
    print(c - (a + b))


def main():
    # demo_arange()

    n = 128
    vector_add(n)

    # a = hidet.randn([n], dtype=hidet.float32, device='cuda')
    # b = hidet.randn([n], dtype=hidet.float32, device='cuda')
    # c = hidet.zeros([n], dtype=hidet.float32, device='cuda')
    # kernel = vector_add(n)
    # kernel(a, b, c)
    # print(c)


if __name__ == '__main__':
    main()
