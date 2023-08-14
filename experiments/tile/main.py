import hidet
from hidet.transforms import lower, PassContext, instruments

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()

hidet.utils.clear_cache_dir()


def vector_add(n: int):
    from hidet.lang.types import f32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    block_size = 128

    with hidet.script_module() as script_module:

        @hidet.script
        def vec_add(a: ~f32, b: ~f32, c: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = (n + block_size - 1) // block_size

            pid = ti.program_id()
            offsets = pid * block_size + ti.arange(0, block_size)
            mask = offsets < n

            result = ti.load(a + offsets, mask=mask) + ti.load(b + offsets, mask=mask)
            ti.store(c + offsets, result, mask=mask)

    ir_module = script_module.ir_module()
    with PassContext(
        instruments=[
            instruments.SaveIRInstrument('./outs/ir')
        ]
    ) as ctx:
        ir_module = lower(ir_module)
    print(ir_module)


def main():
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
