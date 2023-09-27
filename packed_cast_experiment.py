# %%
import math
import hidet
from hidet.ir.dtypes import int8, f16
from hidet.ir.stmt import asm
from hidet.lang import attrs, view, u32, u64, tensor_pointer, register_tensor, cast
from hidet.lang.cuda import blockIdx, threadIdx

hidet.option.cache_dir('./outs')
hidet.option.debug_cache_tuning(True)

def cast_i8x4_f16x4(x: hidet.Tensor):
    shape = math.prod(x.shape)
    assert shape % 4 == 0
    n_iter = shape // 4
    nblocks = n_iter // 512
    with hidet.script_module() as module:
        @hidet.script
        def cvt_i8x4_f16x4(x: int8[4], y: f16[4]):
            xi = view(x, u32[1])
            yi = view(y, u32[2])

            asm("prmt.b32 %0,%1,%1,%2;", inputs=[xi[0], 0x9180], outputs=[yi[0]])
            asm("prmt.b32 %0,%1,%1,%2;", inputs=[xi[0], 0xB3A2], outputs=[yi[1]])
            
            asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[0], 0x03FF03FF, 0x66006600, 106], outputs=[yi[0]])
            asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[1], 0x03FF03FF, 0x66006600, 106], outputs=[yi[1]])

            rk = 0x66006600
            asm("sub.f16x2 %0, %1, %2;", inputs=[yi[0], rk], outputs=[yi[0]])
            asm("sub.f16x2 %0, %1, %2;", inputs=[yi[1], rk], outputs=[yi[1]])

        @hidet.script
        def cast_kernel(x: ~int8, y: ~f16):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = nblocks
            attrs.cuda.block_dim = 512
            tid = threadIdx.x
            pid = blockIdx.x

            xu = cast(x, ~u32)
            yu = cast(y, ~u64)

            i = pid * 512 + tid
            while i < n_iter:
                yi = register_tensor(f16, [4])
                xi = register_tensor(u32, [1])
                xi[0] = xu[i]
                cvt_i8x4_f16x4(view(xi, int8[4]), yi)
                y_ = view(yi, u64[1])
                yu[i] = y_[0]
                i += 512 * nblocks
    
    y = hidet.ones([shape], dtype=f16, device='cuda')
    func = module.build()
    func(x, y)
    return y

x = hidet.randint(-128, 128, [512], dtype='int32').to('int8', 'cuda')
y2 = cast_i8x4_f16x4(x)


# %%