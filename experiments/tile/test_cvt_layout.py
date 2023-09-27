# %%
import hidet
from hidet.lang.types import float16, float32, int32

from hidet.ir.type import data_type
from hidet.lang.types import f32, int32
from hidet.lang import attrs, cast
from hidet.lang import tile as ti

hidet.option.cache_dir('.')
hidet.option.save_lower_ir()


dtype = 'float16'
num_warps = 8
block_m = 32
block_n = 128
block_k = 32
dtype = data_type(dtype)

with hidet.script_module() as script_module:
    @hidet.script
    def matmul(a_ptr: ~dtype, b_ptr: ~dtype, c_ptr: ~dtype):
        attrs.func_kind = 'cuda_tile'
        attrs.cuda.block_dim = 32
        attrs.cuda.grid_dim = 1

        a = ti.arange(0, 32)
        xs = ti.expand_dims(a, 0) * 32 + ti.expand_dims(a, 1)

        a1 = ti.load(a_ptr + xs)
        # b = ti.expand_dims(ti.sum(a1, 0), 0) * a1
        ti.store(b_ptr + a, ti.sum(a1, 0))

func = script_module.build()
