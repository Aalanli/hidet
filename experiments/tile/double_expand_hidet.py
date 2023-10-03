# %%
import hidet
from hidet.lang.types import f32
from hidet.lang import attrs
from hidet.lang import tile as ti

WARPS = 4
hidet.option.cache_dir(f'hidet_tile_double_expand_w{WARPS}')
hidet.option.save_lower_ir(True)
hidet.utils.clear_cache_dir()

with hidet.script_module() as script_module:
    @hidet.script
    def over_sub_load(a: ~f32, b: ~f32, c: ~f32):
        attrs.func_kind = 'cuda_tile'
        attrs.cuda.block_dim = 32 * WARPS
        attrs.cuda.grid_dim = 1

        a_idx1 = ti.arange(0, 32)
        a_idx2 = ti.arange(0, 64)
        b_idx1 = ti.arange(0, 8)
        b_idx2 = ti.arange(0, 64)
        d_idx = ti.arange(0, 2)
        
        a_idx = ti.expand_dims(ti.expand_dims(d_idx, 1), 2) + (ti.expand_dims(ti.expand_dims(a_idx2, 1), 0) * 32 + ti.expand_dims(ti.expand_dims(a_idx1, 0), 0)) * 32 * 64
        b_idx = ti.expand_dims(ti.expand_dims(d_idx, 1), 2) + (ti.expand_dims(ti.expand_dims(b_idx2, 1), 0) * 8  + ti.expand_dims(ti.expand_dims(b_idx1, 0), 0)) * 8 * 64

        a_ptr = a + a_idx
        b_ptr = b + b_idx
        a1 = ti.load(a_ptr)
        b1 = ti.load(b_ptr)
        c1 = b1 * ti.expand_dims(ti.expand_dims(ti.sum(ti.sum(a1, 2), 1), 1), 2)

        ti.store(c + b_idx, c1)

script_module.build()

# %%
