# %%
import hidet
from hidet.lang.types import f32
from hidet.lang import attrs
from hidet.lang import tile as ti

WARPS = 4
hidet.option.cache_dir(f'hidet_tile_oversub_load_w{WARPS}')
hidet.option.save_lower_ir(True)
hidet.utils.clear_cache_dir()

with hidet.script_module() as script_module:
    @hidet.script
    def over_sub_load(a: ~f32, b: ~f32, c: ~f32):
        attrs.func_kind = 'cuda_tile'
        attrs.cuda.block_dim = 32 * WARPS
        attrs.cuda.grid_dim = 1

        a_idx = ti.arange(0, 32)
        b_idx = ti.arange(0, 64)

        a_ptr = a + a_idx
        b_ptr = b + b_idx
        a1 = ti.load(a_ptr)
        b1 = ti.load(b_ptr)
        cres = ti.expand_dims(b1, 1) + ti.expand_dims(a1, 0)
        ti.store(c + (ti.expand_dims(b_idx, 1) * 32 + ti.expand_dims(a_idx, 0)), cres)

script_module.build()

# %%
