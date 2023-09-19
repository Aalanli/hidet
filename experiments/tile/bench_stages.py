# %%
import numpy
import hidet
import hllm
import torch
import triton
import triton.language as tl
import vllm.model_executor.parallel_utils
from hidet.ir.dtypes import float16, float32
from hidet.transforms import lower, PassContext, instruments
from hidet.utils.ncu_utils import ncu_run
from hidet import ops
import numpy as np

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()
# hidet.option.debug_show_var_id()

hidet.utils.clear_cache_dir()

numpy.set_printoptions(precision=2, edgeitems=64, linewidth=256)


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence
from hidet.ir.module import IRModule

import hidet
from software_pipelinev2 import software_pipeline_pass as software_pipeline_pass_v2
from hidet.transforms.base import Pass, FunctionPass, SequencePass, RepeatFunctionPass, PassContext
from hidet.transforms.instruments import PassInstrument, SaveIRInstrument, ProfileInstrument
from hidet.transforms.unify_global_objects import unify_global_objects_pass
from hidet.transforms.flatten_tensor_slice import flatten_tensor_slice_pass
from hidet.transforms.flatten_tensor_index import flatten_tensor_index_pass
from hidet.transforms.generate_launch_func import generate_launch_func_pass
from hidet.transforms.explicit_unroll import explicit_unroll_pass
from hidet.transforms.import_primitive_functions import import_primitive_functions_pass
from hidet.transforms.simplify_stmt import simplify_stmt_pass
from hidet.transforms.expand_let_expr import expand_let_expr_pass
from hidet.transforms.instantiate_symbols import instantiate_symbols_pass
from hidet.transforms.resolve_generic_primitive_function import resolve_primitive_func_pass
from hidet.transforms.inline_function import inline_function_pass
from hidet.transforms.add_explicit_cast import add_explicit_cast_pass
from hidet.transforms.inline_let_stmt import inline_let_stmt_pass
from hidet.transforms.rule_based_simplifier import rule_based_simplify_pass
from hidet.transforms.normalize_const_tensor import normalize_const_tensor_pass
from hidet.transforms.lower_task_mapping import lower_task_mapping_pass
from hidet.transforms.lower_protect_access import lower_protect_access_pass
from hidet.transforms.declare_to_let import declare_to_let_pass
from hidet.transforms.propagate_launch_bound import propagate_launch_bound_pass
from hidet.transforms.check_launch_configuration import check_launch_configuration_pass
from hidet.transforms.lower_special_cast import lower_special_cast_pass
from hidet.transforms.annotate_header_and_libs import annotate_header_and_libs_pass
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa_pass
from hidet.transforms.tile.generic.inject_explicit_transform_ops import inject_explicit_transform_ops_pass
from hidet.transforms.tile.generic.canonicalize_expressions import canonicalize_expressions_pass
from hidet.transforms.tile.generic.fold_constant import fold_constant_pass
from hidet.transforms.tile.generic.pattern_transform import pattern_transform_pass
from hidet.transforms.tile.generic.loop_invariant_code_motion import loop_invariant_code_motion_pass
from hidet.transforms.tile.cuda.resolve_dot import resolve_dot_pass
from hidet.transforms.tile.cuda.instantiate_layout import instantiate_layout_pass
from hidet.transforms.tile.cuda.coalesce_memory_access import coalesce_memory_access_pass
from hidet.transforms.tile.cuda.remove_layout_convert import remove_layout_convert_pass
from hidet.transforms.tile.cuda.software_pipeline import software_pipeline_pass
from hidet.transforms.tile.cuda.split_dot_k import split_dot_k_pass
from hidet.transforms.tile.cuda.plan_shared_memory import plan_shared_memory_pass
from hidet.transforms.tile.cuda.lower_tile_dialect import lower_tile_dialect_pass


def lower_with(ir_module: IRModule, transforms: Sequence[Pass]) -> IRModule:
    ctx = PassContext.current()
    for instrument in ctx.instruments:
        instrument.before_all_passes(ir_module)
    for transform in transforms:
        ir_module = transform(ir_module)
    for instrument in ctx.instruments:
        instrument.after_all_passes(ir_module)

    return ir_module


def lower(ir_module: IRModule, num_stages=3, new=True, disable=False) -> IRModule:

    tile_generic_transforms = [
        inject_explicit_transform_ops_pass(),
        canonicalize_expressions_pass(),
        canonicalize_to_ssa_pass(),
        fold_constant_pass(),
        pattern_transform_pass(),
        loop_invariant_code_motion_pass(),
    ]

    tile_cuda_transforms = [
        resolve_dot_pass(),
        instantiate_layout_pass(),
        coalesce_memory_access_pass(),
        remove_layout_convert_pass(),
        loop_invariant_code_motion_pass(),
        (lambda x: x) if disable else (software_pipeline_pass(num_stages) if not new else software_pipeline_pass_v2(num_stages)),
        split_dot_k_pass(),
        plan_shared_memory_pass(),
        lower_tile_dialect_pass(),
    ]

    ir_module = lower_with(ir_module, tile_generic_transforms + tile_cuda_transforms)
    return ir_module


def demo_matmul(lower_fn, m_size=1024, n_size=1024, k_size=1024, dtype='float32', bench=False):
    from hidet.ir.type import data_type
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs, cast
    from hidet.lang import tile as ti

    num_warps = 8
    block_m = 32
    block_n = 128
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
        
    ir_module = lower_fn(script_module.ir_module())
    func = ir_module.build()
    import torch

    a = hidet.randn([m_size, k_size], dtype=dtype, device='cuda')
    b = hidet.randn([k_size, n_size], dtype=dtype, device='cuda')
    c = hidet.zeros([m_size, n_size], dtype=dtype, device='cuda')

    if bench:
        print('{} x {} x {}'.format(m_size, n_size, k_size))

    func(a, b, c)
    if bench:
        num = hidet.utils.benchmark_func(lambda: func(a, b, c), repeat=20)
        print('  tile: {:.3f} ms'.format(num))
        return num
    
    ta, tb, tc = a.torch(), b.torch(), c.torch().clone()
    torch.matmul(ta, tb, out=tc)
    # if bench:
    #     print(' torch: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: torch.matmul(ta, tb, out=tc), repeat=20)))

    if dtype == float16:
        atol, rtol = 5e-2, 5e-2
    elif dtype == float32:
        atol, rtol = 1e-4, 1e-4
    else:
        assert False
    # hidet.utils.assert_close(c, tc, atol=atol, rtol=rtol)


kdim = {ns: [] for ns in [1, 2, 3, 4]}
kdimn = {ns: [] for ns in [2, 3, 4]}

for k in [512, 1024, 2048, 4096]:
    print('disable-pipeline:')
    ms = demo_matmul(lambda x: lower(x, num_stages=1, new=False, disable=True), bench=True, k_size=k)
    kdim[1].append(ms)
    for num_stages in [2, 3, 4]:
        print(f'k: {k}, num-stages: {num_stages}')
        print('original:')
        ms = demo_matmul(lambda x: lower(x, num_stages=num_stages, new=False), bench=True, k_size=k)
        kdim[num_stages].append(ms)
        print('new:')
        ms = demo_matmul(lambda x: lower(x, num_stages=num_stages, new=True), bench=True, k_size=k)
        kdimn[num_stages].append(ms)
        print('---')

from matplotlib import pyplot as plt

ax = plt.subplot()
for k, v in kdim.items():
    ax.plot([512, 1024, 2048, 4096], v, label=f'orig-n={k}')

for k, v in kdimn.items():
    ax.plot([512, 1024, 2048, 4096], v, label=f'new-n={k}')

ax.legend()
plt.show()

# %%
