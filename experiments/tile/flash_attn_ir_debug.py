# %%
import hidet
from hidet.ir.dtypes import float16, float32
import torch

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()

hidet.utils.clear_cache_dir()

def demo_flash_attn(B, H, M, N, D, dtype, BLOCK_M, BLOCK_N, BLOCK_D, num_warps=4):
    from hidet.ir.type import data_type
    from hidet.lang.types import f32, f16
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    dtype = data_type(dtype)
    q_part = ti.cdiv(M, BLOCK_M)
    num_blocks = q_part * B * H
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and D == BLOCK_D

    with hidet.script_module() as script_module:
        @hidet.script
        def flash_attn(q_ptr: ~dtype, k_ptr: ~dtype, v_ptr: ~dtype, y_ptr: ~dtype):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = num_warps * 32
            attrs.cuda.grid_dim = num_blocks

            pid = ti.program_id()
            bh_id = pid // q_part
            pid_m = pid % q_part
            offset_q = bh_id * M * D + pid_m * BLOCK_M * D
            offset_kv = bh_id * N * D

            midx = ti.arange(0, BLOCK_M)
            nidx = ti.arange(0, BLOCK_N)
            didx = ti.arange(0, BLOCK_D)

            q_ptrs = q_ptr + ti.expand_dims(midx * D, 1) + ti.expand_dims(didx, 0) + offset_q
            k_ptrs = k_ptr + ti.expand_dims(nidx * D, 0) + ti.expand_dims(didx, 1) + offset_kv
            v_ptrs = v_ptr + ti.expand_dims(nidx * D, 1) + ti.expand_dims(didx, 0) + offset_kv
            
            q = ti.load(q_ptrs)
            maxes = ti.zeros([BLOCK_M], dtype=f32) - f32.max_value
            sums  = ti.zeros([BLOCK_M], dtype=f32)
            acc = ti.zeros([BLOCK_M, BLOCK_D], dtype=f32)

            for ki in range((N + BLOCK_N - 1) // BLOCK_N):
                k = ti.load(k_ptrs)
                qk = ti.dot(q, k)
                qk1 = ti.cast(qk, f32)
                new_max = ti.maximum(ti.max(qk1, 1), maxes)
                alpha = ti.exp(maxes - new_max)
                acc *= ti.expand_dims(alpha, 1)
                p = ti.exp(qk1 - ti.expand_dims(new_max, 1))
                sums = sums * alpha + ti.sum(p, 1)
                maxes = new_max
                p1 = ti.cast(p, dtype)
                v = ti.load(v_ptrs)
                acc += ti.dot(p1, v)
                k_ptrs += BLOCK_N * D
                v_ptrs += BLOCK_N * D
            
            acc1 = acc / ti.expand_dims(sums, 1)
            acc2 = ti.cast(acc1, dtype)
            midx = ti.arange(0, BLOCK_M)
            didx = ti.arange(0, BLOCK_D)
            y_ptrs = y_ptr + ti.expand_dims(midx * D, 1) + ti.expand_dims(didx, 0) + offset_q
            ti.store(y_ptrs, acc2)
    
    
    return script_module.ir_module()


hidet.option.debug_show_var_id(True)
B = 2
H = 4
M = 32
N = 32
D = 64
ir_module = demo_flash_attn(B, H, M, N, D, float16, BLOCK_M=32, BLOCK_N=32, BLOCK_D=64)


from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa_pass
from hidet.transforms.tile.generic.inject_explicit_transform_ops import inject_explicit_transform_ops_pass
from hidet.transforms.tile.generic.canonicalize_expressions import canonicalize_expressions_pass
from hidet.transforms.tile.generic.fold_constant import fold_constant_pass
from hidet.transforms.tile.generic.pattern_transform import pattern_transform_pass
from hidet.transforms.tile.generic.loop_invariant_code_motion import loop_invariant_code_motion_pass
from hidet.transforms.tile.cuda.resolve_dot import resolve_dot_pass
from hidet.transforms.tile.cuda.instantiate_layout import instantiate_layout_pass
from hidet.transforms.tile.cuda.coalesce_memory_access import coalesce_memory_access_pass
from hidet.transforms.tile.cuda.remove_layout_convert import *
from hidet.transforms.tile.cuda.software_pipeline import software_pipeline_pass
from hidet.transforms.tile.cuda.split_dot_k import split_dot_k_pass
from hidet.transforms.tile.cuda.plan_shared_memory import plan_shared_memory_pass
from hidet.transforms.tile.cuda.lower_tile_dialect import lower_tile_dialect_pass
from hidet.transforms.tile.analyzers.usage_analyzer import SourceAnalyzer, UsageAnalyzer
from hidet.transforms.tile.cuda.remove_layout_convertv2 import PropagateLayoutAnalysis
def transforms(module):
    passes = [
        inject_explicit_transform_ops_pass(),
        canonicalize_expressions_pass(),
        canonicalize_to_ssa_pass(),
        fold_constant_pass(),
        pattern_transform_pass(),
        loop_invariant_code_motion_pass(),

        resolve_dot_pass(),
        instantiate_layout_pass(),
        coalesce_memory_access_pass(),

        loop_invariant_code_motion_pass(),
        canonicalize_to_ssa_pass(),
        # remove_layout_convert_pass(),
        # loop_invariant_code_motion_pass(),
        # software_pipeline_pass(),
        # split_dot_k_pass(),
        # plan_shared_memory_pass(),
        # lower_tile_dialect_pass(),
    ]
    for t in passes:
        module = t(module)
    return module

from hidet.transforms.tile.analyzers import DependencyAnalyzer
ir_module = transforms(ir_module)
print(ir_module)


# %%


def dependency_analysis(ir_module) -> Dict[Var, List[Var]]:
    dep = DependencyAnalyzer()
    dep.visit(ir_module)
    return dep.depends

def contains_layout(v: Var) -> bool:
    return isinstance(v, Var) and isinstance(v.type, TileType) and v.type.layout is not None

def filter_dependencies(depends: Dict[Var, List[Var]]) -> Dict[Var, List[Var]]:
    new_depends = {}
    for k, v in depends.items():
        if contains_layout(k):
            new_depends[k] = [x for x in v if contains_layout(x)]
    return new_depends

def sanity_check_layout_analysis(ir_module):
    layouts = PropagateLayoutAnalysis()
    layouts.visit(ir_module)
    layouts = layouts.layouts
    dep = DependencyAnalyzer()
    dep.visit(ir_module)
    dependence: Dict[Var, List[Var]] = dep.depends
    dep_set = {k for k in dependence.keys() if contains_layout(k)}
    layout_set = {k for k in layouts.keys()}
    if dep_set != layout_set:
        print('dep_set - layout_set', dep_set - layout_set)
        print('layout_set - dep_set', layout_set - dep_set)
        assert False

depends = dependency_analysis(ir_module)
# depends = {k.id: [x.id for x in v] for k, v, in depends.items()}
# depends = filter_dependencies(depends)



# %%
layouts = PropagateLayoutAnalysis()
layouts.visit(ir_module)
sanity_check_layout_analysis(ir_module)

from visualize_var_dependence import ir_to_graph_html, dependence_graph_to_graph_html
# ir_to_graph_html(ir_module, 'flash-attn.html', notebook=True)
dependence_graph_to_graph_html(depends, 'flash-attn-dep.html', notebook=True)

# %%
from typing import Tuple
# todo: better cost function
def cvt_layout_cost(a: TileLayout, b: TileLayout) -> int:
    # assert isinstance(a, TileLayout) and isinstance(b, TileLayout)
    if hash(a) == hash(b):
        return 0
    else:
        return 1

class BruteForceSolver:
    def __init__(self, cost_fn, depends: Dict[Var, List[Var]], layout: Dict[Var, Set[TileLayout]]):
        self.cost_fn = cost_fn
        self.depends = depends
        inv_depends: Dict[Var, Set[Var]] = {}
        for k, v in depends.items():
            for x in v:
                if x not in inv_depends:
                    inv_depends[x] = set()
                inv_depends[x].add(k)
        self.inv_depends: Dict[Var, List[Var]] = {k: list(v) for k, v in inv_depends.items()}
    
        self.layout = layout
        self.assignment: Dict[Var, TileLayout] = {}
        for v, ls in layout.items():
            if len(ls) == 1:
                self.assignment[v] = next(iter(ls))
        self.worklist: List[Var] = []
        seen = set()
        work_list = list(layout.keys())
        while len(work_list) > 0:
            v = work_list.pop()
            if v in seen:
                continue
            self.worklist.append(v)
            seen.add(v)
            for ks in depends[v]:
                if ks not in self.assignment and k not in seen:
                    self.worklist.append(ks)
        
        self.best_seen: Dict[Var, Tuple[int, TileLayout]] = {}
                
    def solve_helper(self, v: Var, layout: TileLayout, cost_so_far: int):
        parent = self.inv_depends[v]
        parent_layout = self.assignment[parent]




# %%
print(len(layouts.layouts), max(map(lambda x: len(x), layouts.layouts.values())))

# %%
layouts.layouts

ids_to_layouts = {k.id: list(v) for k, v in layouts.layouts.items()}
def print_layout(lst):
    for i in lst:
        print(i)


# %%


# %%
usage = UsageAnalyzer()
usage.visit(ir_module)
usage.usages
# %%
source = SourceAnalyzer()
source.visit(ir_module)
print(source.source)
# %%
source.source

# %%
ir_func = ir_module.functions['flash_attn']

cvt_layouts = [
    ChangeForArgLayoutRewriter(),
    IdentityConvertLayoutTransform(),
    ConvertConstructLayoutTransform(),
    FoldConvertLayoutTransform(),
    PushConvertLayoutForUnaryOpTransform(),
    PushConvertLayoutForBinaryOpTransform(),
    FoldConvertLayoutBeforeAndAfterCast(),
    DeadCodeEliminationRewriter()
]

def cvt_layout(func, t):
    func_before = func
    func = t(func)
    if func_before is not func:
        func = canonicalize_to_ssa(func)
    return func

layout_transforms = [[] for _ in range(len(cvt_layouts))]

func = ir_func
for i in range(10):
    for j, t in enumerate(cvt_layouts):
        func = cvt_layout(func, t)
        layout_transforms[j].append(func)


# %%
from hidet.utils.py import diff_lines, fuzzy_diff_text
def concat_horizontal(a: str, b: str) -> str:
    a = a.split('\n')
    b = b.split('\n')
    ls = max(map(lambda x: len(x), a))
    print(len(a), len(b), type(a), type(b))
    combined = []
    for i in range(max(len(a), len(b))):
        s = ''
        if i < len(a):
            s += a[i]
            pad = ls - len(a[i])
            s += ' ' * pad
        else:
            s += ' ' * ls
        
        if i < len(b):
            s += b[i]
        
        combined.append(s)
    return '\n'.join(combined)

print(concat_horizontal(str(layout_transforms[0][4]), str(layout_transforms[-1][3])))
