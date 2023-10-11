# %%
from typing import List, Tuple
from triton.compiler.compiler import *

def _is_cuda(arch):
    return isinstance(arch, int)

def _get_jsonable_constants(constants):
    def _is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False
    serialized_constants = {}
    for constant in constants:
        if _is_jsonable(constants[constant]):
            serialized_constants[constant] = constants[constant]
    return serialized_constants

def run_passes(mod, passes: List[Tuple[str, List[Any]]]) -> List[Tuple[str, str]]:
    intermediate_ir = []
    for p, args in passes:
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        getattr(pm, p)(*args)
        pm.run(mod)
        intermediate_ir.append((p, str(mod)))
    return mod, intermediate_ir


def optimize_ttir2(mod, arch):
    passes = [
        ("add_inliner_pass", []),
        ("add_triton_combine_pass", []),
        ("add_canonicalizer_pass", []),
        ("add_reorder_broadcast_pass", []),
        ("add_cse_pass", []),
        ("add_licm_pass", []),
        ("add_symbol_dce_pass", []),
    ]
    mod = inline_triton_ir(mod)
    mod = ttir_compute_capability_rewrite(mod, arch)
    return run_passes(mod, passes)


def optimize_ttgir2(mod, num_stages, arch):
    passes = [
        ("add_tritongpu_coalesce_pass", []),
        ("add_tritongpu_remove_layout_conversions_pass", []),
    ]
    if isinstance(arch, int):
        passes.append(("add_tritongpu_accelerate_matmul_pass", [arch]))
    passes.extend([
        ("add_tritongpu_remove_layout_conversions_pass", []),
        ("add_tritongpu_optimize_dot_operands_pass", []),
        ("add_tritongpu_pipeline_pass", [num_stages]),
        ("add_tritongpu_prefetch_pass", []),
        ("add_tritongpu_optimize_dot_operands_pass", []),
        ("add_tritongpu_remove_layout_conversions_pass", []),
        ("add_tritongpu_decompose_conversions_pass", []),
        ("add_tritongpu_reorder_instructions_pass", []),
        ("add_cse_pass", []),
        ("add_symbol_dce_pass", []),
    ])
    return run_passes(mod, passes)


def get_ir(fn, **kwargs):
    # Get device type to decide which backend should be used
    device_type = kwargs.get("device_type", "cuda")
    _device_backend = get_backend(device_type)

    if device_type in ["cuda", "hip"]:
        arch = get_architecture_descriptor(kwargs.get("cc", None))
    else:
        _device_backend = get_backend(device_type)
        assert _device_backend
        arch = _device_backend.get_architecture_descriptor(**kwargs)

    is_cuda = device_type == "cuda" and _is_cuda(arch)
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3 if is_cuda and arch >= 75 else 2)
    extern_libs = kwargs.get("extern_libs", dict())
    if extern_libs is None:
        extern_libs = dict()
    debug = kwargs.get("debug", False)

    # find out the signature of the function
    assert isinstance(fn, JITFunction)
    configs = kwargs.get("configs", None)
    signature = kwargs["signature"]
    if configs is None:
        configs = [instance_descriptor()]
    assert len(configs) == 1
 
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    
    
    ir_modules = []
    mod = ast_to_ttir(fn, signature, configs[0], constants, debug=debug, arch=arch)
    ir_modules.append(("ast", str(mod)))
    mod, new_ir = optimize_ttir2(mod, arch)
    ir_modules.extend(new_ir)
    mod = ttir_to_ttgir(mod, num_warps)
    ir_modules.append(("ttir_to_ttgir", str(mod)))
    mod, new_ir = optimize_ttgir2(mod, num_stages, arch)
    ir_modules.extend(new_ir)

    return ir_modules

