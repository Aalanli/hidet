# %%
import triton_utils
import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({}, num_warps=1, num_stages=1)
], key=[])
@triton.jit
def test(a, b, c):
    a_idx = tl.arange(0, 32)
    b_idx = tl.arange(0, 16)

    a_ptr = a + a_idx
    b_ptr = b + b_idx
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    cres = a[:, None] + b[None, :]
    tl.store(c + (a_idx[:, None] * 16 + b_idx[None, :]), cres)


import torch
a = torch.randn(32, device='cuda')
b = torch.randn(16, device='cuda')
c = torch.empty([32, 16], device='cuda')

test[(1,)](a, b, c)
print(torch.allclose(c, a[:, None] + b[None, :]))

# %%
import triton_utils
import triton
import triton.language as tl

@triton.jit
def test(a, b, c, ks):
    a_idx = tl.arange(0, 32)
    b_idx = tl.arange(0, 64)

    a_ptr = a + a_idx[:, None] * 64 + b_idx[None, :]
    b_ptr = b + b_idx[:, None] * 32 + a_idx[None, :]
    cs = tl.zeros((32, 32), dtype=tl.float32)
    for k in range(ks):
        a_ = tl.load(a_ptr)
        b_ = tl.load(b_ptr)
        cs += tl.dot(a_, b_)
        a_ptr += 32
        b_ptr += 32
    cs = cs.to(c.dtype.element_ty)
    tl.store(c + (a_idx[:, None] * 32 + a_idx[None, :]), cs)

from triton.compiler import *
import triton._C.libtriton.triton as _triton

kwargs = {'signature': '*fp32, *fp32, *fp32, i32'}
fn = test

capability = kwargs.get("cc", None)
if capability is None:
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    capability = capability[0] * 10 + capability[1]
# we get the kernel, i.e. the first function generated in the module
# if fn is not a JITFunction, then it
# has to be a path to a file
context = _triton.ir.context()
asm = dict()
constants = kwargs.get("constants", dict())
num_warps = kwargs.get("num_warps", 4)
num_stages = 1
extern_libs = kwargs.get("extern_libs", dict())
# build compilation stages
def ttir_to_ttgir(mod, num_warps, num_stages, compute_capability):
    pm = _triton.ir.pass_manager(mod.context)
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.enable_debug()
    pm.add_coalesce_pass()
    # The combine pass converts blocked layout to mma layout
    # for dot ops so that pipeline can get shared memory swizzled correctly.
    pm.add_tritongpu_combine_pass(compute_capability)
    pm.add_tritongpu_pipeline_pass(num_stages)
    pm.run(mod)
    return mod

stages = {
    "ast": (lambda path: fn, None),
    "ttir": (lambda path: parse_mlir_module(path, context),
                lambda src: ast_to_ttir(src, signature, configs[0], constants)),
    "ttgir": (lambda path: parse_mlir_module(path, context),
                lambda src: ttir_to_ttgir(src, num_warps, num_stages, capability)),
}
# find out the signature of the function
if isinstance(fn, triton.runtime.JITFunction):
    configs = kwargs.get("configs", None)
    signature = kwargs["signature"]
    if configs is None:
        configs = [instance_descriptor()]
    assert len(configs) == 1
    kwargs["configs"] = configs
    name = fn.__name__
    first_stage = 0
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    kwargs["signature"] = signature
else:
    assert isinstance(fn, str)
    _, ir = os.path.basename(fn).split(".")
    src = Path(fn).read_text()
    import re
    match = re.search(prototype_pattern[ir], src, re.MULTILINE)
    name, signature = match.group(1), match.group(2)
    # print(name, signature)
    types = re.findall(arg_type_pattern[ir], signature)
    # print(types)
    param_tys = [convert_type_repr(ty) for ty in types]
    signature = {k: v for k, v in enumerate(param_tys)}
    first_stage = list(stages.keys()).index(ir)

# cache manager
so_path = make_stub(name, signature, constants)
# create cache manager
fn_cache_manager = CacheManager(make_hash(fn, **kwargs))
# determine name and extension type of provided function
if isinstance(fn, triton.runtime.JITFunction):
    name, ext = fn.__name__, "ast"
else:
    name, ext = os.path.basename(fn).split(".")

# load metadata if any
metadata = None
if fn_cache_manager.has_file(f'{name}.json'):
    with open(fn_cache_manager._make_path(f"{name}.json")) as f:
        metadata = json.load(f)
else:
    metadata = {"num_warps": num_warps, "num_stages": num_stages, "ctime": dict()}
    if ext == "ptx":
        assert "shared" in kwargs, "ptx compilation must provide shared memory size"
        metadata["shared"] = kwargs["shared"]

first_stage = list(stages.keys()).index(ext)
asm = dict()
module = fn
# run compilation pipeline  and populate metadata
for ir, (parse, compile) in list(stages.items())[first_stage:]:
    path = fn_cache_manager._make_path(f"{name}.{ir}")
    if ir == ext:
        next_module = parse(fn)
    elif os.path.exists(path) and\
            ir in metadata["ctime"] and\
            os.path.getctime(path) == metadata["ctime"][ir]:
        next_module = parse(path)
    else:
        next_module = compile(module)
        fn_cache_manager.put(next_module, f"{name}.{ir}")
    if os.path.exists(path):
        metadata["ctime"][ir] = os.path.getctime(path)
    asm[ir] = next_module if ir == "cubin" else str(next_module)
    if ir == "llir" and "shared" not in metadata:
        metadata["shared"] = _triton.get_shared_memory_size(module)
    if ir == "ptx":
        metadata["name"] = ptx_get_kernel_name(next_module)
    module = next_module
# write-back metadata
fn_cache_manager.put(json.dumps(metadata), f"{name}.json", binary=False)
# return handle to compiled kernel

# %%
print(asm['ttgir'])
