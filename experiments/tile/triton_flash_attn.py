# %%
import triton
import triton.language as tl

from collections import namedtuple
from save_triton_ir import get_ir

from triton.ops.flash_attention import _fwd_kernel
@triton.jit
def flash_attn(q_ptr, k_ptr, v_ptr, y_ptr,
               M, N, D, 
               BLOCK_M: tl.constexpr, 
               BLOCK_N: tl.constexpr, 
               BLOCK_D: tl.constexpr):
    
    q_part = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id()
    bh_id = pid // q_part
    pid_m = pid % q_part
    offset_q = bh_id * M * D + pid_m * BLOCK_M * D
    offset_kv = bh_id * N * D

    midx = tl.arange(0, BLOCK_M)
    nidx = tl.arange(0, BLOCK_N)
    didx = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + tl.expand_dims(midx * D, 1) + tl.expand_dims(didx, 0) + offset_q
    k_ptrs = k_ptr + tl.expand_dims(nidx * D, 0) + tl.expand_dims(didx, 1) + offset_kv
    v_ptrs = v_ptr + tl.expand_dims(nidx * D, 1) + tl.expand_dims(didx, 0) + offset_kv
    
    q = tl.load(q_ptrs)
    maxes = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    sums  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for ki in range((N + BLOCK_N - 1) // BLOCK_N):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        new_max = tl.maximum(tl.max(qk, 1), maxes)
        alpha = tl.exp(maxes - new_max)
        acc *= tl.expand_dims(alpha, 1)
        p = tl.exp(qk - tl.expand_dims(new_max, 1))
        sums = sums * alpha + tl.sum(p, 1)
        maxes = new_max
        p1 = p.to(tl.float16)
        v = tl.load(v_ptrs)
        acc += tl.dot(p1, v)
        k_ptrs += BLOCK_N * D
        v_ptrs += BLOCK_N * D
    
    acc1 = acc / tl.expand_dims(sums, 1)
    acc2 = acc1.to(tl.float16)
    midx = tl.arange(0, BLOCK_M)
    didx = tl.arange(0, BLOCK_D)
    y_ptrs = y_ptr + tl.expand_dims(midx * D, 1) + tl.expand_dims(didx, 0) + offset_q
    tl.store(y_ptrs, acc2)


@triton.jit
def _fwd_kernel(
    Q, K, V,
    Out,
    B, H, S, D,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * S * D
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(S, BLOCK_DMODEL),
        strides=(D, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, S),
        strides=(1, D),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(S, BLOCK_DMODEL),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(K.dtype.element_ty)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else S
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(S, BLOCK_DMODEL),
        strides=(D, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


import torch
a = torch.randn([1, 1, 512, 64], dtype=torch.float16, device='cuda')
b = torch.randn([1, 1, 512, 64], dtype=torch.float16, device='cuda')
ker = _fwd_kernel[(1,)](a, a, a, b, 1, 1, 512, 128, BLOCK_M=32, BLOCK_N=64, BLOCK_DMODEL=64, IS_CAUSAL=False)
# n_warps = 4
# Spec = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])
# specialization = Spec(divisible_by_16={0, 1, 2, 3, 4, 5, 6, 7}, equal_to_1=set())
# irs = get_ir(_fwd_kernel, num_warps=n_warps, signature='*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32', # configs=[specialization],
#              BLOCK_M=32, BLOCK_N=64, BLOCK_DMODEL=32, IS_CASUAL=False)

# %%
print(ker.asm['ttgir'])

"""
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_fwd_kernel_0d1d2d3d4c5c6d7d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i64 = arith.constant 128 : i64
    %cst = arith.constant dense<64> : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<64> : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %cst_3 = arith.constant dense<0xFF800000> : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %cst_4 = arith.constant dense<1.44269502> : tensor<32x64xf32, #blocked>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg4 : i32
    %3 = arith.muli %2, %arg5 : i32
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, i32
    %5 = arith.muli %0, %c32_i32 : i32
    %6 = arith.extsi %arg5 : i32 to i64
    %7 = arith.extsi %5 : i32 to i64
    %8 = tt.addptr %arg1, %3 : !tt.ptr<f16>, i32
    %9 = tt.addptr %arg2, %3 : !tt.ptr<f16>, i32
    %10 = tt.splat %7 : (i64) -> tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.extsi %11 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.addi %10, %12 : tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : (tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi64, #blocked>
    %15 = tt.splat %6 : (i64) -> tensor<32x1xi64, #blocked>
    %16 = arith.muli %14, %15 : tensor<32x1xi64, #blocked>
    %17 = tt.splat %4 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %18 = tt.addptr %17, %16 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi64, #blocked>
    %19 = tt.broadcast %18 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %24 = arith.extsi %20 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %25 = arith.extsi %21 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %26 = arith.extsi %22 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %27 = arith.extsi %23 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %28 = tt.expand_dims %24 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x64xi64, #blocked>
    %29 = tt.broadcast %28 : (tensor<1x64xi64, #blocked>) -> tensor<32x64xi64, #blocked>
    %30 = tt.addptr %19, %29 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi64, #blocked>
    %31 = tt.load %30 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked>
    %32 = arith.extf %31 : tensor<32x64xf16, #blocked> to tensor<32x64xf32, #blocked>
    %33 = arith.mulf %32, %cst_4 : tensor<32x64xf32, #blocked>
    %34 = arith.truncf %33 : tensor<32x64xf32, #blocked> to tensor<32x64xf16, #blocked>
    %35 = triton_gpu.convert_layout %34 : (tensor<32x64xf16, #blocked>) -> tensor<32x64xf16, #shared>
    %36 = tt.expand_dims %25 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi64, #blocked1>
    %37 = tt.splat %8 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked1>
    %38 = tt.addptr %37, %36 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi64, #blocked1>
    %39 = tt.broadcast %38 : (tensor<64x1x!tt.ptr<f16>, #blocked1>) -> tensor<64x64x!tt.ptr<f16>, #blocked1>
    %40 = tt.splat %6 : (i64) -> tensor<1x64xi64, #blocked1>
    %41 = tt.splat %6 : (i64) -> tensor<64x1xi64, #blocked>
    %42 = tt.splat %9 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %43 = tt.broadcast %28 : (tensor<1x64xi64, #blocked>) -> tensor<64x64xi64, #blocked>
    %44 = arith.cmpi sgt, %arg4, %c0_i32 : i32
    %45 = tt.expand_dims %26 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
    %46 = arith.muli %45, %40 : tensor<1x64xi64, #blocked1>
    %47 = tt.broadcast %46 : (tensor<1x64xi64, #blocked1>) -> tensor<64x64xi64, #blocked1>
    %48 = tt.addptr %39, %47 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi64, #blocked1>
    %49 = triton_gpu.alloc_tensor : tensor<3x64x64xf16, #shared1>
    %50 = tt.splat %44 : (i1) -> tensor<64x64xi1, #blocked1>
    %51 = triton_gpu.insert_slice_async %48, %49, %c0_i32, %50 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f16>, #blocked1> -> tensor<3x64x64xf16, #shared1>
    triton_gpu.async_commit_group
    %52 = tt.expand_dims %27 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi64, #blocked>
    %53 = arith.muli %52, %41 : tensor<64x1xi64, #blocked>
    %54 = tt.addptr %42, %53 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi64, #blocked>
    %55 = tt.broadcast %54 : (tensor<64x1x!tt.ptr<f16>, #blocked>) -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %56 = tt.addptr %55, %43 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi64, #blocked>
    %57 = triton_gpu.alloc_tensor : tensor<3x64x64xf16, #shared>
    %58 = tt.splat %44 : (i1) -> tensor<64x64xi1, #blocked>
    %59 = triton_gpu.insert_slice_async %56, %57, %c0_i32, %58 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f16>, #blocked> -> tensor<3x64x64xf16, #shared>
    triton_gpu.async_commit_group
    %60 = arith.cmpi sgt, %arg4, %c64_i32 : i32
    %61 = arith.addi %26, %cst_0 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %62 = tt.expand_dims %61 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
    %63 = arith.muli %62, %40 : tensor<1x64xi64, #blocked1>
    %64 = tt.broadcast %63 : (tensor<1x64xi64, #blocked1>) -> tensor<64x64xi64, #blocked1>
    %65 = tt.addptr %39, %64 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi64, #blocked1>
    %66 = tt.splat %60 : (i1) -> tensor<64x64xi1, #blocked1>
    %67 = triton_gpu.insert_slice_async %65, %51, %c1_i32, %66 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f16>, #blocked1> -> tensor<3x64x64xf16, #shared1>
    triton_gpu.async_commit_group
    %68 = arith.addi %27, %cst : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %69 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi64, #blocked>
    %70 = arith.muli %69, %41 : tensor<64x1xi64, #blocked>
    %71 = tt.addptr %42, %70 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi64, #blocked>
    %72 = tt.broadcast %71 : (tensor<64x1x!tt.ptr<f16>, #blocked>) -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %73 = tt.addptr %72, %43 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi64, #blocked>
    %74 = tt.splat %60 : (i1) -> tensor<64x64xi1, #blocked>
    %75 = triton_gpu.insert_slice_async %73, %59, %c1_i32, %74 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f16>, #blocked> -> tensor<3x64x64xf16, #shared>
    triton_gpu.async_commit_group
    triton_gpu.async_wait {num = 2 : i32}
    %76 = triton_gpu.extract_slice %67[0, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<3x64x64xf16, #shared1> to tensor<64x64xf16, #shared1>
    %77 = triton_gpu.extract_slice %75[0, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<3x64x64xf16, #shared> to tensor<64x64xf16, #shared>
    %78:14 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst_2, %arg9 = %cst_3, %arg10 = %c0_i64, %arg11 = %c0_i64, %arg12 = %67, %arg13 = %75, %arg14 = %76, %arg15 = %77, %arg16 = %c128_i64, %arg17 = %c128_i64, %arg18 = %c64_i32, %arg19 = %c2_i32, %arg20 = %c1_i32) -> (tensor<32x64xf32, #mma>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, i64, i64, tensor<3x64x64xf16, #shared1>, tensor<3x64x64xf16, #shared>, tensor<64x64xf16, #shared1>, tensor<64x64xf16, #shared>, i64, i64, i32, i32, i32)  : i32 {
      %89 = triton_gpu.convert_layout %35 : (tensor<32x64xf16, #shared>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %90 = triton_gpu.convert_layout %arg14 : (tensor<64x64xf16, #shared1>) -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %91 = tt.dot %89, %90, %cst_1 {allowTF32 = true} : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      %92 = "tt.reduce"(%91) <{axis = 1 : i32}> ({
      ^bb0(%arg21: f32, %arg22: f32):
        %142 = tt.pure_extern_elementwise %arg21, %arg22 {libname = "libdevice", libpath = "/home/allan/.conda/envs/hidet/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_fmaxf"} : (f32, f32) -> f32
        tt.reduce.return %142 : f32
      }) : (tensor<32x64xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %93 = "triton_gpu.cmpf"(%arg9, %92) <{predicate = 2 : i64}> : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %94 = "triton_gpu.select"(%93, %arg9, %92) : (tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %95 = arith.subf %arg9, %94 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %96 = tt.pure_extern_elementwise %95 {libname = "libdevice", libpath = "/home/allan/.conda/envs/hidet/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_exp2f"} : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %97 = tt.expand_dims %94 {axis = 1 : i32} : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<32x1xf32, #mma>
      %98 = tt.broadcast %97 : (tensor<32x1xf32, #mma>) -> tensor<32x64xf32, #mma>
      %99 = arith.subf %91, %98 : tensor<32x64xf32, #mma>
      %100 = tt.pure_extern_elementwise %99 {libname = "libdevice", libpath = "/home/allan/.conda/envs/hidet/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_exp2f"} : (tensor<32x64xf32, #mma>) -> tensor<32x64xf32, #mma>
      %101 = arith.mulf %arg8, %cst_2 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %102 = arith.addf %101, %96 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %103 = tt.expand_dims %102 {axis = 1 : i32} : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<32x1xf32, #mma>
      %104 = tt.broadcast %103 : (tensor<32x1xf32, #mma>) -> tensor<32x64xf32, #mma>
      %105 = arith.mulf %arg7, %104 : tensor<32x64xf32, #mma>
      %106 = arith.truncf %100 : tensor<32x64xf32, #mma> to tensor<32x64xf16, #mma>
      %107 = triton_gpu.convert_layout %106 : (tensor<32x64xf16, #mma>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %108 = triton_gpu.convert_layout %arg15 : (tensor<64x64xf16, #shared>) -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %109 = tt.dot %107, %108, %105 {allowTF32 = true} : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      %110 = arith.mulf %arg8, %96 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %111 = "tt.reduce"(%100) <{axis = 1 : i32}> ({
      ^bb0(%arg21: f32, %arg22: f32):
        %142 = arith.addf %arg21, %arg22 : f32
        tt.reduce.return %142 : f32
      }) : (tensor<32x64xf32, #mma>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %112 = arith.addf %110, %111 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %113 = arith.addi %arg10, %c64_i64 : i64
      %114 = arith.addi %arg11, %c64_i64 : i64
      %115 = arith.addi %arg18, %c64_i32 : i32
      %116 = arith.cmpi slt, %115, %arg4 : i32
      %117 = arith.remsi %arg19, %c3_i32 : i32
      %118 = arith.remsi %arg20, %c3_i32 : i32
      %119 = tt.splat %arg16 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %120 = arith.addi %119, %26 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %121 = tt.expand_dims %120 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
      %122 = arith.muli %121, %40 : tensor<1x64xi64, #blocked1>
      %123 = tt.broadcast %122 : (tensor<1x64xi64, #blocked1>) -> tensor<64x64xi64, #blocked1>
      %124 = tt.addptr %39, %123 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi64, #blocked1>
      %125 = tt.splat %arg17 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %126 = arith.addi %125, %27 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %127 = tt.expand_dims %126 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi64, #blocked>
      %128 = arith.muli %127, %41 : tensor<64x1xi64, #blocked>
      %129 = tt.addptr %42, %128 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi64, #blocked>
      %130 = tt.broadcast %129 : (tensor<64x1x!tt.ptr<f16>, #blocked>) -> tensor<64x64x!tt.ptr<f16>, #blocked>
      %131 = tt.addptr %130, %43 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi64, #blocked>
      %132 = arith.addi %arg16, %c64_i64 : i64
      %133 = arith.addi %arg17, %c64_i64 : i64
      %134 = tt.splat %116 : (i1) -> tensor<64x64xi1, #blocked1>
      %135 = triton_gpu.insert_slice_async %124, %arg12, %117, %134 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f16>, #blocked1> -> tensor<3x64x64xf16, #shared1>
      triton_gpu.async_commit_group
      %136 = tt.splat %116 : (i1) -> tensor<64x64xi1, #blocked>
      %137 = triton_gpu.insert_slice_async %131, %arg13, %117, %136 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f16>, #blocked> -> tensor<3x64x64xf16, #shared>
      triton_gpu.async_commit_group
      triton_gpu.async_wait {num = 2 : i32}
      %138 = triton_gpu.extract_slice %135[%118, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<3x64x64xf16, #shared1> to tensor<64x64xf16, #shared1>
      %139 = triton_gpu.extract_slice %137[%118, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<3x64x64xf16, #shared> to tensor<64x64xf16, #shared>
      %140 = arith.addi %arg19, %c1_i32 : i32
      %141 = arith.addi %arg20, %c1_i32 : i32
      scf.yield %109, %112, %94, %113, %114, %135, %137, %138, %139, %132, %133, %115, %140, %141 : tensor<32x64xf32, #mma>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, i64, i64, tensor<3x64x64xf16, #shared1>, tensor<3x64x64xf16, #shared>, tensor<64x64xf16, #shared1>, tensor<64x64xf16, #shared>, i64, i64, i32, i32, i32
    }
    triton_gpu.async_wait {num = 0 : i32}
    %79 = tt.expand_dims %78#1 {axis = 1 : i32} : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<32x1xf32, #mma>
    %80 = tt.broadcast %79 : (tensor<32x1xf32, #mma>) -> tensor<32x64xf32, #mma>
    %81 = arith.divf %78#0, %80 : tensor<32x64xf32, #mma>
    %82 = tt.addptr %arg3, %3 : !tt.ptr<f16>, i32
    %83 = arith.truncf %81 : tensor<32x64xf32, #mma> to tensor<32x64xf16, #mma>
    %84 = tt.splat %82 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %85 = tt.addptr %84, %16 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi64, #blocked>
    %86 = tt.broadcast %85 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %87 = tt.addptr %86, %29 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi64, #blocked>
    %88 = triton_gpu.convert_layout %83 : (tensor<32x64xf16, #mma>) -> tensor<32x64xf16, #blocked>
    tt.store %87, %88 {cache = 1 : i32, evict = 1 : i32} : tensor<32x64xf16, #blocked>
    tt.return
  }
}
"""