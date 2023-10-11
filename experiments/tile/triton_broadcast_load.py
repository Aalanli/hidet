# %%
import triton
import triton.language as tl

from collections import namedtuple
from save_triton_ir import get_ir


@triton.jit
def test(a, b):
    idx = tl.arange(0, 64)[:, None] * 32 + tl.arange(0, 32)[None, :]

    a_ptr = a + idx
    b_ptr = b + idx
    a = tl.load(a_ptr)
    tl.store(b_ptr, a)

n_warps = 1
Spec = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])
specialization = Spec(divisible_by_16={0, 1}, equal_to_1=set())
irs = get_ir(test, num_warps=n_warps, signature='*fp16, *fp16', configs=[specialization])

# %%
@triton.jit
def test2(a, b, c):
    idx1 = tl.arange(0, 64)
    idx2 = tl.arange(0, 32)
    idx_a = idx1[:, None] * 32 + idx2[None, :]
    idx_b = idx2[:, None] * 64 + idx1[None, :]

    a_ptr = a + idx_a
    b_ptr = b + idx_b
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    c_ptr = c + (tl.arange(0, 64)[:, None] * 32 + tl.arange(0, 32))
    c = a + tl.trans(b)
    tl.store(c_ptr, c)

n_warps = 1
Spec = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])
specialization = Spec(divisible_by_16={0, 1}, equal_to_1=set())
irs = get_ir(test2, num_warps=n_warps, signature='*fp32, *fp32, *fp32', configs=[specialization])

"""
---ttir_to_ttgir---
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @test2_0d1d2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>

    %2 = triton_gpu.convert_layout %0 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %4 = arith.muli %3, %cst_0 : tensor<64x1xi32, #blocked>

    %5 = triton_gpu.convert_layout %1 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %7 = tt.broadcast %4 : (tensor<64x1xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %8 = tt.broadcast %6 : (tensor<1x32xi32, #blocked2>) -> tensor<64x32xi32, #blocked2>
    %9 = triton_gpu.convert_layout %8 : (tensor<64x32xi32, #blocked2>) -> tensor<64x32xi32, #blocked>
    
    # idx_a
    %10 = arith.addi %7, %9 : tensor<64x32xi32, #blocked>

    %11 = triton_gpu.convert_layout %1 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %13 = arith.muli %12, %cst : tensor<32x1xi32, #blocked>
    %14 = triton_gpu.convert_layout %0 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2>
    %16 = tt.broadcast %13 : (tensor<32x1xi32, #blocked>) -> tensor<32x64xi32, #blocked>
    %17 = tt.broadcast %15 : (tensor<1x64xi32, #blocked2>) -> tensor<32x64xi32, #blocked2>
    %18 = triton_gpu.convert_layout %17 : (tensor<32x64xi32, #blocked2>) -> tensor<32x64xi32, #blocked>
    
    # idx_b
    %19 = arith.addi %16, %18 : tensor<32x64xi32, #blocked>


    %20 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x32x!tt.ptr<f32>, #blocked>
    # a_ptr
    %21 = tt.addptr %20, %10 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked>


    %22 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x64x!tt.ptr<f32>, #blocked>
    # b_ptr
    %23 = tt.addptr %22, %19 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>

    %24 = tt.load %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf32, #blocked>
    %25 = tt.load %23 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf32, #blocked>
    %26 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x32x!tt.ptr<f32>, #blocked>
    
    # cse takes care of the second rematerialization
    %27 = tt.addptr %26, %10 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked>
    %28 = triton_gpu.convert_layout %25 : (tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #shared>
    %29 = tt.trans %28 : (tensor<32x64xf32, #shared>) -> tensor<64x32xf32, #shared1>
    %30 = triton_gpu.convert_layout %29 : (tensor<64x32xf32, #shared1>) -> tensor<64x32xf32, #blocked>
    %31 = arith.addf %24, %30 : tensor<64x32xf32, #blocked>
    tt.store %27, %31 {cache = 1 : i32, evict = 1 : i32} : tensor<64x32xf32, #blocked>
    tt.return
  }
}
"""


"""
---coalesce-layouts---
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>

#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @test2_0d1d2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>

    %2 = triton_gpu.convert_layout %0 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %4 = arith.muli %3, %cst_0 : tensor<64x1xi32, #blocked>
    %5 = triton_gpu.convert_layout %1 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %7 = tt.broadcast %4 : (tensor<64x1xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %8 = tt.broadcast %6 : (tensor<1x32xi32, #blocked2>) -> tensor<64x32xi32, #blocked2>
    %9 = triton_gpu.convert_layout %8 : (tensor<64x32xi32, #blocked2>) -> tensor<64x32xi32, #blocked>
    
    # idx_a
    %10 = arith.addi %7, %9 : tensor<64x32xi32, #blocked>

    %11 = triton_gpu.convert_layout %1 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %13 = arith.muli %12, %cst : tensor<32x1xi32, #blocked>
    %14 = triton_gpu.convert_layout %0 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2>
    %16 = tt.broadcast %13 : (tensor<32x1xi32, #blocked>) -> tensor<32x64xi32, #blocked>
    %17 = tt.broadcast %15 : (tensor<1x64xi32, #blocked2>) -> tensor<32x64xi32, #blocked2>
    %18 = triton_gpu.convert_layout %17 : (tensor<32x64xi32, #blocked2>) -> tensor<32x64xi32, #blocked>
    
    # idx_b
    %19 = arith.addi %16, %18 : tensor<32x64xi32, #blocked>

    %20 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x32x!tt.ptr<f32>, #blocked>
    # ptr_a
    %21 = tt.addptr %20, %10 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked>

    %22 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x64x!tt.ptr<f32>, #blocked>
    # ptr_b
    %23 = tt.addptr %22, %19 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>

    # introduced by coalesce pass
    %24 = triton_gpu.convert_layout %21 : (tensor<64x32x!tt.ptr<f32>, #blocked>) -> tensor<64x32x!tt.ptr<f32>, #blocked3>
    %25 = tt.load %24 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf32, #blocked3>
    %26 = triton_gpu.convert_layout %25 : (tensor<64x32xf32, #blocked3>) -> tensor<64x32xf32, #blocked>

    # introduced by coalesce pass    
    %27 = triton_gpu.convert_layout %23 : (tensor<32x64x!tt.ptr<f32>, #blocked>) -> tensor<32x64x!tt.ptr<f32>, #blocked4>
    %28 = tt.load %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf32, #blocked4>
    %29 = triton_gpu.convert_layout %28 : (tensor<32x64xf32, #blocked4>) -> tensor<32x64xf32, #blocked>

    %30 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %31 = tt.addptr %30, %10 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked>
    %32 = triton_gpu.convert_layout %29 : (tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #shared>
    %33 = tt.trans %32 : (tensor<32x64xf32, #shared>) -> tensor<64x32xf32, #shared1>
    %34 = triton_gpu.convert_layout %33 : (tensor<64x32xf32, #shared1>) -> tensor<64x32xf32, #blocked>


    %35 = arith.addf %26, %34 : tensor<64x32xf32, #blocked>
    %36 = triton_gpu.convert_layout %31 : (tensor<64x32x!tt.ptr<f32>, #blocked>) -> tensor<64x32x!tt.ptr<f32>, #blocked3>
    %37 = triton_gpu.convert_layout %35 : (tensor<64x32xf32, #blocked>) -> tensor<64x32xf32, #blocked3>
    tt.store %36, %37 {cache = 1 : i32, evict = 1 : i32} : tensor<64x32xf32, #blocked3>
    tt.return
  }
}
"""

"""
---remove-layouts-pass---
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @test2_0d1d2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %cst_0 = arith.constant dense<64> : tensor<32x1xi32, #blocked1>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %4 = arith.muli %2, %cst : tensor<64x1xi32, #blocked>
    %5 = arith.muli %3, %cst : tensor<64x1xi32, #blocked>
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %9 = tt.expand_dims %7 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %10 = tt.broadcast %4 : (tensor<64x1xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %11 = tt.broadcast %5 : (tensor<64x1xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %12 = tt.broadcast %8 : (tensor<1x32xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %13 = tt.broadcast %9 : (tensor<1x32xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %14 = arith.addi %10, %12 : tensor<64x32xi32, #blocked>
    %15 = arith.addi %11, %13 : tensor<64x32xi32, #blocked>
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.expand_dims %16 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %18 = arith.muli %17, %cst_0 : tensor<32x1xi32, #blocked1>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi32, #blocked1>
    %21 = tt.broadcast %18 : (tensor<32x1xi32, #blocked1>) -> tensor<32x64xi32, #blocked1>
    %22 = tt.broadcast %20 : (tensor<1x64xi32, #blocked1>) -> tensor<32x64xi32, #blocked1>
    %23 = arith.addi %21, %22 : tensor<32x64xi32, #blocked1>
    %24 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %25 = tt.addptr %24, %14 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked>
    %26 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %27 = tt.addptr %26, %23 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
    %28 = tt.load %25 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf32, #blocked>
    %29 = tt.load %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf32, #blocked1>
    %30 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %31 = tt.addptr %30, %15 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi32, #blocked>
    %32 = triton_gpu.convert_layout %29 : (tensor<32x64xf32, #blocked1>) -> tensor<32x64xf32, #shared>
    %33 = tt.trans %32 : (tensor<32x64xf32, #shared>) -> tensor<64x32xf32, #shared1>
    %34 = triton_gpu.convert_layout %33 : (tensor<64x32xf32, #shared1>) -> tensor<64x32xf32, #blocked>
    %35 = arith.addf %28, %34 : tensor<64x32xf32, #blocked>
    tt.store %31, %35 {cache = 1 : i32, evict = 1 : i32} : tensor<64x32xf32, #blocked>
    tt.return
  }
}

"""