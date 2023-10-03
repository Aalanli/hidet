#include <stdint.h>
#include <hidet/runtime/symbols.h>
#include <hidet/runtime/memory_planner.h>
#include <hidet/runtime/cpu/context.h>
#include <hidet/runtime/cuda/complex.h>
#include <hidet/runtime/cuda/context.h>
#include <hidet/runtime/logging.h>


static __device__ __forceinline__ void hidet_cuda_ld_global_nc_L2_128B_v4_b32(void * __restrict__ addr, void * __restrict__ v0, void * __restrict__ v1, void * __restrict__ v2, void * __restrict__ v3) {
  asm volatile ("ld.global.nc.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(*((uint32_t*)(v0))), "=r"(*((uint32_t*)(v1))), "=r"(*((uint32_t*)(v2))), "=r"(*((uint32_t*)(v3))) : "l"(addr));
}

static __device__ __forceinline__ void* hidet_cuda_dynamic_shared_memory_void(int32_t byte_offset) {
  extern __shared__ uint8_t dynamic_smem[];
  return ((void*)((&dynamic_smem[byte_offset])));
}

static __device__ __forceinline__ void hidet_cuda_st_global_v4_b32(void * __restrict__ addr_0, void * __restrict__ v0_0, void * __restrict__ v1_0, void * __restrict__ v2_0, void * __restrict__ v3_0) {
  asm volatile ("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :  : "l"(addr_0), "r"(*((uint32_t*)(v0_0))), "r"(*((uint32_t*)(v1_0))), "r"(*((uint32_t*)(v2_0))), "r"(*((uint32_t*)(v3_0))));
}

static __global__ void __launch_bounds__(128) hidet_over_sub_load(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  float* cvt[16];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    for (int32_t i_0 = 0; (i_0 < 4); i_0 = (i_0 + 1)) {
      cvt[((i * 4) + i_0)] = (a + ((((((i * 4) + ((int)threadIdx.x / 32)) * 4) + (((int)threadIdx.x % 32) / 8)) * 32) + ((((int)threadIdx.x % 8) * 4) + i_0)));
    } 
  } 
  float load[16];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    hidet_cuda_ld_global_nc_L2_128B_v4_b32(cvt[(i * 4)], (&load[(i * 4)]), (&load[((i * 4) + 1)]), (&load[((i * 4) + 2)]), (&load[((i * 4) + 3)]));
  } 
  float a1[16];
  float *cvt_smem = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    for (int32_t i_0 = 0; (i_0 < 4); i_0 = (i_0 + 1)) {
      cvt_smem[((((((i * 4) + ((int)threadIdx.x / 32)) * 4) + (((int)threadIdx.x % 32) / 8)) * 33) + ((((int)threadIdx.x % 8) * 4) + i_0))] = load[((i * 4) + i_0)];
    } 
  } 
  __syncthreads();
  for (int32_t i = 0; (i < 16); i = (i + 1)) {
    a1[i] = cvt_smem[((((i * 4) + ((int)threadIdx.x / 32)) * 33) + ((int)threadIdx.x % 32))];
  } 
  __syncthreads();
  float* cvt_0[8];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_0[i] = (b + ((((((int)threadIdx.x / 32) * 8) + (((int)threadIdx.x % 32) / 4)) * 16) + ((((int)threadIdx.x % 4) * 4) + i)));
  } 
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_0[(4 + i)] = ((((((((int)threadIdx.x / 32) * 8) + (((int)threadIdx.x % 32) / 4)) * 16) + ((((int)threadIdx.x % 4) * 4) + i)) + b) + 512);
  } 
  float load_0[8];
  hidet_cuda_ld_global_nc_L2_128B_v4_b32(cvt_0[0], (&load_0[0]), (&load_0[1]), (&load_0[2]), (&load_0[3]));
  hidet_cuda_ld_global_nc_L2_128B_v4_b32(cvt_0[4], (&load_0[4]), (&load_0[5]), (&load_0[6]), (&load_0[7]));
  float reduce_op[16];
  for (int32_t i = 0; (i < 16); i = (i + 1)) {
    reduce_op[i] = 0.0f;
    reduce_op[i] = (reduce_op[i] + a1[i]);
  } 
  for (int32_t i = 0; (i < 16); i = (i + 1)) {
    for (int32_t i_0 = 0; (i_0 < 5); i_0 = (i_0 + 1)) {
      reduce_op[i] = (reduce_op[i] + __shfl_down_sync(4294967295, reduce_op[i], (1 << i_0), 32));
    } 
  } 
  for (int32_t i = 0; (i < 16); i = (i + 1)) {
    for (int32_t i_0 = 0; (i_0 < 5); i_0 = (i_0 + 1)) {
      reduce_op[i] = __shfl_up_sync(4294967295, reduce_op[i], (1 << ((5 - i_0) - 1)), 32);
    } 
    reduce_op[i] = reduce_op[i];
  } 
  float cvt_1[1];
  float *cvt_smem_0 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  for (int32_t i = 0; (i < 16); i = (i + 1)) {
    if (((int)threadIdx.x % 32) < 1) {
      cvt_smem_0[((i * 4) + ((int)threadIdx.x / 32))] = reduce_op[i];
    } 
  } 
  __syncthreads();
  cvt_1[0] = cvt_smem_0[((((int)threadIdx.x / 64) * 32) + ((int)threadIdx.x % 32))];
  __syncthreads();
  float expand_dims[1];
  expand_dims[0] = cvt_1[0];
  float cvt_2[8];
  float *cvt_smem_1 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  if ((((int)threadIdx.x / 32) % 2) < 1) {
    cvt_smem_1[(((((int)threadIdx.x / 64) * 32) + ((int)threadIdx.x % 32)) * 2)] = expand_dims[0];
  } 
  __syncthreads();
  for (int32_t i = 0; (i < 8); i = (i + 1)) {
    cvt_2[i] = cvt_smem_1[((((((int)threadIdx.x / 64) * 32) + ((int)threadIdx.x % 32)) * 2) + ((i * 2) + (((int)threadIdx.x / 32) % 2)))];
  } 
  __syncthreads();
  float broadcast[8];
  for (int32_t i = 0; (i < 8); i = (i + 1)) {
    broadcast[i] = cvt_2[0];
  } 
  float* cvt_3[8];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_3[i] = (c + ((((((int)threadIdx.x / 32) * 8) + (((int)threadIdx.x % 32) / 4)) * 16) + ((((int)threadIdx.x % 4) * 4) + i)));
  } 
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_3[(4 + i)] = ((((((((int)threadIdx.x / 32) * 8) + (((int)threadIdx.x % 32) / 4)) * 16) + ((((int)threadIdx.x % 4) * 4) + i)) + c) + 512);
  } 
  float cvt_4[8];
  float *cvt_smem_2 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  for (int32_t i = 0; (i < 8); i = (i + 1)) {
    cvt_smem_2[((((((int)threadIdx.x / 64) * 32) + ((int)threadIdx.x % 32)) * 17) + ((i * 2) + (((int)threadIdx.x / 32) % 2)))] = broadcast[i];
  } 
  __syncthreads();
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_4[i] = cvt_smem_2[((((((int)threadIdx.x / 32) * 8) + (((int)threadIdx.x % 32) / 4)) * 17) + ((((int)threadIdx.x % 4) * 4) + i))];
  } 
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_4[(4 + i)] = cvt_smem_2[(((((((int)threadIdx.x / 32) * 8) + (((int)threadIdx.x % 32) / 4)) * 17) + ((((int)threadIdx.x % 4) * 4) + i)) + 544)];
  } 
  __syncthreads();
  float cvt_5[8];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_5[i] = (load_0[i] * cvt_4[i]);
  } 
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_5[(4 + i)] = (load_0[(4 + i)] * cvt_4[(4 + i)]);
  } 
  hidet_cuda_st_global_v4_b32(cvt_3[0], (&cvt_5[0]), (&cvt_5[1]), (&cvt_5[2]), (&cvt_5[3]));
  hidet_cuda_st_global_v4_b32(cvt_3[4], (&cvt_5[4]), (&cvt_5[5]), (&cvt_5[6]), (&cvt_5[7]));
}

DLL void hidet_launch(float * __restrict__ a_0, float * __restrict__ b_0, float * __restrict__ c_0) {
  hidet_over_sub_load<<<dim3(1, 1, 1), dim3(128, 1, 1), 8580, (cudaStream_t)get_cuda_stream()>>>(a_0, b_0, c_0);
  {cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err) << "\n";}
}
