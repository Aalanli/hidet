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

static __device__ __forceinline__ void hidet_cuda_st_global_b32(void * __restrict__ addr_0, void * __restrict__ v0_0) {
  asm volatile ("st.global.b32 [%0], %1;" :  : "l"(addr_0), "r"(*((uint32_t*)(v0_0))));
}

static __global__ void __launch_bounds__(32) hidet_over_sub_load(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  float* cvt[4];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt[i] = (a + ((((int)threadIdx.x % 8) * 4) + i));
  } 
  float load[4];
  hidet_cuda_ld_global_nc_L2_128B_v4_b32(cvt[0], (&load[0]), (&load[1]), (&load[2]), (&load[3]));
  float* cvt_0[4];
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    cvt_0[i] = (b + ((((int)threadIdx.x % 16) * 4) + i));
  } 
  float load_0[4];
  hidet_cuda_ld_global_nc_L2_128B_v4_b32(cvt_0[0], (&load_0[0]), (&load_0[1]), (&load_0[2]), (&load_0[3]));
  float cvt_1[2];
  float *cvt_smem = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    if ((int)threadIdx.x < 16) {
      cvt_smem[((((int)threadIdx.x % 16) * 4) + i)] = load_0[i];
    } 
  } 
  __syncthreads();
  cvt_1[0] = cvt_smem[(int)threadIdx.x];
  cvt_1[1] = cvt_smem[(32 + (int)threadIdx.x)];
  __syncthreads();
  float expand_dims[2];
  expand_dims[0] = cvt_1[0];
  expand_dims[1] = cvt_1[1];
  float cvt_2[64];
  float *cvt_smem_0 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  cvt_smem_0[((int)threadIdx.x * 2)] = expand_dims[0];
  cvt_smem_0[(64 + ((int)threadIdx.x * 2))] = expand_dims[1];
  __syncthreads();
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cvt_2[i] = cvt_smem_0[(((int)threadIdx.x * 2) + i)];
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cvt_2[(32 + i)] = cvt_smem_0[((((int)threadIdx.x * 2) + i) + 64)];
  } 
  __syncthreads();
  float broadcast[64];
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    broadcast[i] = cvt_2[0];
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    broadcast[(32 + i)] = cvt_2[32];
  } 
  float cvt_3[1];
  float *cvt_smem_1 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  for (int32_t i = 0; (i < 4); i = (i + 1)) {
    if ((int)threadIdx.x < 8) {
      cvt_smem_1[((((int)threadIdx.x % 8) * 4) + i)] = load[i];
    } 
  } 
  __syncthreads();
  cvt_3[0] = cvt_smem_1[(int)threadIdx.x];
  __syncthreads();
  float expand_dims_0[1];
  expand_dims_0[0] = cvt_3[0];
  float cvt_4[64];
  float *cvt_smem_2 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  cvt_smem_2[(int)threadIdx.x] = expand_dims_0[0];
  __syncthreads();
  for (int32_t i = 0; (i < 64); i = (i + 1)) {
    cvt_4[i] = cvt_smem_2[((i * 33) + (int)threadIdx.x)];
  } 
  __syncthreads();
  float broadcast_0[64];
  for (int32_t i = 0; (i < 64); i = (i + 1)) {
    broadcast_0[i] = cvt_4[0];
  } 
  float cvt_5[64];
  float *cvt_smem_3 = ((float*)(hidet_cuda_dynamic_shared_memory_void(0)));
  __syncthreads();
  for (int32_t i = 0; (i < 64); i = (i + 1)) {
    cvt_smem_3[((i * 33) + (int)threadIdx.x)] = broadcast_0[i];
  } 
  __syncthreads();
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cvt_5[i] = cvt_smem_3[(((int)threadIdx.x * 33) + i)];
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cvt_5[(32 + i)] = cvt_smem_3[((((int)threadIdx.x * 33) + i) + 1056)];
  } 
  __syncthreads();
  float cres[64];
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cres[i] = (broadcast[i] + cvt_5[i]);
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cres[(32 + i)] = (broadcast[(32 + i)] + cvt_5[(32 + i)]);
  } 
  float* cvt_6[64];
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cvt_6[i] = (c + (((int)threadIdx.x * 32) + i));
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    cvt_6[(32 + i)] = (((((int)threadIdx.x * 32) + i) + c) + 1024);
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    hidet_cuda_st_global_b32(cvt_6[i], (&cres[i]));
  } 
  for (int32_t i = 0; (i < 32); i = (i + 1)) {
    hidet_cuda_st_global_b32(cvt_6[(32 + i)], (&cres[(32 + i)]));
  } 
}

DLL void hidet_launch(float * __restrict__ a_0, float * __restrict__ b_0, float * __restrict__ c_0) {
  hidet_over_sub_load<<<dim3(1, 1, 1), dim3(32, 1, 1), 8580, (cudaStream_t)get_cuda_stream()>>>(a_0, b_0, c_0);
  {cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) LOG(ERROR) << "CUDA error: " << cudaGetErrorString(err) << "\n";}
}
