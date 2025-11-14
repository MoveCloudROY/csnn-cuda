#pragma once
#include <cuda_runtime.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define MATMUL_WARP_M WMMA_M
#define MATMUL_WARP_N WMMA_N
#define MATMUL_WARP_K WMMA_K

#define NUM_MATMUL_WARPS_M 4
#define NUM_MATMUL_WARPS_N 2
#define NUM_MATMUL_WARPS_K 2

#define MATMUL_BLOCK_M (NUM_MATMUL_WARPS_M * MATMUL_WARP_M)
#define MATMUL_BLOCK_N (NUM_MATMUL_WARPS_N * MATMUL_WARP_N)
#define MATMUL_BLOCK_K (NUM_MATMUL_WARPS_K * MATMUL_WARP_K)


#define MATMUL_THREAD_M (NUM_MATMUL_WARPS_M * 32)
#define MATMUL_THREAD_N (NUM_MATMUL_WARPS_N)

__global__ void linear_forward(
    const float* __restrict__ x, // [N, In]
    const float* __restrict__ w, // [Out, In]
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int N, int Out, int In
);


__global__ void linear_fuse_if(
    const float* __restrict__ x, // [N, In]
    const float* __restrict__ w, // [Out, In]
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    float* __restrict__ v,       // [N, Out]
    int N, int Out, int In
);


__global__ void linear_fuse_argmax10(
    const float* __restrict__ x, // [N, In]
    const float* __restrict__ w, // [Out, In]
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int* __restrict__ pred,       // [N, Out]
    int N, int Out, int In
);
// __global__ __launch_bounds__(1024) void linear_fuse_if(
//     int M, int N, int K, half alpha, half* A, half* B, half beta, half* C
// );