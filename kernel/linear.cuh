#pragma once
#include "common.cuh"
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

// warp size = 4 * 2 , each warp compute a tile of size 64 * 32
#define MATMUL4_WARP_M WMMA_M
#define MATMUL4_WARP_N WMMA_N
#define MATMUL4_WARP_K WMMA_K

#define NUM_MATMUL4_WARPS_M 4
#define NUM_MATMUL4_WARPS_N 2
#define NUM_MATMUL4_WARPS_K 2

#define MATMUL4_BLOCK_M (NUM_MATMUL4_WARPS_M * MATMUL4_WARP_M)
#define MATMUL4_BLOCK_N (NUM_MATMUL4_WARPS_N * MATMUL4_WARP_N)
#define MATMUL4_BLOCK_K (NUM_MATMUL4_WARPS_K * MATMUL4_WARP_K)

#define MATMUL4_BLOCK_SKEW_M 16
#define MATMUL4_BLOCK_SKEW_N 16
#define MATMUL4_BLOCK_SKEW_K 16

#define MATMUL4_THREAD_M (NUM_MATMUL4_WARPS_M * 32)
#define MATMUL4_THREAD_N (NUM_MATMUL4_WARPS_N)

static_assert(MATMUL4_THREAD_M <= 1024, "Too many threads per block");
static_assert(NUM_MATMUL4_WARPS_M * MATMUL4_WARP_M == MATMUL4_BLOCK_M, "Inconsistent matmul3 block M");
static_assert(NUM_MATMUL4_WARPS_N * MATMUL4_WARP_N == MATMUL4_BLOCK_N, "Inconsistent matmul3 block N");
static_assert(NUM_MATMUL4_WARPS_K * MATMUL4_WARP_K == MATMUL4_BLOCK_K, "Inconsistent matmul3 block K");

static_assert(MATMUL4_WARP_M % WMMA_M == 0, "Inconsistent warp M");
static_assert(MATMUL4_WARP_N % WMMA_N == 0, "Inconsistent warp N");
static_assert(MATMUL4_WARP_K % WMMA_K == 0, "Inconsistent warp K");

__global__ void linear_forward(
    float* __restrict__ x, // [N, In]
    float* __restrict__ w, // [Out, In]
    float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int N, int Out, int In
);


__global__ void linear_fuse_if(
    float* __restrict__ x, // [N, In]
    float* __restrict__ w, // [Out, In]
    float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    float* __restrict__ v,       // [N, Out]
    int N, int Out, int In
);

// __global__ void linear_01x_fuse_if(
//     float* __restrict__ x, // [N, In]
//     float* __restrict__ w, // [Out, In]
//     float* __restrict__ b, // [Out]
//     float* __restrict__ y,       // [N, Out]
//     float* __restrict__ v,       // [N, Out]
//     int N, int Out, int In
// );


__global__ void linear_fuse_argmax10(
    float* __restrict__ x, // [N, In]
    float* __restrict__ w, // [Out, In]
    float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int* __restrict__ pred,       // [N, Out]
    int N, int Out, int In
);
// __global__ __launch_bounds__(1024) void linear_fuse_if(
//     int M, int N, int K, half alpha, half* A, half* B, half beta, half* C
// );