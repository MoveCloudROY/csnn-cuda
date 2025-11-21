#pragma once
#include <cuda_runtime.h>

#define CONV1_KENREL_SIZE 5
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

#define CONV2_KERNEL_SIZE 5

#define CONV_KERNEL_K 5


template <int N, int Ci, int Hi, int Wi, int Co>
__global__ void conv2d_implicit_gemm(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y        // [N, Co, Ho, Wo]
);

__global__ void conv2d_nchw_kernel_n_only(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, Ho, Wo]
    int N, int Ci, int Hi, int Wi, int Co
);

__global__ void conv2d_c1_k5_kernel(
    const float* __restrict__ x, // [N, 1, 28, 28]
    const float* __restrict__ w, // [Co, 1, 5, 5]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, 24, 24]
    int N, int Co
);

__global__ void conv2d_nchw_fuse_if_kernel_n_only(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, Ho, Wo]
    float* __restrict__ v,
    int N, int Ci, int Hi, int Wi, int Co
);

__global__ void conv2d_c1_k5_fuse_if_kernel(
    const float* __restrict__ x, // [N, 1, 28, 28]
    const float* __restrict__ w, // [Co, 1, 5, 5]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, 24, 24]
    float* __restrict__ v,
    int N, int Co
);

__global__ void ifnode_integrate_fire_kernel(
    const float* __restrict__ x, float* __restrict__ mem, float* __restrict__ spk, float thr,
    int total
);


__global__ void maxpool2x2_s2_nchw_kernel(
    const float* __restrict__ x, float* __restrict__ y, int N, int C, int Hi, int Wi
);


__global__ void fc_forward_kernel(
    const float* __restrict__ x, // [N, In]
    const float* __restrict__ W, // [Out, In]
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int N, int In, int Out
);