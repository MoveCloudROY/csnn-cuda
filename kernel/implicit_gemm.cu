#include "kernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

using namespace nvcuda;


// conv general (NCHW), K=5 templated by compile-time constant
template <int N, int Ci, int Hi, int Wi, int Co>
__global__ void conv2d_implicit_gemm(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y        // [N, Co, Ho, Wo]
) {
    // const int n  = blockIdx.z;
    // const int tx = threadIdx.x;
    // const int ty = threadIdx.y;
    // const int ow = blockIdx.x * blockDim.x + tx;
    // const int oh = blockIdx.y * blockDim.y + ty;

    constexpr int Ho = Hi - CONV1_KENREL_SIZE + 1;
    constexpr int Wo = Wi - CONV1_KENREL_SIZE + 1;
    constexpr int MM = Co;
    constexpr int KK = Ci * CONV1_KENREL_SIZE * CONV1_KENREL_SIZE;
    constexpr int NN = N * Ho * Wo;
    // }



    // A: row major, B: col major, C: row major
    // __global__ void mysgemm_v2(int M, int N, int K, half alpha, half* A, half* B, half beta, half* C) {
    const int BM      = MATMUL_BLOCK_M;
    const int BN      = MATMUL_BLOCK_N;
    const int BK      = MATMUL_BLOCK_K;
    int       block_m = blockIdx.x * BM;
    int       block_n = blockIdx.y * BN;

    // prepare fragments
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
            a_frag[NUM_MATMUL_WARPS_M];
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
            b_frag[NUM_MATMUL_WARPS_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        c_frag[NUM_MATMUL_WARPS_M * NUM_MATMUL_WARPS_N];
    // 缓存A_tile和B_tile
    __shared__ half As[MATMUL_BLOCK_M * MATMUL_BLOCK_K];
    __shared__ half Bs[MATMUL_BLOCK_N * MATMUL_BLOCK_K];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL_WARPS_M * NUM_MATMUL_WARPS_N; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        // load A tile
        for (int i = threadIdx.y; i < BM; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int a_row = block_m + i;
                int a_col = block_k + j;
                if (a_row < M && a_col < K) {
                    As[i * BK + j] = A[a_row * K + a_col];
                }
            }
        }
        __syncthreads();
        // load B tile
        for (int i = threadIdx.y; i < BN; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int b_row = block_n + i;
                int b_col = block_k + j;
                if (b_row < N && b_col < K) {
                    Bs[i * BK + j] = B[b_row * K + b_col];
                }
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL_WARP_K) {
            // use wmma to compute C tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load A fragment
            int a_row = warp_m * MATMUL_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load B fragment
            int b_row = warp_n * MATMUL_WARP_N;
            int b_col = warp_k;
            wmma::load_matrix_sync(b_frag[warp_n], &Bs[b_row * BK + b_col], BK);
            // wmma mma
            wmma::mma_sync(c_frag[warp_id], a_frag[warp_m], b_frag[warp_n], c_frag[warp_id]);
            // store C fragment
        }
        __syncthreads();
    }
    // store C fragments to C

    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m  = warp_id / NUM_MATMUL_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL_WARP_M;
    int c_col   = block_n + warp_n * MATMUL_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            (half*)&C[c_row * N + c_col],
            c_frag[warp_id],
            N,
            wmma::mem_row_major
        );
    }
}



template __global__ void conv2d_implicit_gemm<1000, 1, 28, 28, 8>(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y        // [N, Co, Ho, Wo]
);


template __global__ void conv2d_implicit_gemm<1000, 8, 28, 28, 16>(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y        // [N, Co, Ho, Wo]
);