#include "linear.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

using namespace nvcuda;

// x: row major, w: col major, y: row major
// __global__ void
// linear_fuse_if(int M, int N, int K, half alpha, half* x, half* w, half beta, half* y) {

__global__ void linear_forward(
    const float* x, // [N, In]
    const float* w, // [Out, In]
    const float* b, // [Out]
    float*       y, // [N, Out]
    int M, int N, int K
) {
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
    __shared__ half Cs[MATMUL_BLOCK_M * MATMUL_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL_WARPS_M * NUM_MATMUL_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        // load x tile
        for (int i = threadIdx.y; i < BM; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int a_row = block_m + i;
                int a_col = block_k + j;
                if (a_row < M && a_col < K) {
                    As[i * BK + j] = __float2half(x[a_row * K + a_col]);
                }
            }
        }
        __syncthreads();
        // load w tile
        for (int i = threadIdx.y; i < BN; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int b_row = block_n + i;
                int b_col = block_k + j;
                if (b_row < N && b_col < K) {
                    Bs[i * BK + j] = __float2half(w[b_row * K + b_col]);
                }
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL_WARP_N;
            int b_col = warp_k;
            wmma::load_matrix_sync(b_frag[warp_n], &Bs[b_row * BK + b_col], BK);
            // wmma mma
            wmma::mma_sync(c_frag[warp_id], a_frag[warp_m], b_frag[warp_n], c_frag[warp_id]);
            // store y fragment
        }
        __syncthreads();
    }
    // store y fragments to y

    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m  = warp_id / NUM_MATMUL_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL_WARP_M;
    int c_col   = block_n + warp_n * MATMUL_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL_WARP_M * BN + warp_n * MATMUL_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }
#pragma unroll
    for (int i = threadIdx.y; i < BM; i += blockDim.y) {
        for (int j = threadIdx.x; j < BN; j += blockDim.x) {
            int c_row = block_m + i;
            int c_col = block_n + j;
            if (c_row < M && c_col < N) {
                Cs[i * BN + j]       = __float2half(__half2float(Cs[i * BN + j]) + b[c_col]);
                y[c_row * N + c_col] = __half2float(Cs[i * BN + j]);
            }
        }
    }
}


__global__ void linear_fuse_if(
    const float* x, // [N, In]
    const float* w, // [Out, In]
    const float* b, // [Out]
    float*       y, // [N, Out]
    float*       v, // [N, Out]
    int M, int N, int K
) {
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
    __shared__ half Cs[MATMUL_BLOCK_M * MATMUL_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL_WARPS_M * NUM_MATMUL_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        // load x tile
        for (int i = threadIdx.y; i < BM; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int a_row = block_m + i;
                int a_col = block_k + j;
                if (a_row < M && a_col < K) {
                    As[i * BK + j] = __float2half(x[a_row * K + a_col]);
                }
            }
        }
        __syncthreads();
        // load w tile
        for (int i = threadIdx.y; i < BN; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int b_row = block_n + i;
                int b_col = block_k + j;
                if (b_row < N && b_col < K) {
                    Bs[i * BK + j] = __float2half(w[b_row * K + b_col]);
                }
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL_WARP_N;
            int b_col = warp_k;
            wmma::load_matrix_sync(b_frag[warp_n], &Bs[b_row * BK + b_col], BK);
            // wmma mma
            wmma::mma_sync(c_frag[warp_id], a_frag[warp_m], b_frag[warp_n], c_frag[warp_id]);
            // store y fragment
        }
        __syncthreads();
    }
    // store y fragments to y

    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m  = warp_id / NUM_MATMUL_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL_WARP_M;
    int c_col   = block_n + warp_n * MATMUL_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL_WARP_M * BN + warp_n * MATMUL_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }
#pragma unroll
    for (int i = threadIdx.y; i < BM; i += blockDim.y) {
        for (int j = threadIdx.x; j < BN; j += blockDim.x) {
            int c_row = block_m + i;
            int c_col = block_n + j;
            if (c_row < M && c_col < N) {
                // Cs[i * BN + j]       = Cs[i * BN + j] + __float2half(b[c_col]);
                float vv = v[c_row * N + c_col] + __half2float(Cs[i * BN + j]) + b[c_col];
                int   f  = vv >= 1.0f;
                y[c_row * N + c_col] = f;
                v[c_row * N + c_col] = f ? 0 : vv;
            }
        }
    }
}


__global__ void linear_fuse_argmax10(
    const float* __restrict__ x, // [N, In]
    const float* __restrict__ w, // [Out, In]
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int* __restrict__ pred,       // [N, Out]
    int M, int N, int K         // M: 1000  N: 10  K: 96
){
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
    __shared__ half Cs[MATMUL_BLOCK_M * MATMUL_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL_WARPS_M * NUM_MATMUL_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        // load x tile
        for (int i = threadIdx.y; i < BM; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int a_row = block_m + i;
                int a_col = block_k + j;
                if (a_row < M && a_col < K) {
                    As[i * BK + j] = __float2half(x[a_row * K + a_col]);
                }
            }
        }
        __syncthreads();
        // load w tile
        for (int i = threadIdx.y; i < BN; i += blockDim.y) {
            for (int j = threadIdx.x; j < BK; j += blockDim.x) {
                int b_row = block_n + i;
                int b_col = block_k + j;
                if (b_row < N && b_col < K) {
                    Bs[i * BK + j] = __float2half(w[b_row * K + b_col]);
                }
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL_WARP_N;
            int b_col = warp_k;
            wmma::load_matrix_sync(b_frag[warp_n], &Bs[b_row * BK + b_col], BK);
            // wmma mma
            wmma::mma_sync(c_frag[warp_id], a_frag[warp_m], b_frag[warp_n], c_frag[warp_id]);
            // store y fragment
        }
        __syncthreads();
    }
    // store y fragments to y

    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m  = warp_id / NUM_MATMUL_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL_WARP_M;
    int c_col   = block_n + warp_n * MATMUL_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL_WARP_M * BN + warp_n * MATMUL_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }
#pragma unroll


    for (int i = threadIdx.y; i < BM; i += blockDim.y) {
        for (int j = threadIdx.x; j < BN; j += blockDim.x) {
            int c_row = block_m + i;
            int c_col = block_n + j;
            if (c_row < M && c_col < N) {
                // Cs[i * BN + j]       = Cs[i * BN + j] + __float2half(b[c_col]);
                float vv = v[c_row * N + c_col] + __half2float(Cs[i * BN + j]) + b[c_col];
                int   f  = vv >= 1.0f;
                y[c_row * N + c_col] = f;
                v[c_row * N + c_col] = f ? 0 : vv;
            }
        }
    }
}