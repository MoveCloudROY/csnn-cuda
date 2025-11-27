#include "linear.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// x: row major, w: col major, y: row major
// __global__ void
// linear_fuse_if(int M, int N, int K, half alpha, half* x, half* w, half beta, half* y) {

__global__ void linear_forward(
    float* x, // [N, In]
    float* w, // [Out, In]
    float* b, // [Out]
    float*       y, // [N, Out]
    int M, int N, int K
) {
    const int BM      = MATMUL4_BLOCK_M;
    const int BN      = MATMUL4_BLOCK_N;
    const int BK      = MATMUL4_BLOCK_K;
    int       block_m = blockIdx.x * BM;
    int       block_n = blockIdx.y * BN;

    // prepare fragments
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
            a_frag[NUM_MATMUL4_WARPS_M];
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
            b_frag[NUM_MATMUL4_WARPS_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        c_frag[NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N];
    // 缓存A_tile和B_tile
    __shared__ half As[MATMUL4_BLOCK_M * MATMUL4_BLOCK_K];
    __shared__ half Bs[MATMUL4_BLOCK_N * MATMUL4_BLOCK_K];
    __shared__ half Cs[MATMUL4_BLOCK_M * MATMUL4_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        int threadid = (threadIdx.y * blockDim.x + threadIdx.x);
        // load x tile
        // float4 tmp = FETCH_FLOAT4(&A[]);
        for (int i = 0; i < BM * BK / 4; i += blockDim.x * blockDim.y ) {
            int index = threadid + i;
            int a_row = (index * 4) / BK;
            int a_col = (index * 4) % BK;
            if (index < BM * BK / 4) {
                float4 tmp = FETCH_FLOAT4(x[(block_m + a_row) * K + block_k + a_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&As[a_row * BK + a_col]) = h2_0;
                *((half2*)&As[a_row * BK + a_col+ 2]) = h2_1;
            }
        }

        __syncthreads();
        // load w tile
        for (int i = 0; i < BN * BK / 4; i += blockDim.x * blockDim.y) {
            int index = threadid + i;
            int b_row = (index * 4) / BK;
            int b_col = (index * 4) % BK;
            if (index < BN * BK / 4) {
                float4 tmp = FETCH_FLOAT4(w[(block_n + b_row) * K + block_k + b_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&Bs[b_row * BK + b_col]) = h2_0;
                *((half2*)&Bs[b_row * BK + b_col+ 2]) = h2_1;
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL4_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL4_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL4_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL4_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL4_WARP_N;
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
    int warp_m  = warp_id / NUM_MATMUL4_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL4_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL4_WARP_M;
    int c_col   = block_n + warp_n * MATMUL4_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL4_WARP_M * BN + warp_n * MATMUL4_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }

    __syncthreads();
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
    float* x, // [N, In]
    float* w, // [Out, In]
    float* b, // [Out]
    float*       y, // [N, Out]
    float*       v, // [N, Out]
    int M, int N, int K
) {

    const int BM      = MATMUL4_BLOCK_M;
    const int BN      = MATMUL4_BLOCK_N;
    const int BK      = MATMUL4_BLOCK_K;
    int       block_m = blockIdx.x * BM;
    int       block_n = blockIdx.y * BN;

    // prepare fragments
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
            a_frag[NUM_MATMUL4_WARPS_M];
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
            b_frag[NUM_MATMUL4_WARPS_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        c_frag[NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N];
    // 缓存A_tile和B_tile
    __shared__ half As[MATMUL4_BLOCK_M * MATMUL4_BLOCK_K];
    __shared__ half Bs[MATMUL4_BLOCK_N * MATMUL4_BLOCK_K];
    __shared__ half Cs[MATMUL4_BLOCK_M * MATMUL4_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        int threadid = (threadIdx.y * blockDim.x + threadIdx.x);
        // load x tile
        // float4 tmp = FETCH_FLOAT4(&A[]);
        for (int i = 0; i < BM * BK / 4; i += blockDim.x * blockDim.y ) {
            int index = threadid + i;
            int a_row = (index * 4) / BK;
            int a_col = (index * 4) % BK;
            if (index < BM * BK / 4) {
                float4 tmp = FETCH_FLOAT4(x[(block_m + a_row) * K + block_k + a_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&As[a_row * BK + a_col]) = h2_0;
                *((half2*)&As[a_row * BK + a_col+ 2]) = h2_1;
            }
        }

        __syncthreads();
        // load w tile
        for (int i = 0; i < BN * BK / 4; i += blockDim.x * blockDim.y) {
            int index = threadid + i;
            int b_row = (index * 4) / BK;
            int b_col = (index * 4) % BK;
            if (index < BN * BK / 4) {
                float4 tmp = FETCH_FLOAT4(w[(block_n + b_row) * K + block_k + b_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&Bs[b_row * BK + b_col]) = h2_0;
                *((half2*)&Bs[b_row * BK + b_col+ 2]) = h2_1;
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL4_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL4_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL4_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL4_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL4_WARP_N;
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
    int warp_m  = warp_id / NUM_MATMUL4_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL4_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL4_WARP_M;
    int c_col   = block_n + warp_n * MATMUL4_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL4_WARP_M * BN + warp_n * MATMUL4_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }
    __syncthreads();
    // store y fragments to y
    int threadid = (threadIdx.y * blockDim.x + threadIdx.x);
    #pragma unroll
    for (int i = threadid; i < BM * BN / 4; i += blockDim.x * blockDim.y ) {
        int index = i;
        int c_row = (index * 4) / BN;
        int c_col = (index * 4) % BN;
        int gc_row = block_m + c_row;
        int gc_col = block_n + c_col;
        // printf("threadid=%d, i=%d, c_row=%d, c_col=%d, gc_row=%d, gc_col=%d\n", threadid, i, c_row, c_col, gc_row, gc_col);
        if (gc_row < M && gc_col + 3 < N) {
            float4 b4 = FETCH_FLOAT4(b[gc_col]);
            float4 v4 = FETCH_FLOAT4(v[gc_row * N + gc_col]);
            v4.x += __half2float(Cs[c_row * BN + c_col]) + b4.x;
            v4.y += __half2float(Cs[c_row * BN + c_col + 1]) + b4.y;
            v4.z += __half2float(Cs[c_row * BN + c_col + 2]) + b4.z;
            v4.w += __half2float(Cs[c_row * BN + c_col + 3]) + b4.w;
            float4 s;
            s.x = (v4.x >= 1.0f)? 1.0f : 0.0f;
            s.y = (v4.y >= 1.0f)? 1.0f : 0.0f;
            s.z = (v4.z >= 1.0f)? 1.0f : 0.0f;
            s.w = (v4.w >= 1.0f)? 1.0f : 0.0f;
            v4.x *= (1. - s.x);
            v4.y *= (1. - s.y);
            v4.z *= (1. - s.z);
            v4.w *= (1. - s.w);
            // FETCH_FLOAT4(v[gc_row * N + gc_col]) = s;
            float4* addr_v = (float4 *)&v[gc_row * N + gc_col];
            addr_v[0] = v4;
            // v[gc_row * N + gc_col] = v4.x;
            // v[gc_row * N + gc_col + 1] = v4.y;
            // v[gc_row * N + gc_col + 2] = v4.z;
            // v[gc_row * N + gc_col + 3] = v4.w;

            // float4 tmp = {__half2float(,
            //             __half2float(Cs[c_row * BN + c_col + 1]) + b4.y,
            //             __half2float(Cs[c_row * BN + c_col + 2]) + b4.z,
            //             __half2float(Cs[c_row * BN + c_col + 3]) + b4.w};
            float4 * addr_y = (float4 *)&y[gc_row * N + gc_col];
            addr_y[0] = s;
            // FETCH_FLOAT4(y[(gc_row) * N + gc_col]) = s;
            // y[gc_row * N + gc_col] = s.x;
            // y[gc_row * N + gc_col + 1] = s.y;
            // y[gc_row * N + gc_col + 2] = s.z;
            // y[gc_row * N + gc_col + 3] = s.w;

        }
    }

}

__global__ void linear_01x_fuse_if(
    const float* __restrict__ x, // [N, In], 0/1
    const float* __restrict__ w, // [Out, In] in logical shape, ROW-major in memory
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    float* __restrict__ v,       // [N, Out]
    int N, int Out, int In
)
{
    // 优化版：针对 0/1 脉冲输入
    // - 使用 shared memory 缓存 x[n, :] (所有线程共享)
    // - 每个线程处理一个输出神经元，直接从 global memory 以 float4 读取权重
    // - 显式向量化加载和计算

    int n = blockIdx.y; // batch 样本索引
    if (n >= N) return;

    int o = blockIdx.x * blockDim.x + threadIdx.x; // 输出神经元索引
    if (o >= Out) return;

    const int VEC = 4; // float4 向量化

    // Shared memory 只缓存 x 行
    extern __shared__ float x_s[];

    const float* x_row = x + n * In;    // x[n, :]
    const float* w_row = w + o * In;    // W[o, :] 权重行

    // 1) 协作加载 x[n, :] 到 shared memory (float4 方式)
    int In4_aligned = (In / VEC) * VEC;
    for (int k = threadIdx.x * VEC; k < In4_aligned; k += blockDim.x * VEC) {
        if (k + VEC <= In) {
            float4 v4 = *reinterpret_cast<const float4*>(&x_row[k]);
            *reinterpret_cast<float4*>(&x_s[k]) = v4;
        }
    }
    // 处理非对齐尾部
    for (int k = In4_aligned + threadIdx.x; k < In; k += blockDim.x) {
        x_s[k] = x_row[k];
    }

    __syncthreads();

    // 2) 每个线程独立计算自己的输出神经元
    float acc = 0.0f;

    // 向量化累加：从 global memory 读取权重，从 shared memory 读取输入
    #pragma unroll 8
    for (int k = 0; k < In4_aligned; k += VEC) {
        float4 xv4 = *reinterpret_cast<float4*>(&x_s[k]);
        float4 wv4 = *reinterpret_cast<const float4*>(&w_row[k]);
        
        acc = fmaf(xv4.x, wv4.x, acc);
        acc = fmaf(xv4.y, wv4.y, acc);
        acc = fmaf(xv4.z, wv4.z, acc);
        acc = fmaf(xv4.w, wv4.w, acc);
    }

    // 处理尾部（非 4 对齐部分）
    for (int k = In4_aligned; k < In; ++k) {
        acc = fmaf(x_s[k], w_row[k], acc);
    }

    acc += b[o];

    // IF 节点：整合并触发脉冲
    int idx = n * Out + o;
    float vm = v[idx] + acc;
    float spike = (vm >= 1.0f) ? 1.0f : 0.0f;
    y[idx] = spike;
    v[idx] = vm * (1.0f - spike);
}


__global__ void linear_fuse_argmax10(
    float* __restrict__ x, // [N, In]
    float* __restrict__ w, // [Out, In]
    float* __restrict__ b, // [Out]
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
// #pragma unroll
    // for (int i = threadIdx.y; i < BM; i += blockDim.y) {
    //     for (int j = threadIdx.x; j < BN; j += blockDim.x) {
    //         int c_row = block_m + i;
    //         int c_col = block_n + j;
    //         if (c_row < M && c_col < N) {
    //             // Cs[i * BN + j]       = Cs[i * BN + j] + __float2half(b[c_col]);
    //             float vv = v[c_row * N + c_col] + __half2float(Cs[i * BN + j]) + b[c_col];
    //             int   f  = vv >= 1.0f;
    //             y[c_row * N + c_col] = f;
    //             v[c_row * N + c_col] = f ? 0 : vv;
    //         }
    //     }
    // }
}