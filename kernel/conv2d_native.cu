#include "common.cuh"
#include "conv2d_native.cuh"
#include "kernel.cuh"
#include <iostream>


__device__ __constant__ float d_Conv1_ker[CONV_KERNEL_K * CONV_KERNEL_K];
__device__ __constant__ float d_Conv1_b[CONV1_Co];

void conv2d_c1_k5_native_wrapper(
    float* __restrict__ x, // [N, 1, 28, 28]
    float*  w, // [Co=8, 1, K=5, K=5]
    float*  b, // [Co=8]
    float* __restrict__ y,       // [N, Co=8, 24, 24]
    int N, int Co
) {
    cudaMemcpyToSymbol(d_Conv1_ker, &w, CONV_KERNEL_K * CONV_KERNEL_K,0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(d_Conv1_b, &b, CONV1_IN_SIZE ,0,cudaMemcpyDeviceToDevice);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("conv1 w b initialize failed: %s\n", cudaGetErrorString(err));

    const dim3 block(CONV1_THREAD_PER_BLOCK);
    const dim3 grid(8, 100);
    conv2d_c1_k5_native<<<grid, block>>>(
        x,
        w,
        b,
        y,
        N,
        Co
    );
}

__inline__ __device__
float warpReduceSum(float val) {
    const unsigned FULL_MASK = 0xffffffffu;
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// __constant__ float d_Conv1_ker[CONV_KERNEL_K * CONV_KERNEL_K];
// __constant__ float d_Conv1_b[CONV1_Co];

__global__ void conv2d_c1_k5_native(
    float* __restrict__ x, // [N, 1, 28, 28]
    float* __restrict__ w, // [Co=8, 1, K=5, K=5]
    float* __restrict__ b, // [Co=8]
    float* __restrict__ y,       // [N, Co=8, 24, 24]
    int N, int Co
) {
    __shared__ __half tile[CONV1_SHARED_M * CONV1_SHARED_N];
    __shared__ __half result[CONV1_CALC_M_PER_WARP * CONV1_CALC_N_PER_WARP];
    // int by = blockIdx.y;
    for (int n = blockIdx.y; n < N; n += gridDim.y) {
        int Co = blockIdx.x; // Co
        // for (int n = blockIdx.x; n < N;)
        const int tx    = threadIdx.x;
        int warp_id = tx >> 5;
        int lane_id = tx & 31;
        // this warp process area begin at (g_row, g_col)
        int w_row = warp_id / CONV1_WARP_N;
        int w_col = warp_id % CONV1_WARP_N;
        // Load input tile into shared memory, ld128
        constexpr int thread_per_row = CONV1_SHARED_N / 4;
        
        int row =  lane_id / thread_per_row;
        int col = (lane_id % thread_per_row) * 4;
        // 8 * 12 / 4 = 24 threads to load total tile, so just check row
        if (row < CONV1_SHARED_M) {
            int g_row = w_row * CONV1_CALC_M_PER_WARP;
            int g_col = w_col * CONV1_CALC_N_PER_WARP;

            // FETCH_FLOAT4(tile[row * CONV1_SHARED_N + col]) = 
            int id = row * CONV1_SHARED_N + col;
            float4 x4 = FETCH_FLOAT4(x[n * CONV1_IN_SIZE + (g_row + row) * CONV1_Wi + (g_col + col)]);
            half2 h2_0 = __float22half2_rn(make_float2(x4.x, x4.y));
            half2 h2_1 = __float22half2_rn(make_float2(x4.z, x4.w));
            *((half2*)&tile[id]) = h2_0;
            *((half2*)&tile[id + 2]) = h2_1;
        }
        __syncthreads();
        // Compute convolution
        for (int orow = 0; orow < CONV1_CALC_M_PER_WARP; ++orow) {
            for (int ocol = 0; ocol < CONV1_CALC_N_PER_WARP; ++ocol) {
                // result[orow * CONV1_CALC_N_PER_WARP + ocol] = __float2half(0.0f);
                // warp reduce
                int crow = lane_id / CONV1_KENREL_SIZE;
                int ccol = lane_id % CONV1_KENREL_SIZE;
                float acc = 0.0f;
                if (crow < CONV1_KENREL_SIZE) {
                    float wv = d_Conv1_ker[crow * CONV1_KENREL_SIZE + ccol];
                    float xv = __half2float(tile[(crow+orow) * CONV1_SHARED_N + ccol + ocol]);
                    acc = xv * wv;
                }
                
                acc = warpReduceSum(acc);
                acc += d_Conv1_b[Co];
                result[orow * CONV1_CALC_N_PER_WARP + ocol] = __float2half(acc);
                
            }
        }
        __syncthreads();
        // write to y
        for (int orow = 0; orow < CONV1_CALC_M_PER_WARP; ++orow) {
            for (int ocol = 0; ocol < CONV1_CALC_N_PER_WARP; ++ocol) {
                int g_row = w_row * CONV1_CALC_M_PER_WARP + orow;
                int g_col = w_col * CONV1_CALC_N_PER_WARP + ocol;
                float v = __half2float(result[orow * CONV1_CALC_N_PER_WARP + ocol]);
                y[(n * Co + Co) * CONV1_Ho * CONV1_Wo + g_row * CONV1_Wo + g_col] = v;
            }
        }
    }
    


//     if (n >= N)
//         return;
//     const int ow0 = blockIdx.x * blockDim.x;
//     const int oh0 = blockIdx.y * blockDim.y;
//     for (int yy = ty; yy < tileH; yy += blockDim.y) {
//         int ih = oh0 + yy;
//         for (int xx = tx; xx < tileW; xx += blockDim.x) {
//             int   iw = ow0 + xx;
//             float v  = 0.0f;
//             if (ih >= 0 && ih < CONV1_Hi && iw >= 0 && iw < CONV1_Wi) {
//                 const int x_idx = ((n * 1 + 0) * CONV1_Hi + ih) * CONV1_Wi + iw;
//                 v               = __ldg(x + x_idx);
//             }
//             tile[yy * tileW + xx] = v;
//         }
//     }
//     __syncthreads();
//     if (oh < CONV1_Ho && ow < CONV1_Wo) {
//         const float* w_oc = w + oc * (CONV_KERNEL_K * CONV_KERNEL_K);
//         float        acc  = __ldg(b + oc);
// #pragma unroll
//         for (int ky = 0; ky < CONV_KERNEL_K; ++ky) {
// #pragma unroll
//             for (int kx = 0; kx < CONV_KERNEL_K; ++kx) {
//                 float xv = tile[(ty + ky) * tileW + (tx + kx)];
//                 float ww = __ldg(w_oc + ky * CONV_KERNEL_K + kx);
//                 acc      = fmaf(xv, ww, acc);
//             }
//         }
//         const int y_idx = ((n * Co + oc) * CONV1_Ho + oh) * CONV1_Wo + ow;
//         y[y_idx]        = acc;
//     }
}



__global__ void conv2d_nchw_native_compress(
    float* __restrict__ x, // [N, Ci=8, Hi=12, Wi=12]
    float* __restrict__ w, // [Co=16, Ci=8, K=5, K=5]
    float* __restrict__ b, // [Co=16]
    float* __restrict__ y,       // [N, Co=16, Ho=8, Wo=8]
    int N, int Ci, int Hi, int Wi, int Co
) {

}